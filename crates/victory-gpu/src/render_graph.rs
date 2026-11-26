//! Render-graph pass classification, stand-in accounting, and calibrated D3D12
//! hardware GPU timestamp telemetry (`ID3D12QueryHeap`).
//!
//! Why this exists (REA-003 Frente A):
//! - Synchronous CPU wait clocks (`SUBMISSION_US[1]`) measure CPU thread blocking,
//!   not actual GPU execution time — and read `0.0 ms` under `VICTORY_RELAY=1`
//!   when submissions land on the background relay worker.
//! - [`GpuTimestampRing`] brackets every `ExecuteCommandLists` batch with lightweight
//!   `BeginQuery`/`EndQuery` (`D3D12_QUERY_TYPE_TIMESTAMP`) + `ResolveQueryData`
//!   into a persistent readback buffer and converts GPU ticks via
//!   `ID3D12CommandQueue::GetTimestampFrequency`.
//! - [`GpuTimeStatus`] makes measurement provenance explicit so unmeasured windows
//!   are reported as `pending` / `wait_clock_only` / `unavailable` rather than
//!   misleading `0.0 ms` GPU timings.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::core::Interface;

use crate::pipeline::PipelineMiss;

/// Number of in-flight submission timestamp slots in the ring buffer.
const SLOT_COUNT: usize = 256;
/// Queries per submission slot (start + end timestamp).
const QUERIES_PER_SLOT: usize = 2;
/// Total queries in the D3D12 timestamp query heap.
const TOTAL_QUERIES: u32 = (SLOT_COUNT * QUERIES_PER_SLOT) as u32;

/// Provenance status of `gpu_ms` / `gpu_hw_ms` for a telemetry window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GpuTimeStatus {
    /// Calibrated hardware timestamps were resolved from the D3D12 command queue.
    Measured,
    /// GPU batches were submitted asynchronously and have not resolved yet.
    Pending,
    /// Hardware timestamps had no resolved samples, only synchronous wait clocks.
    WaitClockOnly,
    /// No submissions or timestamp ring unavailable in this window.
    Unavailable,
}

impl GpuTimeStatus {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Measured => "measured",
            Self::Pending => "pending",
            Self::WaitClockOnly => "wait_clock_only",
            Self::Unavailable => "unavailable",
        }
    }
}

/// Stand-in / stub provenance for a translated console shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StandInKind {
    Ambient,
    ShadowMask,
    Exposure,
}

/// High-level render-graph pass category classified from pipeline metadata.
#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PassKind {
    GraphicsColor = 0,
    DepthPrepass = 1,
    MeshGeometry = 2,
    RectPass = 3,
    Compute = 4,
    IndirectDraw = 5,
    IndirectDispatch = 6,
    StandInAmbient = 7,
    StandInShadowMask = 8,
    StandInExposure = 9,
}

const PASS_KIND_COUNT: usize = 10;

static RING_AVAILABLE: AtomicBool = AtomicBool::new(false);
static HW_TIME_US: AtomicU64 = AtomicU64::new(0);
static HW_SAMPLES: AtomicU64 = AtomicU64::new(0);
static HW_RAW_TICKS: AtomicU64 = AtomicU64::new(0);
static LAST_FREQ_HZ: AtomicU64 = AtomicU64::new(0);
static SUBMITTED_BRACKETS: AtomicU64 = AtomicU64::new(0);

static PASS_COUNTS: [AtomicU64; PASS_KIND_COUNT] = [const { AtomicU64::new(0) }; PASS_KIND_COUNT];
static STANDIN_AMBIENT: AtomicU64 = AtomicU64::new(0);
static STANDIN_SHADOWMASK: AtomicU64 = AtomicU64::new(0);
static STANDIN_EXPOSURE: AtomicU64 = AtomicU64::new(0);
static DROPPED_NULL: AtomicU64 = AtomicU64::new(0);
static DROPPED_PENDING: AtomicU64 = AtomicU64::new(0);
static DROPPED_FAILED: AtomicU64 = AtomicU64::new(0);
static DROPPED_OTHER: AtomicU64 = AtomicU64::new(0);
static DROPPED_INDIRECT_ONLY: AtomicU64 = AtomicU64::new(0);
static SUBMITTED_DRAWS: AtomicU64 = AtomicU64::new(0);
static SUBMITTED_DISPATCHES: AtomicU64 = AtomicU64::new(0);

/// Classifies a pipeline's [`PassKind`] and stand-in status from host metadata.
pub(crate) fn classify_pipeline(
    compute: bool,
    mesh: bool,
    rects: bool,
    num_render_targets: u32,
    standin: Option<StandInKind>,
) -> PassKind {
    if let Some(kind) = standin {
        return match kind {
            StandInKind::Ambient => PassKind::StandInAmbient,
            StandInKind::ShadowMask => PassKind::StandInShadowMask,
            StandInKind::Exposure => PassKind::StandInExposure,
        };
    }
    if compute {
        PassKind::Compute
    } else if mesh {
        PassKind::MeshGeometry
    } else if rects {
        PassKind::RectPass
    } else if num_render_targets == 0 {
        PassKind::DepthPrepass
    } else {
        PassKind::GraphicsColor
    }
}

/// Converts raw D3D12 timestamp delta ticks and frequency (Hz) to microseconds.
pub(crate) fn ticks_to_us(delta_ticks: u64, frequency_hz: u64) -> u64 {
    if frequency_hz == 0 {
        return 0;
    }
    ((u128::from(delta_ticks) * 1_000_000) / u128::from(frequency_hz)) as u64
}

/// Determines the honest GPU timing provenance status for a telemetry window.
pub(crate) fn classify_gpu_time_status(
    hw_samples: u64,
    wait_gpu_ms: f64,
    submitted_count: u64,
    ring_available: bool,
) -> GpuTimeStatus {
    if hw_samples > 0 && ring_available {
        GpuTimeStatus::Measured
    } else if wait_gpu_ms > 0.000_5 {
        GpuTimeStatus::WaitClockOnly
    } else if submitted_count > 0 {
        GpuTimeStatus::Pending
    } else {
        GpuTimeStatus::Unavailable
    }
}

/// Marks whether a host [`GpuTimestampRing`] is active on the device.
pub(crate) fn set_ring_available(available: bool) {
    RING_AVAILABLE.store(available, Ordering::Relaxed);
}

/// Records one host execution against pass and stand-in telemetry counters.
pub(crate) fn record_pipeline_exec(
    pass: PassKind,
    standin: Option<StandInKind>,
    is_compute: bool,
    indirect: bool,
) {
    PASS_COUNTS[pass as usize].fetch_add(1, Ordering::Relaxed);
    if is_compute {
        SUBMITTED_DISPATCHES.fetch_add(1, Ordering::Relaxed);
    } else {
        SUBMITTED_DRAWS.fetch_add(1, Ordering::Relaxed);
    }
    if indirect {
        let bucket = if is_compute {
            PassKind::IndirectDispatch
        } else {
            PassKind::IndirectDraw
        };
        PASS_COUNTS[bucket as usize].fetch_add(1, Ordering::Relaxed);
    }
    if let Some(kind) = standin {
        match kind {
            StandInKind::Ambient => {
                STANDIN_AMBIENT.fetch_add(1, Ordering::Relaxed);
            }
            StandInKind::ShadowMask => {
                STANDIN_SHADOWMASK.fetch_add(1, Ordering::Relaxed);
            }
            StandInKind::Exposure => {
                STANDIN_EXPOSURE.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// Records one dropped draw/dispatch caused by a missing or refused PSO.
pub(crate) fn record_dropped_pso(miss: PipelineMiss) {
    match miss {
        PipelineMiss::Null => {
            DROPPED_NULL.fetch_add(1, Ordering::Relaxed);
        }
        PipelineMiss::Pending => {
            DROPPED_PENDING.fetch_add(1, Ordering::Relaxed);
        }
        PipelineMiss::Failed => {
            DROPPED_FAILED.fetch_add(1, Ordering::Relaxed);
        }
        PipelineMiss::Other => {
            DROPPED_OTHER.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Records one dropped indirect draw/dispatch caused by non-PSO reasons
/// (argument bounds, missing signature, or mesh skip).
pub(crate) fn record_dropped_indirect_only() {
    DROPPED_INDIRECT_ONLY.fetch_add(1, Ordering::Relaxed);
}

/// Snapshot of calibrated GPU timing, stand-in activity, and pass breakdowns
/// over one telemetry reporting window.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RenderGraphSnapshot {
    pub(crate) window_frames: u64,
    pub(crate) gpu_time_status: GpuTimeStatus,
    pub(crate) gpu_hw_total_ms: f64,
    pub(crate) gpu_hw_ms: f64,
    pub(crate) gpu_hw_ms_per_frame: f64,
    pub(crate) gpu_hw_ms_per_sample: f64,
    pub(crate) gpu_hw_samples: u64,
    pub(crate) gpu_timestamp_freq_hz: u64,
    pub(crate) standin_draws: f64,
    pub(crate) standin_window_total: u64,
    pub(crate) standin_ambient: u64,
    pub(crate) standin_shadowmask: u64,
    pub(crate) standin_exposure: u64,
    pub(crate) dropped_pso_draws: f64,
    pub(crate) dropped_window_total: u64,
    pub(crate) dropped_null: u64,
    pub(crate) dropped_pending: u64,
    pub(crate) dropped_failed: u64,
    pub(crate) dropped_other: u64,
    pub(crate) dropped_indirect_only: u64,
    pub(crate) submitted_draws: u64,
    pub(crate) submitted_dispatches: u64,
    pub(crate) graphics_color: u64,
    pub(crate) depth_prepass: u64,
    pub(crate) mesh_geometry: u64,
    pub(crate) rect_pass: u64,
    pub(crate) compute: u64,
    pub(crate) indirect_draw: u64,
    pub(crate) indirect_dispatch: u64,
}

impl RenderGraphSnapshot {
    /// Formats calibrated timestamp, stand-in, PSO drop, and pass counters as JSON object fields.
    pub(crate) fn json_fields(&self) -> String {
        format!(
            "\"window_frames\":{},\"gpu_time_status\":\"{}\",\"gpu_hw_total_ms\":{:.3},\
             \"gpu_hw_ms\":{:.3},\"gpu_hw_ms_per_frame\":{:.3},\"gpu_hw_ms_per_sample\":{:.3},\
             \"gpu_hw_samples\":{},\"gpu_ts_freq_hz\":{},\
             \"standin_draws\":{:.2},\"standin_window_total\":{},\"standin_ambient\":{},\"standin_shadowmask\":{},\"standin_exposure\":{},\
             \"dropped_pso_draws\":{:.2},\"dropped_window_total\":{},\"dropped_null\":{},\"dropped_pending\":{},\"dropped_failed\":{},\"dropped_other\":{},\"dropped_indirect_only\":{},\
             \"submitted_draws\":{},\"submitted_dispatches\":{},\
             \"rg_gfx\":{},\"rg_depth\":{},\"rg_mesh\":{},\"rg_rect\":{},\"rg_cs\":{},\"rg_ind_draw\":{},\"rg_ind_disp\":{}",
            self.window_frames,
            self.gpu_time_status.as_str(),
            self.gpu_hw_total_ms,
            self.gpu_hw_ms,
            self.gpu_hw_ms_per_frame,
            self.gpu_hw_ms_per_sample,
            self.gpu_hw_samples,
            self.gpu_timestamp_freq_hz,
            self.standin_draws,
            self.standin_window_total,
            self.standin_ambient,
            self.standin_shadowmask,
            self.standin_exposure,
            self.dropped_pso_draws,
            self.dropped_window_total,
            self.dropped_null,
            self.dropped_pending,
            self.dropped_failed,
            self.dropped_other,
            self.dropped_indirect_only,
            self.submitted_draws,
            self.submitted_dispatches,
            self.graphics_color,
            self.depth_prepass,
            self.mesh_geometry,
            self.rect_pass,
            self.compute,
            self.indirect_draw,
            self.indirect_dispatch,
        )
    }
}

/// Atomically swaps per-window render-graph counters and computes averages for `window_frames`.
pub(crate) fn take_window_snapshot(window_frames: f64, wait_gpu_ms: f64) -> RenderGraphSnapshot {
    let frames = window_frames.max(1.0);
    let window_frames_u64 = frames.round() as u64;
    let hw_us = HW_TIME_US.swap(0, Ordering::Relaxed);
    let hw_samples = HW_SAMPLES.swap(0, Ordering::Relaxed);
    let _hw_ticks = HW_RAW_TICKS.swap(0, Ordering::Relaxed);
    let submitted = SUBMITTED_BRACKETS.swap(0, Ordering::Relaxed);
    let ring_available = RING_AVAILABLE.load(Ordering::Relaxed);
    let gpu_timestamp_freq_hz = LAST_FREQ_HZ.load(Ordering::Relaxed);
    let gpu_time_status =
        classify_gpu_time_status(hw_samples, wait_gpu_ms, submitted, ring_available);

    let gpu_hw_total_ms = hw_us as f64 / 1000.0;
    let gpu_hw_ms = gpu_hw_total_ms / frames;
    let gpu_hw_ms_per_sample = if hw_samples > 0 {
        gpu_hw_total_ms / hw_samples as f64
    } else {
        0.0
    };

    let standin_ambient = STANDIN_AMBIENT.swap(0, Ordering::Relaxed);
    let standin_shadowmask = STANDIN_SHADOWMASK.swap(0, Ordering::Relaxed);
    let standin_exposure = STANDIN_EXPOSURE.swap(0, Ordering::Relaxed);
    let standin_window_total = standin_ambient + standin_shadowmask + standin_exposure;

    let dropped_null = DROPPED_NULL.swap(0, Ordering::Relaxed);
    let dropped_pending = DROPPED_PENDING.swap(0, Ordering::Relaxed);
    let dropped_failed = DROPPED_FAILED.swap(0, Ordering::Relaxed);
    let dropped_other = DROPPED_OTHER.swap(0, Ordering::Relaxed);
    let dropped_window_total = dropped_null + dropped_pending + dropped_failed + dropped_other;
    let dropped_indirect_only = DROPPED_INDIRECT_ONLY.swap(0, Ordering::Relaxed);
    let submitted_draws = SUBMITTED_DRAWS.swap(0, Ordering::Relaxed);
    let submitted_dispatches = SUBMITTED_DISPATCHES.swap(0, Ordering::Relaxed);

    let graphics_color = PASS_COUNTS[PassKind::GraphicsColor as usize].swap(0, Ordering::Relaxed);
    let depth_prepass = PASS_COUNTS[PassKind::DepthPrepass as usize].swap(0, Ordering::Relaxed);
    let mesh_geometry = PASS_COUNTS[PassKind::MeshGeometry as usize].swap(0, Ordering::Relaxed);
    let rect_pass = PASS_COUNTS[PassKind::RectPass as usize].swap(0, Ordering::Relaxed);
    let compute = PASS_COUNTS[PassKind::Compute as usize].swap(0, Ordering::Relaxed);
    let indirect_draw = PASS_COUNTS[PassKind::IndirectDraw as usize].swap(0, Ordering::Relaxed);
    let indirect_dispatch =
        PASS_COUNTS[PassKind::IndirectDispatch as usize].swap(0, Ordering::Relaxed);
    let _ = PASS_COUNTS[PassKind::StandInAmbient as usize].swap(0, Ordering::Relaxed);
    let _ = PASS_COUNTS[PassKind::StandInShadowMask as usize].swap(0, Ordering::Relaxed);
    let _ = PASS_COUNTS[PassKind::StandInExposure as usize].swap(0, Ordering::Relaxed);

    RenderGraphSnapshot {
        window_frames: window_frames_u64,
        gpu_time_status,
        gpu_hw_total_ms,
        gpu_hw_ms,
        gpu_hw_ms_per_frame: gpu_hw_ms,
        gpu_hw_ms_per_sample,
        gpu_hw_samples: hw_samples,
        gpu_timestamp_freq_hz,
        standin_draws: standin_window_total as f64 / frames,
        standin_window_total,
        standin_ambient,
        standin_shadowmask,
        standin_exposure,
        dropped_pso_draws: dropped_window_total as f64 / frames,
        dropped_window_total,
        dropped_null,
        dropped_pending,
        dropped_failed,
        dropped_other,
        dropped_indirect_only,
        submitted_draws,
        submitted_dispatches,
        graphics_color,
        depth_prepass,
        mesh_geometry,
        rect_pass,
        compute,
        indirect_draw,
        indirect_dispatch,
    }
}

struct TimestampSlot {
    begin_alloc: ID3D12CommandAllocator,
    begin_list: ID3D12GraphicsCommandList,
    end_alloc: ID3D12CommandAllocator,
    end_list: ID3D12GraphicsCommandList,
    /// Fence value signalled after this slot's batch, or `0` when free.
    fence_value: u64,
    submission_id: u64,
}

struct RingState {
    slots: Vec<TimestampSlot>,
    next_slot: usize,
    readback_ptr: *const u64,
}

/// Prepared begin/end command lists ready to bracket one `ExecuteCommandLists`.
pub(crate) struct TimestampBracket {
    pub(crate) slot_idx: usize,
    pub(crate) begin_cmd: ID3D12CommandList,
    pub(crate) end_cmd: ID3D12CommandList,
}

/// Asynchronous D3D12 hardware timestamp ring bracketing direct queue submissions.
pub(crate) struct GpuTimestampRing {
    query_heap: ID3D12QueryHeap,
    readback: ID3D12Resource,
    frequency_hz: u64,
    state: Mutex<RingState>,
}

// SAFETY: D3D12 query heaps, resources, allocators and closed command lists are
// free-threaded; mutable slot recycling and readback reading are serialized under `state`.
unsafe impl Send for GpuTimestampRing {}
// SAFETY: as above.
unsafe impl Sync for GpuTimestampRing {}

impl Drop for GpuTimestampRing {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.state.lock()
            && !guard.readback_ptr.is_null()
        {
            // SAFETY: `readback` was mapped persistently at construction; unmap once on drop.
            unsafe {
                self.readback.Unmap(0, None);
            }
            guard.readback_ptr = std::ptr::null();
        }
    }
}

impl GpuTimestampRing {
    /// Creates a `SLOT_COUNT`-entry direct-queue timestamp ring on `device`.
    pub(crate) fn new(
        device: &ID3D12Device,
        queue: &ID3D12CommandQueue,
    ) -> windows::core::Result<Self> {
        // SAFETY: plain D3D12 query on the direct queue.
        let frequency_hz = unsafe { queue.GetTimestampFrequency() }?;
        if frequency_hz == 0 {
            return Err(windows::core::Error::from_hresult(
                windows::Win32::Foundation::E_FAIL,
            ));
        }
        LAST_FREQ_HZ.store(frequency_hz, Ordering::Relaxed);

        let heap_desc = D3D12_QUERY_HEAP_DESC {
            Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
            Count: TOTAL_QUERIES,
            NodeMask: 0,
        };
        let mut query_heap: Option<ID3D12QueryHeap> = None;
        // SAFETY: valid query heap description.
        unsafe { device.CreateQueryHeap(&heap_desc, &mut query_heap) }?;
        let query_heap = query_heap.ok_or_else(windows::core::Error::empty)?;

        let buf_bytes = (TOTAL_QUERIES as usize * size_of::<u64>()) as u64;
        let readback_props = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let buf_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: buf_bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut readback: Option<ID3D12Resource> = None;
        // SAFETY: valid readback buffer creation for query resolve target.
        unsafe {
            device.CreateCommittedResource(
                &readback_props,
                D3D12_HEAP_FLAG_NONE,
                &buf_desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut readback,
            )
        }?;
        let readback = readback.ok_or_else(windows::core::Error::empty)?;

        let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: D3D12 readback buffers allow persistent CPU read mapping across GPU writes.
        unsafe { readback.Map(0, None, Some(&mut mapped)) }?;
        let readback_ptr = mapped.cast::<u64>();

        let mut slots = Vec::with_capacity(SLOT_COUNT);
        for _ in 0..SLOT_COUNT {
            // SAFETY: creating direct command allocators and initial closed lists on `device`.
            unsafe {
                let begin_alloc: ID3D12CommandAllocator =
                    device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                let begin_list: ID3D12GraphicsCommandList = device.CreateCommandList(
                    0,
                    D3D12_COMMAND_LIST_TYPE_DIRECT,
                    &begin_alloc,
                    None,
                )?;
                begin_list.Close()?;

                let end_alloc: ID3D12CommandAllocator =
                    device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                let end_list: ID3D12GraphicsCommandList = device.CreateCommandList(
                    0,
                    D3D12_COMMAND_LIST_TYPE_DIRECT,
                    &end_alloc,
                    None,
                )?;
                end_list.Close()?;

                slots.push(TimestampSlot {
                    begin_alloc,
                    begin_list,
                    end_alloc,
                    end_list,
                    fence_value: 0,
                    submission_id: 0,
                });
            }
        }

        Ok(Self {
            query_heap,
            readback,
            frequency_hz,
            state: Mutex::new(RingState {
                slots,
                next_slot: 0,
                readback_ptr,
            }),
        })
    }

    #[cfg(test)]
    pub(crate) fn frequency_hz(&self) -> u64 {
        self.frequency_hz
    }

    fn drain_locked(&self, state: &mut RingState, completed_fence: u64) {
        if completed_fence == u64::MAX || state.readback_ptr.is_null() {
            return;
        }
        for (slot_idx, slot) in state.slots.iter_mut().enumerate() {
            if slot.fence_value == 0 || slot.fence_value > completed_fence {
                continue;
            }
            let base = slot_idx * QUERIES_PER_SLOT;
            // SAFETY: GPU has completed `slot.fence_value`, which strictly follows `ResolveQueryData`
            // into `readback` for `[base, base + 2)`. `readback_ptr` stays valid until `Drop`.
            let (t0, t1) = unsafe {
                let t0 = *state.readback_ptr.add(base);
                let t1 = *state.readback_ptr.add(base + 1);
                (t0, t1)
            };
            let delta = t1.saturating_sub(t0);
            let us = ticks_to_us(delta, self.frequency_hz);
            HW_RAW_TICKS.fetch_add(delta, Ordering::Relaxed);
            HW_TIME_US.fetch_add(us, Ordering::Relaxed);
            HW_SAMPLES.fetch_add(1, Ordering::Relaxed);
            slot.fence_value = 0;
            slot.submission_id = 0;
        }
    }

    /// Prepares a start/end timestamp command-list bracket if a ring slot is free
    /// under `completed_fence`. Returns `None` if all slots are still in flight.
    pub(crate) fn prepare_bracket(
        &self,
        submission_id: u64,
        completed_fence: u64,
    ) -> Option<TimestampBracket> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        self.drain_locked(&mut state, completed_fence);

        let slot_idx = state.next_slot % SLOT_COUNT;
        if state.slots[slot_idx].fence_value != 0 {
            return None;
        }
        let begin_query = (slot_idx * QUERIES_PER_SLOT) as u32;
        let end_query = begin_query + 1;
        let resolve_offset = (begin_query as u64) * (size_of::<u64>() as u64);

        {
            let slot = &mut state.slots[slot_idx];
            // SAFETY: `slot.fence_value == 0`, so GPU execution from any prior use has finished.
            let recorded = unsafe {
                slot.begin_alloc
                    .Reset()
                    .and_then(|()| slot.begin_list.Reset(&slot.begin_alloc, None))
                    .and_then(|()| {
                        slot.begin_list.EndQuery(
                            &self.query_heap,
                            D3D12_QUERY_TYPE_TIMESTAMP,
                            begin_query,
                        );
                        slot.begin_list.Close()
                    })
                    .and_then(|()| slot.end_alloc.Reset())
                    .and_then(|()| slot.end_list.Reset(&slot.end_alloc, None))
                    .and_then(|()| {
                        slot.end_list.EndQuery(
                            &self.query_heap,
                            D3D12_QUERY_TYPE_TIMESTAMP,
                            end_query,
                        );
                        slot.end_list.ResolveQueryData(
                            &self.query_heap,
                            D3D12_QUERY_TYPE_TIMESTAMP,
                            begin_query,
                            QUERIES_PER_SLOT as u32,
                            &self.readback,
                            resolve_offset,
                        );
                        slot.end_list.Close()
                    })
            };
            if recorded.is_err() {
                return None;
            }
            slot.submission_id = submission_id;
        }
        state.next_slot = (slot_idx + 1) % SLOT_COUNT;

        let begin_cmd: ID3D12CommandList = state.slots[slot_idx].begin_list.cast().ok()?;
        let end_cmd: ID3D12CommandList = state.slots[slot_idx].end_list.cast().ok()?;
        Some(TimestampBracket {
            slot_idx,
            begin_cmd,
            end_cmd,
        })
    }

    /// Associates a submitted bracket with the host `fence_value` signalled after it.
    pub(crate) fn commit_bracket(&self, slot_idx: usize, fence_value: u64) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = state.slots.get_mut(slot_idx) {
            slot.fence_value = fence_value.max(1);
        }
        SUBMITTED_BRACKETS.fetch_add(1, Ordering::Relaxed);
    }

    /// Releases an unused bracket if submission failed before queueing.
    pub(crate) fn abort_bracket(&self, slot_idx: usize) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = state.slots.get_mut(slot_idx) {
            slot.fence_value = 0;
            slot.submission_id = 0;
        }
    }

    /// Resolves any in-flight timestamp brackets whose `fence_value <= fence.GetCompletedValue()`.
    pub(crate) fn resolve_completed(&self, fence: &ID3D12Fence) {
        // SAFETY: plain D3D12 fence query.
        let completed = unsafe { fence.GetCompletedValue() };
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        self.drain_locked(&mut state, completed);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::Mutex;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};

    pub(crate) static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn gpu_time_status_never_reports_zero_as_measured_when_unresolved() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(
            classify_gpu_time_status(0, 0.0, 12, true),
            GpuTimeStatus::Pending
        );
        assert_eq!(
            classify_gpu_time_status(0, 24.5, 4, true),
            GpuTimeStatus::WaitClockOnly
        );
        assert_eq!(
            classify_gpu_time_status(4, 0.0, 4, true),
            GpuTimeStatus::Measured
        );
        assert_eq!(
            classify_gpu_time_status(0, 0.0, 0, true),
            GpuTimeStatus::Unavailable
        );
        assert_eq!(ticks_to_us(10_000_000, 1_000_000_000), 10_000);
    }

    #[test]
    fn standin_and_drop_counters_aggregate_and_reset_per_window() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _ = take_window_snapshot(1.0, 0.0);
        set_ring_available(false);
        record_pipeline_exec(
            classify_pipeline(false, false, false, 4, Some(StandInKind::Ambient)),
            Some(StandInKind::Ambient),
            false,
            false,
        );
        record_pipeline_exec(
            classify_pipeline(true, false, false, 0, Some(StandInKind::ShadowMask)),
            Some(StandInKind::ShadowMask),
            true,
            true,
        );
        record_pipeline_exec(
            classify_pipeline(false, true, false, 1, None),
            None,
            false,
            false,
        );
        record_dropped_pso(PipelineMiss::Pending);
        record_dropped_pso(PipelineMiss::Failed);
        record_dropped_pso(PipelineMiss::Other);
        record_dropped_indirect_only();

        let snap = take_window_snapshot(10.0, 18.0);
        assert_eq!(snap.window_frames, 10);
        assert_eq!(snap.gpu_time_status, GpuTimeStatus::WaitClockOnly);
        assert_eq!(snap.standin_ambient, 1);
        assert_eq!(snap.standin_shadowmask, 1);
        assert_eq!(snap.standin_window_total, 2);
        assert!((snap.standin_draws - 0.2).abs() < 1e-6);
        assert_eq!(snap.mesh_geometry, 1);
        assert_eq!(snap.indirect_dispatch, 1);
        assert_eq!(snap.submitted_draws, 2);
        assert_eq!(snap.submitted_dispatches, 1);
        assert_eq!(snap.dropped_pending, 1);
        assert_eq!(snap.dropped_failed, 1);
        assert_eq!(snap.dropped_other, 1);
        assert_eq!(
            snap.dropped_window_total,
            snap.dropped_null + snap.dropped_pending + snap.dropped_failed + snap.dropped_other
        );
        assert_eq!(snap.dropped_indirect_only, 1);
        assert!((snap.dropped_pso_draws - 0.3).abs() < 1e-6);
    }

    #[test]
    fn gpu_hw_ms_uses_window_frames_not_one_second_divisor() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _ = take_window_snapshot(1.0, 0.0);
        set_ring_available(true);
        HW_TIME_US.store(50_000, Ordering::Relaxed);
        HW_SAMPLES.store(5, Ordering::Relaxed);
        let snap = take_window_snapshot(10.0, 0.0);
        assert_eq!(snap.window_frames, 10);
        assert!((snap.gpu_hw_total_ms - 50.0).abs() < 1e-6);
        assert!((snap.gpu_hw_ms - 5.0).abs() < 1e-6);
        assert!((snap.gpu_hw_ms_per_frame - 5.0).abs() < 1e-6);
        assert!((snap.gpu_hw_ms_per_sample - 10.0).abs() < 1e-6);
    }

    #[test]
    fn d3d12_timestamp_ring_measures_real_gpu_work() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _ = take_window_snapshot(1.0, 0.0);

        let mut device: Option<ID3D12Device> = None;
        // SAFETY: adapter probe + D3D12 device creation for standalone test.
        if unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device) }.is_err() {
            return;
        }
        let Some(device) = device else {
            return;
        };
        // SAFETY: standard direct command queue, fence, and upload->default 4 MiB copy test.
        unsafe {
            let queue: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .expect("direct queue");
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
            let event = CreateEventW(None, false, false, None).expect("event");

            let ring = GpuTimestampRing::new(&device, &queue).expect("timestamp ring");
            set_ring_available(true);
            assert!(ring.frequency_hz() > 0);

            const COPY_BYTES: u64 = 8 * 1024 * 1024;
            let upload_props = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_UPLOAD,
                ..Default::default()
            };
            let default_props = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let buf_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: COPY_BYTES,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut src: Option<ID3D12Resource> = None;
            let mut dst: Option<ID3D12Resource> = None;
            device
                .CreateCommittedResource(
                    &upload_props,
                    D3D12_HEAP_FLAG_NONE,
                    &buf_desc,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut src,
                )
                .expect("src upload buffer");
            device
                .CreateCommittedResource(
                    &default_props,
                    D3D12_HEAP_FLAG_NONE,
                    &buf_desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut dst,
                )
                .expect("dst default buffer");
            let src = src.unwrap();
            let dst = dst.unwrap();

            let alloc: ID3D12CommandAllocator = device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .expect("work allocator");
            let work_list: ID3D12GraphicsCommandList = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
                .expect("work list");
            for _ in 0..8 {
                work_list.CopyBufferRegion(&dst, 0, &src, 0, COPY_BYTES);
            }
            work_list.Close().expect("close work list");
            let work_cmd: ID3D12CommandList = work_list.cast().unwrap();

            let bracket = ring
                .prepare_bracket(1, fence.GetCompletedValue())
                .expect("timestamp bracket");
            let batch = [
                Some(bracket.begin_cmd.clone()),
                Some(work_cmd),
                Some(bracket.end_cmd.clone()),
            ];
            queue.ExecuteCommandLists(&batch);
            queue.Signal(&fence, 1).expect("signal fence");
            ring.commit_bracket(bracket.slot_idx, 1);

            fence.SetEventOnCompletion(1, event).expect("set event");
            WaitForSingleObject(event, INFINITE);
            let _ = windows::Win32::Foundation::CloseHandle(event);

            ring.resolve_completed(&fence);
            let raw_ticks = HW_RAW_TICKS.load(Ordering::Relaxed);
            let snap = take_window_snapshot(1.0, 0.0);
            assert_eq!(snap.gpu_time_status, GpuTimeStatus::Measured);
            assert_eq!(snap.gpu_hw_samples, 1);
            assert!(
                raw_ticks > 0,
                "expected non-zero hardware ticks on GPU work"
            );
            assert!(snap.gpu_hw_ms >= 0.0);
            eprintln!(
                "RTX GPU timestamp check: freq={} Hz, raw_ticks={}, hw_ms={:.4}",
                ring.frequency_hz(),
                raw_ticks,
                snap.gpu_hw_ms
            );
        }
    }
}
