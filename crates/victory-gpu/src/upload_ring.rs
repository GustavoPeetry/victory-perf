//! Pooled fence-retired upload ring & optional D3D12 Async Copy Queue (Front G).
//!
//! Eliminates per-frame `CreateCommittedResource` / `Map` / `Release` churn on
//! transient uploads when `VICTORY_REA_ASYNC_STREAMING=1` is set, and optionally
//! dispatches batched DMA transfers on a dedicated `D3D12_COMMAND_LIST_TYPE_COPY`
//! queue synchronized via cross-queue fence when `VICTORY_REA_COPY_QUEUE=1`.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::core::Interface;

use crate::accounting;

/// Alignment for pooled sub-allocations in [`UploadRing`] (matches D3D12 texture placement).
pub const RING_ALIGN: u64 = 512;

/// Default size of each [`UploadRing`] backing chunk (32 MiB).
pub const DEFAULT_RING_CHUNK: u64 = 32 << 20;

/// Cap on cached idle upload buffers in [`UploadPool`] (16 MiB) to stay friendly to 16 GiB hosts.
const MAX_CACHED_POOL_BYTES: u64 = 16 << 20;

/// Maximum bucketed single-buffer size in [`UploadPool`] (4 MiB).
const MAX_POOLED_BUFFER_BYTES: u64 = 4 << 20;

fn parse_bool(name: &str) -> bool {
    std::env::var(name).ok().is_some_and(|v| {
        let t = v.trim();
        t == "1" || t.eq_ignore_ascii_case("true") || t.eq_ignore_ascii_case("yes")
    })
}

/// Whether `VICTORY_REA_ASYNC_STREAMING=1` enables the pooled upload ring and
/// DStorage scratch/backpressure optimizations (OFF by default).
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = parse_bool("VICTORY_REA_ASYNC_STREAMING");
        if on {
            tracing::info!("REA-002 pooled upload ring enabled (VICTORY_REA_ASYNC_STREAMING=1)");
        }
        on
    })
}

/// Whether `VICTORY_REA_COPY_QUEUE=1` enables the dedicated D3D12 Copy Queue
/// for batched async DMA uploads with cross-queue fence sync (OFF by default).
pub fn copy_queue_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = parse_bool("VICTORY_REA_COPY_QUEUE");
        if on {
            tracing::info!("REA-002 dedicated D3D12 Copy Queue enabled (VICTORY_REA_COPY_QUEUE=1)");
        }
        on
    })
}

static POOL_HITS: AtomicU64 = AtomicU64::new(0);
static POOL_MISSES: AtomicU64 = AtomicU64::new(0);
static POOL_CACHED_BYTES: AtomicU64 = AtomicU64::new(0);
static RING_ALLOCS: AtomicU64 = AtomicU64::new(0);
static RING_BYTES_SUBALLOC: AtomicU64 = AtomicU64::new(0);
static COPY_QUEUE_BATCHES: AtomicU64 = AtomicU64::new(0);
static COPY_QUEUE_BYTES: AtomicU64 = AtomicU64::new(0);
static CROSS_QUEUE_WAITS: AtomicU64 = AtomicU64::new(0);

/// Telemetry snapshot for GPU upload ring & copy queue activity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UploadRingStats {
    pub enabled: bool,
    pub copy_queue_enabled: bool,
    pub pool_hits: u64,
    pub pool_misses: u64,
    pub pool_cached_bytes: u64,
    pub ring_allocs: u64,
    pub ring_bytes_suballoc: u64,
    pub copy_queue_batches: u64,
    pub copy_queue_bytes: u64,
    pub cross_queue_waits: u64,
}

pub fn stats() -> UploadRingStats {
    UploadRingStats {
        enabled: enabled(),
        copy_queue_enabled: copy_queue_enabled(),
        pool_hits: POOL_HITS.load(Ordering::Relaxed),
        pool_misses: POOL_MISSES.load(Ordering::Relaxed),
        pool_cached_bytes: POOL_CACHED_BYTES.load(Ordering::Relaxed),
        ring_allocs: RING_ALLOCS.load(Ordering::Relaxed),
        ring_bytes_suballoc: RING_BYTES_SUBALLOC.load(Ordering::Relaxed),
        copy_queue_batches: COPY_QUEUE_BATCHES.load(Ordering::Relaxed),
        copy_queue_bytes: COPY_QUEUE_BYTES.load(Ordering::Relaxed),
        cross_queue_waits: CROSS_QUEUE_WAITS.load(Ordering::Relaxed),
    }
}

/// Mapped whole upload buffer wrapped safely for bucketed reuse.
pub(crate) struct PooledEntry {
    pub(crate) resource: ID3D12Resource,
    pub(crate) data: *mut u8,
    pub(crate) bucket_bytes: u64,
}

// SAFETY: Upload heap resources and persistent CPU mappings are thread-safe when
// synchronized via fence submission boundaries.
unsafe impl Send for PooledEntry {}

struct RetiredEntry {
    retire_submission: u64,
    entry: PooledEntry,
}

struct PoolState {
    free: BTreeMap<u64, Vec<PooledEntry>>,
    retired: Vec<RetiredEntry>,
    cached_bytes: u64,
}

impl PoolState {
    const fn new() -> Self {
        Self {
            free: BTreeMap::new(),
            retired: Vec::new(),
            cached_bytes: 0,
        }
    }

    fn drain(&mut self, finished_submission: u64) {
        if self.retired.is_empty() {
            return;
        }
        let mut keep = Vec::new();
        for item in self.retired.drain(..) {
            if item.retire_submission <= finished_submission {
                if self.cached_bytes + item.entry.bucket_bytes <= MAX_CACHED_POOL_BYTES {
                    self.cached_bytes += item.entry.bucket_bytes;
                    self.free
                        .entry(item.entry.bucket_bytes)
                        .or_default()
                        .push(item.entry);
                } else {
                    accounting::on_upload_ring_delta(-(item.entry.bucket_bytes as i64));
                    drop(item.entry);
                }
            } else {
                keep.push(item);
            }
        }
        self.retired = keep;
        POOL_CACHED_BYTES.store(self.cached_bytes, Ordering::Relaxed);
    }

    fn take(&mut self, bucket: u64, finished_submission: u64) -> Option<PooledEntry> {
        self.drain(finished_submission);
        let list = self.free.get_mut(&bucket)?;
        let entry = list.pop()?;
        self.cached_bytes = self.cached_bytes.saturating_sub(entry.bucket_bytes);
        POOL_CACHED_BYTES.store(self.cached_bytes, Ordering::Relaxed);
        Some(entry)
    }

    fn retire(&mut self, submission: u64, entry: PooledEntry) {
        if entry.bucket_bytes > MAX_POOLED_BUFFER_BYTES {
            accounting::on_upload_ring_delta(-(entry.bucket_bytes as i64));
            drop(entry);
            return;
        }
        self.retired.push(RetiredEntry {
            retire_submission: submission,
            entry,
        });
    }
}

static UPLOAD_POOL: Mutex<PoolState> = Mutex::new(PoolState::new());

fn create_upload_resource(device: &ID3D12Device, bytes: u64) -> Option<(ID3D12Resource, *mut u8)> {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_UPLOAD,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes.max(1),
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
    let mut resource: Option<ID3D12Resource> = None;
    let mut ptr: *mut c_void = std::ptr::null_mut();
    // SAFETY: standard D3D12 UPLOAD heap creation & persistent mapping.
    let res = unsafe {
        device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                None,
                &mut resource,
            )
            .and_then(|()| {
                let r = resource.as_ref().ok_or_else(windows::core::Error::empty)?;
                r.Map(0, None, Some(&mut ptr))
            })
    };
    match res {
        Ok(()) => Some((resource?, ptr.cast())),
        Err(e) => {
            tracing::warn!(error = %e, bytes, "upload_ring: failed creating mapped upload buffer");
            None
        }
    }
}

fn bucket_size_for(size: u64) -> Option<u64> {
    if size > MAX_POOLED_BUFFER_BYTES {
        return None;
    }
    if size <= (64 << 10) {
        Some(64 << 10)
    } else if size <= (256 << 10) {
        Some(size.next_multiple_of(64 << 10))
    } else if size <= (1 << 20) {
        Some(size.next_multiple_of(256 << 10))
    } else {
        Some(size.next_multiple_of(512 << 10))
    }
}

/// Checkout a reusable mapped upload buffer from the bucketed pool (or allocate one).
/// Caller gets `(ID3D12Resource, cpu_ptr, bucket_bytes)` with `offset == 0`.
pub(crate) fn checkout_upload(
    device: &ID3D12Device,
    size: u64,
    finished_submission: u64,
) -> Option<(ID3D12Resource, *mut u8, u64)> {
    let bucket = bucket_size_for(size)?;
    if let Some(entry) = UPLOAD_POOL
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take(bucket, finished_submission)
    {
        POOL_HITS.fetch_add(1, Ordering::Relaxed);
        return Some((entry.resource, entry.data, entry.bucket_bytes));
    }
    POOL_MISSES.fetch_add(1, Ordering::Relaxed);
    let (res, ptr) = create_upload_resource(device, bucket)?;
    accounting::on_upload_ring_delta(bucket as i64);
    Some((res, ptr, bucket))
}

/// Retire a pooled upload buffer so it becomes eligible for reuse once
/// `finished_submission >= retire_submission`.
pub(crate) fn return_upload(
    resource: ID3D12Resource,
    data: *mut u8,
    bucket_bytes: u64,
    retire_submission: u64,
) {
    let mut pool = UPLOAD_POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.retire(
        retire_submission,
        PooledEntry {
            resource,
            data,
            bucket_bytes,
        },
    );
    pool.drain(crate::host::finished_submission());
}

/// Drain completed pooled uploads on submission fence advance.
pub(crate) fn drain_pool(finished_submission: u64) {
    if !enabled() {
        return;
    }
    if let Ok(mut pool) = UPLOAD_POOL.try_lock() {
        pool.drain(finished_submission);
    }
}

/// Sub-allocated linear ring slice inside [`UploadRing`].
pub struct RingSlice {
    pub resource: ID3D12Resource,
    pub offset: u64,
    pub cpu_ptr: *mut u8,
    pub size: u64,
}

/// Linear bump-allocated 32 MiB D3D12 upload ring with fence-gated reclamation.
pub struct UploadRing {
    resource: ID3D12Resource,
    base_ptr: *mut u8,
    capacity: u64,
    head: u64,
    tail: u64,
    in_flight: Vec<(u64, u64)>, // (submission, new_tail_after_alloc)
}

// SAFETY: Single synchronized mutex protects ring state and D3D12 UPLOAD memory.
unsafe impl Send for UploadRing {}

impl UploadRing {
    /// Allocate a mapped linear upload ring buffer on `device`.
    pub fn new(device: &ID3D12Device, capacity: u64) -> Option<Self> {
        let cap = capacity.max(64 << 10).next_multiple_of(RING_ALIGN);
        let (resource, base_ptr) = create_upload_resource(device, cap)?;
        accounting::on_upload_ring_delta(cap as i64);
        Some(Self {
            resource,
            base_ptr,
            capacity: cap,
            head: 0,
            tail: 0,
            in_flight: Vec::new(),
        })
    }

    /// Sub-allocate `size` bytes aligned to `RING_ALIGN` after reclaiming allocations
    /// whose GPU submission `<= finished_submission`.
    pub fn alloc(
        &mut self,
        size: u64,
        finished_submission: u64,
        current_submission: u64,
    ) -> Option<RingSlice> {
        self.reclaim(finished_submission);
        let aligned = size.max(1).next_multiple_of(RING_ALIGN);
        if aligned > self.capacity {
            return None;
        }

        let offset = if self.head >= self.tail {
            if self.head + aligned <= self.capacity {
                let off = self.head;
                self.head += aligned;
                off
            } else if aligned < self.tail {
                let off = 0;
                self.head = aligned;
                off
            } else {
                return None;
            }
        } else if self.head + aligned < self.tail {
            let off = self.head;
            self.head += aligned;
            off
        } else {
            return None;
        };

        self.in_flight.push((current_submission, offset + aligned));
        RING_ALLOCS.fetch_add(1, Ordering::Relaxed);
        RING_BYTES_SUBALLOC.fetch_add(aligned, Ordering::Relaxed);
        Some(RingSlice {
            resource: self.resource.clone(),
            offset,
            // SAFETY: `offset + aligned <= self.capacity`.
            cpu_ptr: unsafe { self.base_ptr.add(offset as usize) },
            size: aligned,
        })
    }

    pub fn reclaim(&mut self, finished_submission: u64) {
        let mut advanced_to = None;
        let mut keep = Vec::new();
        for &(sub, end_off) in &self.in_flight {
            if sub <= finished_submission {
                advanced_to = Some(end_off);
            } else {
                keep.push((sub, end_off));
            }
        }
        if let Some(new_tail) = advanced_to {
            self.tail = new_tail % self.capacity;
        }
        self.in_flight = keep;
    }
}

impl Drop for UploadRing {
    fn drop(&mut self) {
        accounting::on_upload_ring_delta(-(self.capacity as i64));
    }
}

/// Dedicated `D3D12_COMMAND_LIST_TYPE_COPY` queue + cross-queue fence sync.
pub struct AsyncCopyQueue {
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    next_fence: AtomicU64,
    closed: AtomicBool,
}

// SAFETY: D3D12 copy queue and allocator/list are used strictly sequentially under lock or test.
unsafe impl Send for AsyncCopyQueue {}

impl AsyncCopyQueue {
    pub fn new(device: &ID3D12Device) -> windows::core::Result<Self> {
        unsafe {
            let queue: ID3D12CommandQueue =
                device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_COPY,
                    ..Default::default()
                })?;
            let allocator: ID3D12CommandAllocator =
                device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY)?;
            let list: ID3D12GraphicsCommandList =
                device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &allocator, None)?;
            list.Close()?;
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
            Ok(Self {
                queue,
                allocator,
                list,
                fence,
                next_fence: AtomicU64::new(1),
                closed: AtomicBool::new(true),
            })
        }
    }

    /// Submit a batch of `(dst, dst_offset, src, src_offset, bytes)` buffer copies on
    /// the Copy Queue, signal the cross-queue fence, and insert `direct_queue.Wait(...)`.
    pub fn copy_buffers_and_sync(
        &mut self,
        direct_queue: &ID3D12CommandQueue,
        copies: &[(&ID3D12Resource, u64, &ID3D12Resource, u64, u64)],
    ) -> windows::core::Result<u64> {
        let wait_val = self.next_fence.fetch_add(1, Ordering::Relaxed);
        // Wait for any previous copy batch before resetting allocator.
        let prev = wait_val - 1;
        if prev > 0 {
            while unsafe { self.fence.GetCompletedValue() } < prev {
                std::thread::yield_now();
            }
        }
        unsafe {
            self.allocator.Reset()?;
            self.list.Reset(&self.allocator, None)?;
            self.closed.store(false, Ordering::Relaxed);
            let mut total_bytes = 0u64;
            for &(dst, dst_off, src, src_off, len) in copies {
                self.list.CopyBufferRegion(dst, dst_off, src, src_off, len);
                total_bytes += len;
            }
            self.list.Close()?;
            self.closed.store(true, Ordering::Relaxed);
            let cmd: ID3D12CommandList = self.list.cast()?;
            self.queue.ExecuteCommandLists(&[Some(cmd)]);
            self.queue.Signal(&self.fence, wait_val)?;
            direct_queue.Wait(&self.fence, wait_val)?;

            COPY_QUEUE_BATCHES.fetch_add(1, Ordering::Relaxed);
            COPY_QUEUE_BYTES.fetch_add(total_bytes, Ordering::Relaxed);
            CROSS_QUEUE_WAITS.fetch_add(1, Ordering::Relaxed);
        }
        Ok(wait_val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;

    #[test]
    fn default_flags_are_off_for_baseline_parity() {
        assert!(!enabled());
        assert!(!copy_queue_enabled());
    }

    #[test]
    fn upload_pool_bucket_sizes_avoid_power_of_two_bloat() {
        assert_eq!(bucket_size_for(16 << 10), Some(64 << 10));
        assert_eq!(bucket_size_for(300 << 10), Some(512 << 10));
        assert_eq!(bucket_size_for(1200 << 10), Some(1536 << 10));
        assert_eq!(bucket_size_for(4 << 20), Some(4 << 20));
        assert_eq!(bucket_size_for(5 << 20), None);
    }

    #[test]
    fn upload_pool_reuses_retired_buffers_across_submissions() {
        let mut device: Option<ID3D12Device> = None;
        if unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device) }.is_err() {
            return;
        }
        let device = device.unwrap();

        let (res1, ptr1, cap1) =
            checkout_upload(&device, 32 << 10, 0).expect("alloc pooled buffer");
        let raw1 = res1.as_raw() as usize;
        assert_eq!(cap1, 64 << 10);
        // Retire at submission 5; before finished=5 it must NOT be reused!
        return_upload(res1, ptr1, cap1, 5);
        assert!(
            UPLOAD_POOL.lock().unwrap().take(cap1, 4).is_none(),
            "in-flight retired buffer must not be reused before fence reaches submission 5"
        );

        let before_hits = stats().pool_hits;
        let (res2, _ptr2, cap2) =
            checkout_upload(&device, 40 << 10, 5).expect("reuse pooled buffer");
        assert_eq!(cap2, 64 << 10);
        assert_eq!(
            res2.as_raw() as usize,
            raw1,
            "bucket reuse should hand back the exact cached resource"
        );
        assert_eq!(stats().pool_hits, before_hits + 1);
    }

    #[test]
    fn gpu_copy_queue_and_upload_ring_transfer_with_cross_queue_fence() {
        let mut device: Option<ID3D12Device> = None;
        if unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device) }.is_err() {
            return;
        }
        let device = device.unwrap();

        let direct_queue: ID3D12CommandQueue = unsafe {
            device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .expect("direct queue")
        };

        let mut ring = UploadRing::new(&device, 1 << 20).expect("upload ring");
        let payload_len = 64 << 10;
        let slice = ring.alloc(payload_len, 0, 1).expect("ring slice");
        let expected: Vec<u8> = (0..payload_len as usize)
            .map(|i| ((i * 29 + 7) & 0xFF) as u8)
            .collect();
        unsafe {
            std::ptr::copy_nonoverlapping(expected.as_ptr(), slice.cpu_ptr, expected.len());
        }

        // Allocate DEFAULT destination buffer and READBACK verification buffer.
        let make_buf =
            |heap_type: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES| -> ID3D12Resource {
                let mut r = None;
                let heap = D3D12_HEAP_PROPERTIES {
                    Type: heap_type,
                    ..Default::default()
                };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: payload_len,
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
                unsafe {
                    device
                        .CreateCommittedResource(
                            &heap,
                            D3D12_HEAP_FLAG_NONE,
                            &desc,
                            state,
                            None,
                            &mut r,
                        )
                        .unwrap();
                }
                r.unwrap()
            };

        let gpu_dst = make_buf(D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON);
        let cpu_readback = make_buf(D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);

        let mut cq = AsyncCopyQueue::new(&device).expect("copy queue");
        cq.copy_buffers_and_sync(
            &direct_queue,
            &[(&gpu_dst, 0, &slice.resource, slice.offset, payload_len)],
        )
        .expect("copy queue submit + direct queue cross-wait");

        // Now execute a copy gpu_dst -> cpu_readback on DirectQueue (which waits on copy_fence!).
        unsafe {
            let alloc: ID3D12CommandAllocator = device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .unwrap();
            let list: ID3D12GraphicsCommandList = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
                .unwrap();
            list.CopyBufferRegion(&cpu_readback, 0, &gpu_dst, 0, payload_len);
            list.Close().unwrap();
            let cmd: ID3D12CommandList = list.cast().unwrap();
            direct_queue.ExecuteCommandLists(&[Some(cmd)]);

            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            direct_queue.Signal(&fence, 1).unwrap();
            while fence.GetCompletedValue() < 1 {
                std::thread::yield_now();
            }

            let mut map_ptr: *mut c_void = std::ptr::null_mut();
            cpu_readback.Map(0, None, Some(&mut map_ptr)).unwrap();
            let got = std::slice::from_raw_parts(map_ptr.cast::<u8>(), payload_len as usize);
            assert_eq!(
                got,
                &expected[..],
                "cross-queue CopyQueue -> DirectQueue transfer must match byte-for-byte"
            );
            cpu_readback.Unmap(0, None);
        }
        ring.reclaim(1);
    }
}
