//! Fixed-capacity preallocated GPU execution breadcrumb ring (`256` slots).
//!
//! Records lightweight op-level (when `VICTORY_GPU_BREADCRUMBS=1`, `VICTORY_D3D12_DRED=1`,
//! or `diagnostics` feature) and submission-level markers with zero per-draw heap
//! allocations so `DXGI_ERROR_DEVICE_HUNG` / `DEVICE_REMOVED` always reports the
//! suspect pipeline, shader hashes, dispatch dimensions, and fence state even without DRED.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// Fixed ring capacity (power of two).
pub const RING_CAPACITY: usize = 256;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum QueueKind {
    #[default]
    Direct = 0,
    Copy = 1,
    Detile = 2,
}

impl QueueKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Copy => "copy",
            Self::Detile => "detile",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum OpKind {
    #[default]
    Submit = 0,
    Draw = 1,
    DrawIndexed = 2,
    Dispatch = 3,
    DispatchMesh = 4,
    ExecuteIndirect = 5,
}

impl OpKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Submit => "Submit",
            Self::Draw => "Draw",
            Self::DrawIndexed => "DrawIndexed",
            Self::Dispatch => "Dispatch",
            Self::DispatchMesh => "DispatchMesh",
            Self::ExecuteIndirect => "ExecuteIndirect",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GpuOpBreadcrumb {
    pub seq: u64,
    pub frame: u64,
    pub queue: QueueKind,
    pub cmd_list_ptr: usize,
    pub submission_id: u64,
    pub fence_value: u64,
    pub op_kind: OpKind,
    pub pipeline_hash: u64,
    pub shader_hashes: [u64; 3],
    pub dispatch_dims: [u32; 3],
    pub root_sig_hash: u64,
    pub bound_resources_count: u32,
    pub resource_state_mask: u32,
}

impl Default for GpuOpBreadcrumb {
    fn default() -> Self {
        Self {
            seq: 0,
            frame: 0,
            queue: QueueKind::Direct,
            cmd_list_ptr: 0,
            submission_id: 0,
            fence_value: u64::MAX,
            op_kind: OpKind::Submit,
            pipeline_hash: 0,
            shader_hashes: [0; 3],
            dispatch_dims: [0; 3],
            root_sig_hash: 0,
            bound_resources_count: 0,
            resource_state_mask: 0,
        }
    }
}

struct Ring {
    slots: [GpuOpBreadcrumb; RING_CAPACITY],
    next_seq: u64,
}

impl Ring {
    const fn new() -> Self {
        const EMPTY: GpuOpBreadcrumb = GpuOpBreadcrumb {
            seq: 0,
            frame: 0,
            queue: QueueKind::Direct,
            cmd_list_ptr: 0,
            submission_id: 0,
            fence_value: u64::MAX,
            op_kind: OpKind::Submit,
            pipeline_hash: 0,
            shader_hashes: [0; 3],
            dispatch_dims: [0; 3],
            root_sig_hash: 0,
            bound_resources_count: 0,
            resource_state_mask: 0,
        };
        Self {
            slots: [EMPTY; RING_CAPACITY],
            next_seq: 1,
        }
    }

    fn push(&mut self, mut entry: GpuOpBreadcrumb) -> u64 {
        let seq = self.next_seq;
        self.next_seq = self.next_seq.saturating_add(1);
        entry.seq = seq;
        let idx = ((seq - 1) as usize) & (RING_CAPACITY - 1);
        self.slots[idx] = entry;
        seq
    }
}

static RING: Mutex<Ring> = Mutex::new(Ring::new());
static CURRENT_SUBMISSION_ID: AtomicU64 = AtomicU64::new(1);

/// Whether per-draw/dispatch GPU op breadcrumbs are captured in the ring buffer.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("VICTORY_GPU_BREADCRUMBS").is_some_and(|v| v != "0")
            || crate::host::dred_on()
            || cfg!(feature = "diagnostics")
    })
}

/// Begins tracking the active recording submission ID (1-based monotonic).
pub fn note_submission_started(submission_id: u64) {
    CURRENT_SUBMISSION_ID.store(submission_id.max(1), Ordering::Relaxed);
}

/// Records a draw/dispatch/indirect operation when op breadcrumbs are enabled.
pub fn record_op(mut entry: GpuOpBreadcrumb) {
    if !enabled() {
        return;
    }
    if entry.submission_id == 0 {
        entry.submission_id = CURRENT_SUBMISSION_ID.load(Ordering::Relaxed);
    }
    if let Ok(mut r) = RING.try_lock() {
        let _ = r.push(entry);
    }
}

/// Records a lightweight submission-level breadcrumb (always active, zero allocations).
pub fn record_submit(
    queue: QueueKind,
    submission_id: u64,
    fence_value: u64,
    list_count: u32,
    pipeline_hash: u64,
) {
    let entry = GpuOpBreadcrumb {
        seq: 0,
        frame: crate::frame::current(),
        queue,
        cmd_list_ptr: 0,
        submission_id,
        fence_value,
        op_kind: OpKind::Submit,
        pipeline_hash,
        shader_hashes: [0; 3],
        dispatch_dims: [list_count, 0, 0],
        root_sig_hash: 0,
        bound_resources_count: list_count,
        resource_state_mask: 0,
    };
    if let Ok(mut r) = RING.lock() {
        let _ = r.push(entry);
    }
}

/// Stamps all recorded ops on `cmd_list_ptr` (or pending `submission_id`) with the queue's signaled fence value.
pub fn mark_submission_fence(cmd_list_ptrs: &[usize], submission_id: u64, fence_value: u64) {
    if let Ok(mut r) = RING.lock() {
        for slot in &mut r.slots {
            if slot.seq == 0 {
                continue;
            }
            let ptr_match = slot.cmd_list_ptr != 0 && cmd_list_ptrs.contains(&slot.cmd_list_ptr);
            let sub_match = slot.submission_id == submission_id;
            if (ptr_match || sub_match) && slot.fence_value == u64::MAX {
                slot.fence_value = fence_value;
                slot.submission_id = submission_id;
            }
        }
    }
}

fn format_entry(e: &GpuOpBreadcrumb) -> String {
    format!(
        "seq={} f={} q={} sub={} fence={} op={} pipe={:#x} shaders=[{:016x},{:016x},{:016x}] dims={}x{}x{} rs={:#x} res={} state={:#x}",
        e.seq,
        e.frame,
        e.queue.as_str(),
        e.submission_id,
        if e.fence_value == u64::MAX {
            0
        } else {
            e.fence_value
        },
        e.op_kind.as_str(),
        e.pipeline_hash,
        e.shader_hashes[0],
        e.shader_hashes[1],
        e.shader_hashes[2],
        e.dispatch_dims[0],
        e.dispatch_dims[1],
        e.dispatch_dims[2],
        e.root_sig_hash,
        e.bound_resources_count,
        e.resource_state_mask,
    )
}

/// Formats a compact diagnostic summary for device-removal logs against `completed_fence`.
pub fn format_removal_report(completed_fence: u64) -> String {
    let r = RING.lock().unwrap_or_else(|e| e.into_inner());
    let mut entries: Vec<GpuOpBreadcrumb> =
        r.slots.iter().copied().filter(|s| s.seq != 0).collect();
    if entries.is_empty() {
        return format!(
            "; gpu_breadcrumbs: completed_fence={completed_fence} (no recorded entries)"
        );
    }
    entries.sort_unstable_by_key(|e| e.seq);
    let last_completed = entries
        .iter()
        .rev()
        .find(|e| e.fence_value <= completed_fence && e.fence_value != u64::MAX);
    let suspects: Vec<String> = entries
        .iter()
        .filter(|e| e.fence_value == u64::MAX || e.fence_value > completed_fence)
        .take(6)
        .map(format_entry)
        .collect();
    let last_done_str = last_completed
        .map(format_entry)
        .unwrap_or_else(|| "none".to_owned());
    let suspects_str = if suspects.is_empty() {
        entries
            .last()
            .map(format_entry)
            .unwrap_or_else(|| "none".to_owned())
    } else {
        suspects.join(" | ")
    };
    format!(
        "; gpu_breadcrumbs: completed_fence={completed_fence}; last_completed=[{last_done_str}]; suspect_inflight=[{suspects_str}]"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_wraps_256_entries_and_identifies_suspect_inflight_ops() {
        for i in 0..300u64 {
            record_submit(QueueKind::Direct, i + 1, i + 10, 1, 0xABCD_0000 + i);
        }
        let op = GpuOpBreadcrumb {
            seq: 0,
            frame: 99,
            queue: QueueKind::Direct,
            cmd_list_ptr: 0xDEAD_BEEF,
            submission_id: 999,
            fence_value: u64::MAX,
            op_kind: OpKind::Dispatch,
            pipeline_hash: 0x1234_5678,
            shader_hashes: [0x0238_4a60_fda6_f2d0, 0, 0],
            dispatch_dims: [8, 1, 1],
            root_sig_hash: 0x55,
            bound_resources_count: 3,
            resource_state_mask: 0x400,
        };
        {
            let mut r = RING.lock().unwrap();
            r.push(op);
        }
        mark_submission_fence(&[0xDEAD_BEEF], 999, 315);

        let report = format_removal_report(308);
        assert!(report.contains("completed_fence=308"), "{report}");
        assert!(report.contains("fence=315"), "{report}");
        assert!(report.contains("02384a60fda6f2d0"), "{report}");
        assert!(report.contains("dims=8x1x1"), "{report}");
    }
}
