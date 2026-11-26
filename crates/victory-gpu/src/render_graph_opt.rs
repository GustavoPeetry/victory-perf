//! Opt-in rendergraph and barrier fastpaths (`VICTORY_REA003_RENDERGRAPH_FASTPATH=1`).
//!
//! Keeps default runtime behavior untouched (`false` when unset). When enabled, eliminates
//! redundant identical resource-state transitions while always preserving UAV ordering barriers,
//! tracks dirty depth-twin synchronization, and tracks non-settling `ValueWait` fastpath hits.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// Gate flag for REA-003 Frente I rendergraph fastpaths (`off` by default).
pub fn rendergraph_fastpath_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("VICTORY_REA003_RENDERGRAPH_FASTPATH")
            .ok()
            .is_some_and(|v| v.trim() == "1")
    })
}

/// Telemetry counters for rendergraph fastpaths.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RenderGraphFastpathStats {
    pub enabled: bool,
    pub transitions_checked: u64,
    pub transitions_elided: u64,
    pub uav_barriers_preserved: u64,
    pub twin_copies_elided: u64,
    pub twin_dirty_invalidations: u64,
    pub value_wait_fast_hits: u64,
    pub value_wait_settles: u64,
}

static TRANSITIONS_CHECKED: AtomicU64 = AtomicU64::new(0);
static TRANSITIONS_ELIDED: AtomicU64 = AtomicU64::new(0);
static UAV_PRESERVED: AtomicU64 = AtomicU64::new(0);
static TWIN_ELIDED: AtomicU64 = AtomicU64::new(0);
static TWIN_INVALIDATIONS: AtomicU64 = AtomicU64::new(0);
static WAIT_FAST_HITS: AtomicU64 = AtomicU64::new(0);
static WAIT_SETTLES: AtomicU64 = AtomicU64::new(0);

pub fn stats() -> RenderGraphFastpathStats {
    RenderGraphFastpathStats {
        enabled: rendergraph_fastpath_enabled(),
        transitions_checked: TRANSITIONS_CHECKED.load(Ordering::Relaxed),
        transitions_elided: TRANSITIONS_ELIDED.load(Ordering::Relaxed),
        uav_barriers_preserved: UAV_PRESERVED.load(Ordering::Relaxed),
        twin_copies_elided: TWIN_ELIDED.load(Ordering::Relaxed),
        twin_dirty_invalidations: TWIN_INVALIDATIONS.load(Ordering::Relaxed),
        value_wait_fast_hits: WAIT_FAST_HITS.load(Ordering::Relaxed),
        value_wait_settles: WAIT_SETTLES.load(Ordering::Relaxed),
    }
}

/// Returns `true` if a D3D12 resource transition barrier (`before -> after`) must be emitted.
/// When `before == after`, skips the redundant transition barrier. Never skips UAV barriers.
pub fn should_emit_transition(before: u32, after: u32) -> bool {
    TRANSITIONS_CHECKED.fetch_add(1, Ordering::Relaxed);
    if before == after {
        TRANSITIONS_ELIDED.fetch_add(1, Ordering::Relaxed);
        false
    } else {
        true
    }
}

/// Records preservation of a UAV barrier (UAV barriers are never elided by [`should_emit_transition`]).
pub fn note_uav_barrier_preserved() {
    UAV_PRESERVED.fetch_add(1, Ordering::Relaxed);
}

/// Per-resource dirty-state tracker for paired depth/colour twin textures.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DepthTwinDirtyTracker {
    pub depth_behind: bool,
    pub colour_behind: bool,
}

impl DepthTwinDirtyTracker {
    /// Returns whether a sync copy is required when accessing `colour_side` (`true` = colour twin,
    /// `false` = depth texture) for either reading or writing (`write`).
    pub fn on_access(&mut self, colour_side: bool, write: bool) -> bool {
        let need_copy = if colour_side {
            self.colour_behind
        } else {
            self.depth_behind
        };
        if !need_copy {
            if !write {
                TWIN_ELIDED.fetch_add(1, Ordering::Relaxed);
            }
        } else if colour_side {
            self.colour_behind = false;
        } else {
            self.depth_behind = false;
        }
        if write {
            if colour_side {
                self.depth_behind = true;
            } else {
                self.colour_behind = true;
            }
            TWIN_INVALIDATIONS.fetch_add(1, Ordering::Relaxed);
        }
        need_copy
    }
}

pub(crate) fn note_twin_read_elided() {
    TWIN_ELIDED.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn note_twin_dirty_invalidation() {
    TWIN_INVALIDATIONS.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn note_value_wait_fast_hit() {
    WAIT_FAST_HITS.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn note_value_wait_settle() {
    WAIT_SETTLES.fetch_add(1, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rendergraph_fastpath_defaults_off_and_elides_noop_transitions_preserving_uav() {
        assert!(!rendergraph_fastpath_enabled());
        let before_stats = stats();
        assert!(!should_emit_transition(0x80, 0x80));
        assert!(should_emit_transition(0x80, 0x400));
        note_uav_barrier_preserved();
        let after = stats();
        assert_eq!(
            after.transitions_checked,
            before_stats.transitions_checked + 2
        );
        assert_eq!(
            after.transitions_elided,
            before_stats.transitions_elided + 1
        );
        assert_eq!(
            after.uav_barriers_preserved,
            before_stats.uav_barriers_preserved + 1
        );
    }

    #[test]
    fn depth_twin_dirty_tracker_elides_readonly_and_invalidates_on_write() {
        let before = stats();
        let mut tracker = DepthTwinDirtyTracker::default();
        // Clean read-only accesses on both sides require zero copies.
        assert!(!tracker.on_access(false, false));
        assert!(!tracker.on_access(true, false));

        // Writing colour side invalidates depth side, triggering a single copy on next depth read.
        assert!(!tracker.on_access(true, true));
        assert!(tracker.depth_behind);
        assert!(tracker.on_access(false, false));
        assert!(!tracker.depth_behind);
        // Subsequent read on depth side is clean again (elided).
        assert!(!tracker.on_access(false, false));

        let after = stats();
        assert!(after.twin_copies_elided >= before.twin_copies_elided + 3);
        assert_eq!(
            after.twin_dirty_invalidations,
            before.twin_dirty_invalidations + 1
        );
    }
}
