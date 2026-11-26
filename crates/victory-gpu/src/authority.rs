//! Semantic resource authority state machine (CPU vs GPU ownership).
//!
//! When `VICTORY_REA_GPU_AUTHORITY=1` is set, buffers and textures that live
//! purely in GPU workflows (render targets, compute-only UAVs) or strict
//! one-way streaming uploads can avoid redundant Software DSM twins and
//! `CARD_COMPARE` dispatches while demoting immediately and safely back to
//! [`AuthorityClass::SharedCoherent`] on any unexpected CPU read/copy access.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{OnceLock, RwLock};

/// Semantic memory authority classification across desktop RAM and VRAM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum AuthorityClass {
    /// CPU holds canonical truth; host reads directly from title memory.
    CpuAuthoritative = 0,
    /// One-way CPU -> GPU streaming upload buffer; no GPU->CPU readback shadow needed.
    CpuToGpuStreamed = 1,
    /// GPU holds canonical truth (RTV/DSV/UAV outputs without observed CPU consumption);
    /// safe to skip pre-flight Card before-twin & compare dispatches while verified.
    GpuAuthoritative = 2,
    /// Strictly GPU-private target / transient UAV; zero CPU access expected.
    GpuOnly = 3,
    /// Bidirectional CPU <-> GPU coherence maintained by Software DSM (`Shadow` / `Card`).
    SharedCoherent = 4,
    /// D3D12 Reserved Resource whose mip residency is managed on-demand in VRAM tiles.
    SparseTiled = 5,
    /// Short-lived fence-retired upload ring slice.
    TransientUpload = 6,
}

#[derive(Clone, Copy, Debug)]
struct AuthorityEntry {
    start: usize,
    end: usize,
    class: AuthorityClass,
    gpu_writes: u32,
    cpu_reads: u32,
    demoted_once: bool,
    readback_verified: bool,
}

impl AuthorityEntry {
    #[inline]
    fn overlaps(&self, start: usize, end: usize) -> bool {
        self.start < end && start < self.end
    }
}

const SHARDS: usize = 64;

struct Shard {
    entries: RwLock<Vec<AuthorityEntry>>,
}

impl Shard {
    const fn new() -> Self {
        Self {
            entries: RwLock::new(Vec::new()),
        }
    }
}

struct AuthorityTable {
    shards: [Shard; SHARDS],
}

impl AuthorityTable {
    const fn new() -> Self {
        Self {
            shards: [const { Shard::new() }; SHARDS],
        }
    }
}

static TABLE: AuthorityTable = AuthorityTable::new();

static PROMOTED_GPU_AUTH: AtomicU64 = AtomicU64::new(0);
static DEMOTED_TO_SHARED: AtomicU64 = AtomicU64::new(0);
static SKIPPED_BEFORE_TWINS: AtomicU64 = AtomicU64::new(0);
static SKIPPED_CARD_COMPARES: AtomicU64 = AtomicU64::new(0);
static CLASS_COUNTS: [AtomicU64; 7] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

/// Aggregate counters exported to REA telemetry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AuthorityStats {
    pub enabled: bool,
    pub strict: bool,
    pub cpu_authoritative: u64,
    pub cpu_to_gpu_streamed: u64,
    pub gpu_authoritative: u64,
    pub gpu_only: u64,
    pub shared_coherent: u64,
    pub sparse_tiled: u64,
    pub transient_upload: u64,
    pub promoted_gpu_auth: u64,
    pub demoted_to_shared: u64,
    pub skipped_before_twins: u64,
    pub skipped_card_compares: u64,
}

fn parse_bool(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| {
            let t = v.trim();
            t == "1" || t.eq_ignore_ascii_case("true") || t.eq_ignore_ascii_case("yes")
        })
        .unwrap_or(false)
}

/// Whether `VICTORY_REA_GPU_AUTHORITY=1` is enabled (OFF by default).
pub fn enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        let on = parse_bool("VICTORY_REA_GPU_AUTHORITY");
        if on {
            tracing::info!("REA-002 GPU-first memory authority enabled");
        }
        on
    })
}

/// Strict fallback bias (`VICTORY_REA_AUTHORITY_STRICT=1`): keeps ambiguous UAV
/// ranges in `SharedCoherent` until readback creator provenance is confirmed.
pub fn strict_mode() -> bool {
    static STRICT: OnceLock<bool> = OnceLock::new();
    *STRICT.get_or_init(|| parse_bool("VICTORY_REA_AUTHORITY_STRICT"))
}

#[inline]
fn shard_index(start: usize) -> usize {
    // Cluster by 64 KiB region so page-aligned ranges mostly hit one shard.
    (start >> 16) & (SHARDS - 1)
}

#[inline]
fn record_class_delta(old: Option<AuthorityClass>, new: Option<AuthorityClass>) {
    if let Some(c) = old {
        let idx = c as usize;
        let _ = CLASS_COUNTS[idx].fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
            Some(v.saturating_sub(1))
        });
    }
    if let Some(c) = new {
        CLASS_COUNTS[c as usize].fetch_add(1, Ordering::Relaxed);
    }
}

/// Register or update a title VA range `[start, start + len)` with its initial class.
pub fn register_range(start: usize, len: usize, class: AuthorityClass, readback_verified: bool) {
    if len == 0 {
        return;
    }
    let end = start.saturating_add(len);
    let idx = shard_index(start);
    let mut entries = TABLE.shards[idx].entries.write().unwrap();
    if let Some(existing) = entries.iter_mut().find(|e| e.start == start && e.end == end) {
        if existing.demoted_once && matches!(class, AuthorityClass::GpuAuthoritative | AuthorityClass::GpuOnly) {
            return;
        }
        let prev = existing.class;
        existing.class = class;
        existing.readback_verified |= readback_verified;
        if prev != class {
            record_class_delta(Some(prev), Some(class));
        }
        return;
    }

    entries.push(AuthorityEntry {
        start,
        end,
        class,
        gpu_writes: 0,
        cpu_reads: 0,
        demoted_once: false,
        readback_verified,
    });
    record_class_delta(None, Some(class));
}

/// Remove a destroyed or unmapped VA range `[start, start + len)`.
pub fn unregister_range(start: usize, len: usize) {
    if len == 0 {
        return;
    }
    let end = start.saturating_add(len);
    let idx = shard_index(start);
    let mut entries = TABLE.shards[idx].entries.write().unwrap();
    let mut i = 0;
    while i < entries.len() {
        if entries[i].start == start && entries[i].end == end {
            let removed = entries.swap_remove(i);
            record_class_delta(Some(removed.class), None);
        } else {
            i += 1;
        }
    }
}

/// Lookup authority class covering `[start, start + len)`.
/// Returns `SharedCoherent` when unknown or when any overlapping range requires coherence.
pub fn class_for(start: usize, len: usize) -> AuthorityClass {
    if !enabled() || len == 0 {
        return AuthorityClass::SharedCoherent;
    }
    let end = start.saturating_add(len);
    // Ranges larger than 64 KiB can touch adjacent shards; scan the primary shard
    // and check neighbor shards if it spans a 64 KiB boundary.
    let first_shard = shard_index(start);
    let last_shard = shard_index(end.saturating_sub(1));
    let mut found: Option<AuthorityClass> = None;

    let mut visit = |idx: usize| {
        let entries = TABLE.shards[idx].entries.read().unwrap();
        for e in entries.iter() {
            if e.overlaps(start, end) {
                found = Some(match (found, e.class) {
                    (None, c) => c,
                    (Some(AuthorityClass::SharedCoherent), _)
                    | (_, AuthorityClass::SharedCoherent) => AuthorityClass::SharedCoherent,
                    (Some(a), b) if a == b => a,
                    _ => AuthorityClass::SharedCoherent,
                });
            }
        }
    };

    visit(first_shard);
    if last_shard != first_shard {
        // Bounded check on the endpoint shard and up to 4 intervening shards for medium ranges.
        let mut s = first_shard;
        let mut steps = 0usize;
        while s != last_shard && steps < 4 {
            s = (s + 1) & (SHARDS - 1);
            visit(s);
            steps += 1;
        }
        if s != last_shard {
            visit(last_shard);
        }
    }

    found.unwrap_or(AuthorityClass::SharedCoherent)
}

/// Observe a GPU UAV / copy write to `[start, start + len)`.
///
/// Only promotes a `SharedCoherent` range to `GpuAuthoritative` if `enabled()` is
/// true, `readback_flagged` is false, `cpu_reads == 0`, the range has never been
/// demoted, and it has sustained at least 2 pure-GPU writes without CPU readback.
pub fn observe_gpu_write(start: usize, len: usize, readback_flagged: bool, readback_proven_complete: bool) {
    if !enabled() || len == 0 {
        return;
    }
    let end = start.saturating_add(len);
    let idx = shard_index(start);
    let mut entries = TABLE.shards[idx].entries.write().unwrap();
    for e in entries.iter_mut() {
        if !e.overlaps(start, end) {
            continue;
        }
        if readback_flagged {
            if !matches!(e.class, AuthorityClass::SharedCoherent) {
                let prev = e.class;
                e.class = AuthorityClass::SharedCoherent;
                record_class_delta(Some(prev), Some(AuthorityClass::SharedCoherent));
            }
            continue;
        }
        e.gpu_writes = e.gpu_writes.saturating_add(1);
        let threshold = if strict_mode() && !readback_proven_complete {
            u32::MAX
        } else if readback_proven_complete {
            2
        } else {
            4
        };
        if e.class == AuthorityClass::SharedCoherent
            && !e.demoted_once
            && e.cpu_reads == 0
            && e.gpu_writes >= threshold
        {
            e.class = AuthorityClass::GpuAuthoritative;
            record_class_delta(
                Some(AuthorityClass::SharedCoherent),
                Some(AuthorityClass::GpuAuthoritative),
            );
            PROMOTED_GPU_AUTH.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Observe a CPU read / copy-back / unexpected shadow probe overlapping `[start, start + len)`.
///
/// Immediately demotes `GpuAuthoritative` or `GpuOnly` entries back to
/// `SharedCoherent` and locks them against re-promotion for their lifetime.
pub fn observe_cpu_read_or_copy(start: usize, len: usize) -> bool {
    if !enabled() || len == 0 {
        return false;
    }
    let end = start.saturating_add(len);
    let idx = shard_index(start);
    let mut entries = TABLE.shards[idx].entries.write().unwrap();
    let mut demoted = false;
    for e in entries.iter_mut() {
        if !e.overlaps(start, end) {
            continue;
        }
        e.cpu_reads = e.cpu_reads.saturating_add(1);
        e.demoted_once = true;
        if matches!(
            e.class,
            AuthorityClass::GpuAuthoritative | AuthorityClass::GpuOnly
        ) {
            let prev = e.class;
            e.class = AuthorityClass::SharedCoherent;
            record_class_delta(Some(prev), Some(AuthorityClass::SharedCoherent));
            DEMOTED_TO_SHARED.fetch_add(1, Ordering::Relaxed);
            demoted = true;
        }
    }
    demoted
}

/// Check whether a GPU shadow / Card before-twin & compare can be skipped safely.
#[inline]
pub fn can_skip_gpu_compare(start: usize, len: usize, readback_flagged: bool) -> bool {
    if !enabled() || readback_flagged || len == 0 {
        return false;
    }
    matches!(
        class_for(start, len),
        AuthorityClass::GpuAuthoritative
            | AuthorityClass::GpuOnly
            | AuthorityClass::CpuToGpuStreamed
            | AuthorityClass::TransientUpload
    )
}

#[inline]
pub fn note_skipped_before_twin() {
    SKIPPED_BEFORE_TWINS.fetch_add(1, Ordering::Relaxed);
}

#[inline]
pub fn note_skipped_card_compare() {
    SKIPPED_CARD_COMPARES.fetch_add(1, Ordering::Relaxed);
}

/// Read aggregate authority statistics.
pub fn stats() -> AuthorityStats {
    AuthorityStats {
        enabled: enabled(),
        strict: strict_mode(),
        cpu_authoritative: CLASS_COUNTS[AuthorityClass::CpuAuthoritative as usize].load(Ordering::Relaxed),
        cpu_to_gpu_streamed: CLASS_COUNTS[AuthorityClass::CpuToGpuStreamed as usize].load(Ordering::Relaxed),
        gpu_authoritative: CLASS_COUNTS[AuthorityClass::GpuAuthoritative as usize].load(Ordering::Relaxed),
        gpu_only: CLASS_COUNTS[AuthorityClass::GpuOnly as usize].load(Ordering::Relaxed),
        shared_coherent: CLASS_COUNTS[AuthorityClass::SharedCoherent as usize].load(Ordering::Relaxed),
        sparse_tiled: CLASS_COUNTS[AuthorityClass::SparseTiled as usize].load(Ordering::Relaxed),
        transient_upload: CLASS_COUNTS[AuthorityClass::TransientUpload as usize].load(Ordering::Relaxed),
        promoted_gpu_auth: PROMOTED_GPU_AUTH.load(Ordering::Relaxed),
        demoted_to_shared: DEMOTED_TO_SHARED.load(Ordering::Relaxed),
        skipped_before_twins: SKIPPED_BEFORE_TWINS.load(Ordering::Relaxed),
        skipped_card_compares: SKIPPED_CARD_COMPARES.load(Ordering::Relaxed),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_authority_reports_shared_coherent() {
        // Default test process has VICTORY_REA_GPU_AUTHORITY unset.
        register_range(0x1000_0000, 0x1000, AuthorityClass::GpuOnly, true);
        assert_eq!(class_for(0x1000_0000, 0x1000), AuthorityClass::SharedCoherent);
        assert!(!can_skip_gpu_compare(0x1000_0000, 0x1000, false));
        unregister_range(0x1000_0000, 0x1000);
    }

    #[test]
    fn table_transitions_and_demotion_stick() {
        let base = 0x2000_0000usize;
        register_range(base, 0x2000, AuthorityClass::SharedCoherent, true);
        {
            let idx = shard_index(base);
            let mut entries = TABLE.shards[idx].entries.write().unwrap();
            let e = entries.iter_mut().find(|e| e.start == base).unwrap();
            e.gpu_writes = 3;
            e.class = AuthorityClass::GpuAuthoritative;
            e.demoted_once = true;
            assert_eq!(e.class, AuthorityClass::GpuAuthoritative);
        }
        // Re-registering as GpuAuthoritative after demoted_once is blocked.
        register_range(base, 0x2000, AuthorityClass::GpuAuthoritative, true);
        {
            let idx = shard_index(base);
            let mut entries = TABLE.shards[idx].entries.write().unwrap();
            let e = entries.iter_mut().find(|e| e.start == base).unwrap();
            e.class = AuthorityClass::SharedCoherent;
        }
        register_range(base, 0x2000, AuthorityClass::GpuAuthoritative, true);
        {
            let idx = shard_index(base);
            let entries = TABLE.shards[idx].entries.read().unwrap();
            let e = entries.iter().find(|e| e.start == base).unwrap();
            assert_eq!(e.class, AuthorityClass::SharedCoherent);
        }
        unregister_range(base, 0x2000);
    }
}
