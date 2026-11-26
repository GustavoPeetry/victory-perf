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
    generation: u64,
    epoch: u64,
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

struct SpanningIndex {
    entries: Vec<AuthorityEntry>,
    prefix_max_end: Vec<usize>,
}

impl SpanningIndex {
    const fn new() -> Self {
        Self {
            entries: Vec::new(),
            prefix_max_end: Vec::new(),
        }
    }

    fn rebuild_prefix(&mut self) {
        self.prefix_max_end.resize(self.entries.len(), 0);
        let mut max_end = 0usize;
        for (i, e) in self.entries.iter().enumerate() {
            max_end = max_end.max(e.end);
            self.prefix_max_end[i] = max_end;
        }
    }

    fn insert_sorted(&mut self, entry: AuthorityEntry) {
        let pos = self
            .entries
            .partition_point(|e| (e.start, e.end) <= (entry.start, entry.end));
        self.entries.insert(pos, entry);
        self.rebuild_prefix();
    }

    fn for_each_overlap(
        &self,
        start: usize,
        end: usize,
        mut f: impl FnMut(&AuthorityEntry) -> bool,
    ) {
        let hi = self.entries.partition_point(|e| e.start < end);
        for i in (0..hi).rev() {
            if self.prefix_max_end[i] <= start {
                break;
            }
            if self.entries[i].overlaps(start, end) && !f(&self.entries[i]) {
                break;
            }
        }
    }

    fn for_each_overlap_mut(
        &mut self,
        start: usize,
        end: usize,
        mut f: impl FnMut(&mut AuthorityEntry),
    ) {
        let hi = self.entries.partition_point(|e| e.start < end);
        for i in (0..hi).rev() {
            if self.prefix_max_end[i] <= start {
                break;
            }
            if self.entries[i].overlaps(start, end) {
                f(&mut self.entries[i]);
            }
        }
    }
}

struct AuthorityTable {
    shards: [Shard; SHARDS],
    /// Sorted cross-shard interval index with prefix `max_end` for O(log N + k) overlap queries.
    spanning: RwLock<SpanningIndex>,
}

impl AuthorityTable {
    const fn new() -> Self {
        Self {
            shards: [const { Shard::new() }; SHARDS],
            spanning: RwLock::new(SpanningIndex::new()),
        }
    }
}

static TABLE: AuthorityTable = AuthorityTable::new();
static SPANNING_COUNT: AtomicU64 = AtomicU64::new(0);
static NEXT_GENERATION: AtomicU64 = AtomicU64::new(1);
static CURRENT_EPOCH: AtomicU64 = AtomicU64::new(1);

static PROMOTED_GPU_AUTH: AtomicU64 = AtomicU64::new(0);
static DEMOTED_TO_SHARED: AtomicU64 = AtomicU64::new(0);
static CPU_WRITE_DEMOTES: AtomicU64 = AtomicU64::new(0);
static EPOCH_INVALIDATIONS: AtomicU64 = AtomicU64::new(0);
static PARTIAL_PAGE_SPLITS: AtomicU64 = AtomicU64::new(0);
static AMBIGUOUS_FALLBACKS: AtomicU64 = AtomicU64::new(0);
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
    pub cpu_write_demotes: u64,
    pub epoch_invalidations: u64,
    pub partial_page_splits: u64,
    pub ambiguous_fallbacks: u64,
    pub current_epoch: u64,
    pub generation_counter: u64,
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
fn spans_multiple_shards(start: usize, end: usize) -> bool {
    start < end && (start >> 16) != (end.saturating_sub(1) >> 16)
}

#[inline]
fn for_each_shard(start: usize, end: usize, mut f: impl FnMut(usize) -> std::ops::ControlFlow<()>) {
    if start >= end {
        return;
    }
    let first = start >> 16;
    let last = end.saturating_sub(1) >> 16;
    let span = last.saturating_sub(first);
    if span + 1 >= SHARDS {
        for idx in 0..SHARDS {
            if f(idx).is_break() {
                break;
            }
        }
        return;
    }
    for p in first..=last {
        if f(p & (SHARDS - 1)).is_break() {
            break;
        }
    }
}

/// Adjust the `SparseTiled` counter without inserting textures into the buffer VA index.
#[inline]
pub(crate) fn note_sparse_texture_delta(delta: isize) {
    if delta > 0 {
        CLASS_COUNTS[AuthorityClass::SparseTiled as usize]
            .fetch_add(delta as u64, Ordering::Relaxed);
    } else if delta < 0 {
        let mag = delta.unsigned_abs() as u64;
        let _ = CLASS_COUNTS[AuthorityClass::SparseTiled as usize].fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |v| Some(v.saturating_sub(mag)),
        );
    }
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

#[inline]
fn authority_table_active() -> bool {
    enabled() || cfg!(test)
}

/// Register or update a title VA range `[start, start + len)` with its initial class.
pub fn register_range(start: usize, len: usize, class: AuthorityClass, readback_verified: bool) {
    if len == 0 || !authority_table_active() {
        return;
    }
    let end = start.saturating_add(len);
    let epoch = CURRENT_EPOCH.load(Ordering::Acquire);
    let is_spanning = spans_multiple_shards(start, end);
    if is_spanning {
        let mut span = TABLE.spanning.write().unwrap();
        let hi = span.entries.partition_point(|e| e.start <= start);
        for e in span.entries[..hi].iter_mut().rev() {
            if e.start < start {
                break;
            }
            if e.start == start && e.end == end {
                if e.demoted_once
                    && matches!(
                        class,
                        AuthorityClass::GpuAuthoritative | AuthorityClass::GpuOnly
                    )
                {
                    return;
                }
                let prev = e.class;
                e.class = class;
                e.readback_verified |= readback_verified;
                e.epoch = epoch;
                if prev != class {
                    record_class_delta(Some(prev), Some(class));
                }
                return;
            }
        }
        let generation = NEXT_GENERATION.fetch_add(1, Ordering::AcqRel);
        span.insert_sorted(AuthorityEntry {
            start,
            end,
            class,
            gpu_writes: 0,
            cpu_reads: 0,
            demoted_once: false,
            readback_verified,
            generation,
            epoch,
        });
        SPANNING_COUNT.fetch_add(1, Ordering::Release);
        record_class_delta(None, Some(class));
        return;
    }

    let mut entries = TABLE.shards[shard_index(start)].entries.write().unwrap();
    if let Some(existing) = entries
        .iter_mut()
        .find(|e| e.start == start && e.end == end)
    {
        if existing.demoted_once
            && matches!(
                class,
                AuthorityClass::GpuAuthoritative | AuthorityClass::GpuOnly
            )
        {
            return;
        }
        let prev = existing.class;
        existing.class = class;
        existing.readback_verified |= readback_verified;
        existing.epoch = epoch;
        if prev != class {
            record_class_delta(Some(prev), Some(class));
        }
        return;
    }

    let generation = NEXT_GENERATION.fetch_add(1, Ordering::AcqRel);
    entries.push(AuthorityEntry {
        start,
        end,
        class,
        gpu_writes: 0,
        cpu_reads: 0,
        demoted_once: false,
        readback_verified,
        generation,
        epoch,
    });
    record_class_delta(None, Some(class));
}

/// Remove a destroyed or unmapped VA range `[start, start + len)`, advancing the
/// epoch so future allocations reusing this VA cannot inherit stale GPU authority.
pub fn unregister_range(start: usize, len: usize) {
    if len == 0 || !authority_table_active() {
        return;
    }
    let end = start.saturating_add(len);
    let epoch = CURRENT_EPOCH.fetch_add(1, Ordering::AcqRel) + 1;
    let mut removed_any = false;

    for_each_shard(start, end, |idx| {
        let mut entries = TABLE.shards[idx].entries.write().unwrap();
        let mut i = 0;
        while i < entries.len() {
            if entries[i].start == start && entries[i].end == end {
                let removed = entries.swap_remove(i);
                record_class_delta(Some(removed.class), None);
                removed_any = true;
            } else {
                if entries[i].overlaps(start, end) {
                    entries[i].epoch = epoch;
                }
                i += 1;
            }
        }
        std::ops::ControlFlow::Continue(())
    });

    if spans_multiple_shards(start, end) || SPANNING_COUNT.load(Ordering::Acquire) > 0 {
        let mut span = TABLE.spanning.write().unwrap();
        let mut removed_count = 0u64;
        let mut i = 0;
        let mut modified = false;
        while i < span.entries.len() {
            if span.entries[i].start == start && span.entries[i].end == end {
                let removed = span.entries.remove(i);
                record_class_delta(Some(removed.class), None);
                removed_count += 1;
                removed_any = true;
                modified = true;
            } else {
                if span.entries[i].overlaps(start, end) {
                    span.entries[i].epoch = epoch;
                }
                i += 1;
            }
        }
        if removed_count > 0 {
            let _ = SPANNING_COUNT.fetch_update(Ordering::Release, Ordering::Relaxed, |v| {
                Some(v.saturating_sub(removed_count))
            });
        }
        if modified {
            span.rebuild_prefix();
        }
    }
    if removed_any {
        EPOCH_INVALIDATIONS.fetch_add(1, Ordering::Relaxed);
    }
}

#[inline]
fn combine_class(found: &mut Option<AuthorityClass>, class: AuthorityClass) {
    *found = Some(match (*found, class) {
        (None, c) => c,
        (Some(AuthorityClass::SharedCoherent), _) | (_, AuthorityClass::SharedCoherent) => {
            AuthorityClass::SharedCoherent
        }
        (Some(a), b) if a == b => a,
        _ => {
            AMBIGUOUS_FALLBACKS.fetch_add(1, Ordering::Relaxed);
            AuthorityClass::SharedCoherent
        }
    });
}

#[inline]
fn class_is_skippable(class: AuthorityClass) -> bool {
    matches!(
        class,
        AuthorityClass::GpuAuthoritative
            | AuthorityClass::GpuOnly
            | AuthorityClass::CpuToGpuStreamed
            | AuthorityClass::TransientUpload
    )
}

/// Lookup authority class covering `[start, start + len)`.
/// Returns `SharedCoherent` when unknown or when any overlapping range requires coherence.
pub fn class_for(start: usize, len: usize) -> AuthorityClass {
    if !enabled() || len == 0 {
        return AuthorityClass::SharedCoherent;
    }
    let end = start.saturating_add(len);
    let mut found: Option<AuthorityClass> = None;

    for_each_shard(start, end, |idx| {
        let entries = TABLE.shards[idx].entries.read().unwrap();
        for e in entries.iter() {
            if e.overlaps(start, end) {
                combine_class(&mut found, e.class);
                if found == Some(AuthorityClass::SharedCoherent) {
                    return std::ops::ControlFlow::Break(());
                }
            }
        }
        std::ops::ControlFlow::Continue(())
    });
    if found != Some(AuthorityClass::SharedCoherent) && SPANNING_COUNT.load(Ordering::Acquire) > 0 {
        let span = TABLE.spanning.read().unwrap();
        span.for_each_overlap(start, end, |e| {
            combine_class(&mut found, e.class);
            found != Some(AuthorityClass::SharedCoherent)
        });
    }

    found.unwrap_or(AuthorityClass::SharedCoherent)
}

#[inline]
fn entry_needs_gpu_write_update(
    e: &AuthorityEntry,
    readback_flagged: bool,
    readback_proven_complete: bool,
) -> bool {
    if readback_flagged {
        !matches!(e.class, AuthorityClass::SharedCoherent)
    } else {
        readback_proven_complete
            && e.class == AuthorityClass::SharedCoherent
            && !e.demoted_once
            && e.cpu_reads == 0
    }
}

#[inline]
fn apply_gpu_write_to_entry(
    e: &mut AuthorityEntry,
    readback_flagged: bool,
    readback_proven_complete: bool,
    threshold: u32,
) {
    if readback_flagged {
        if !matches!(e.class, AuthorityClass::SharedCoherent) {
            let prev = e.class;
            e.class = AuthorityClass::SharedCoherent;
            record_class_delta(Some(prev), Some(AuthorityClass::SharedCoherent));
        }
        return;
    }
    if !readback_proven_complete
        || e.class != AuthorityClass::SharedCoherent
        || e.demoted_once
        || e.cpu_reads != 0
    {
        return;
    }
    e.gpu_writes = e.gpu_writes.saturating_add(1);
    if e.gpu_writes >= threshold {
        e.class = AuthorityClass::GpuAuthoritative;
        record_class_delta(
            Some(AuthorityClass::SharedCoherent),
            Some(AuthorityClass::GpuAuthoritative),
        );
        PROMOTED_GPU_AUTH.fetch_add(1, Ordering::Relaxed);
    }
}

/// Observe a GPU UAV / copy write to `[start, start + len)` and return whether
/// the before-twin compare can be elided in a single traversal.
pub(crate) fn observe_gpu_write_and_check(
    start: usize,
    len: usize,
    readback_flagged: bool,
    readback_proven_complete: bool,
) -> bool {
    if !enabled() || len == 0 {
        return false;
    }
    let end = start.saturating_add(len);
    let threshold = if strict_mode() { 4 } else { 2 };
    let mut found: Option<AuthorityClass> = None;

    for_each_shard(start, end, |idx| {
        let mut needs_write = false;
        {
            let entries = TABLE.shards[idx].entries.read().unwrap();
            for e in entries.iter() {
                if e.overlaps(start, end) {
                    if entry_needs_gpu_write_update(e, readback_flagged, readback_proven_complete) {
                        needs_write = true;
                        break;
                    }
                    combine_class(&mut found, e.class);
                }
            }
        }
        if needs_write {
            let mut entries = TABLE.shards[idx].entries.write().unwrap();
            for e in entries.iter_mut() {
                if e.overlaps(start, end) {
                    apply_gpu_write_to_entry(
                        e,
                        readback_flagged,
                        readback_proven_complete,
                        threshold,
                    );
                    combine_class(&mut found, e.class);
                }
            }
        }
        std::ops::ControlFlow::Continue(())
    });

    if SPANNING_COUNT.load(Ordering::Acquire) > 0 {
        let mut needs_write = false;
        {
            let span = TABLE.spanning.read().unwrap();
            span.for_each_overlap(start, end, |e| {
                if entry_needs_gpu_write_update(e, readback_flagged, readback_proven_complete) {
                    needs_write = true;
                    return false;
                }
                combine_class(&mut found, e.class);
                true
            });
        }
        if needs_write {
            let mut span = TABLE.spanning.write().unwrap();
            span.for_each_overlap_mut(start, end, |e| {
                apply_gpu_write_to_entry(e, readback_flagged, readback_proven_complete, threshold);
                combine_class(&mut found, e.class);
            });
        }
    }

    !readback_flagged && found.is_some_and(class_is_skippable)
}

/// Observe a GPU UAV / copy write to `[start, start + len)`.
pub fn observe_gpu_write(
    start: usize,
    len: usize,
    readback_flagged: bool,
    readback_proven_complete: bool,
) {
    let _ = observe_gpu_write_and_check(start, len, readback_flagged, readback_proven_complete);
}

#[inline]
fn entry_needs_demote(e: &AuthorityEntry) -> bool {
    !e.demoted_once
        || e.cpu_reads == 0
        || matches!(
            e.class,
            AuthorityClass::GpuAuthoritative | AuthorityClass::GpuOnly
        )
}

#[inline]
fn apply_demote_to_entry(e: &mut AuthorityEntry, count_cpu_write: bool, demoted: &mut bool) {
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
        if count_cpu_write {
            CPU_WRITE_DEMOTES.fetch_add(1, Ordering::Relaxed);
        }
        *demoted = true;
    }
}

fn demote_overlap(start: usize, end: usize, count_cpu_write: bool) -> bool {
    let mut demoted = false;
    for_each_shard(start, end, |idx| {
        let mut needs_write = false;
        {
            let entries = TABLE.shards[idx].entries.read().unwrap();
            for e in entries.iter() {
                if e.overlaps(start, end) && entry_needs_demote(e) {
                    needs_write = true;
                    break;
                }
            }
        }
        if needs_write {
            let mut entries = TABLE.shards[idx].entries.write().unwrap();
            for e in entries.iter_mut() {
                if e.overlaps(start, end) {
                    apply_demote_to_entry(e, count_cpu_write, &mut demoted);
                }
            }
        }
        std::ops::ControlFlow::Continue(())
    });

    if SPANNING_COUNT.load(Ordering::Acquire) > 0 {
        let mut needs_write = false;
        {
            let span = TABLE.spanning.read().unwrap();
            span.for_each_overlap(start, end, |e| {
                if entry_needs_demote(e) {
                    needs_write = true;
                    return false;
                }
                true
            });
        }
        if needs_write {
            let mut span = TABLE.spanning.write().unwrap();
            span.for_each_overlap_mut(start, end, |e| {
                apply_demote_to_entry(e, count_cpu_write, &mut demoted);
            });
        }
    }
    demoted
}

/// Observe a CPU read / copy-back / unexpected shadow probe overlapping `[start, start + len)`.
///
/// Immediately demotes `GpuAuthoritative` or `GpuOnly` entries back to
/// `SharedCoherent` and locks them against re-promotion for their lifetime.
pub fn observe_cpu_read_or_copy(start: usize, len: usize) -> bool {
    if !enabled() || len == 0 {
        return false;
    }
    demote_overlap(start, start.saturating_add(len), false)
}

/// Observe a CPU write overlapping `[start, start + len)` (e.g., partial 4 KiB
/// page overwrite or direct title buffer write). Demotes overlapping GPU authority.
pub fn observe_cpu_write(start: usize, len: usize) -> bool {
    if !enabled() || len == 0 {
        return false;
    }
    demote_overlap(start, start.saturating_add(len), true)
}

/// Read-only diagnostic check for readback validation: returns true if a readback
/// interval unexpectedly overlaps a `GpuAuthoritative` or `GpuOnly` entry.
#[inline]
pub fn is_gpu_authoritative_conflict(start: usize, len: usize) -> bool {
    if !enabled() || len == 0 {
        return false;
    }
    matches!(
        class_for(start, len),
        AuthorityClass::GpuAuthoritative | AuthorityClass::GpuOnly
    )
}

#[inline]
pub fn record_partial_page_split() {
    PARTIAL_PAGE_SPLITS.fetch_add(1, Ordering::Relaxed);
}

#[inline]
pub fn record_ambiguous_fallback() {
    AMBIGUOUS_FALLBACKS.fetch_add(1, Ordering::Relaxed);
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
        cpu_authoritative: CLASS_COUNTS[AuthorityClass::CpuAuthoritative as usize]
            .load(Ordering::Relaxed),
        cpu_to_gpu_streamed: CLASS_COUNTS[AuthorityClass::CpuToGpuStreamed as usize]
            .load(Ordering::Relaxed),
        gpu_authoritative: CLASS_COUNTS[AuthorityClass::GpuAuthoritative as usize]
            .load(Ordering::Relaxed),
        gpu_only: CLASS_COUNTS[AuthorityClass::GpuOnly as usize].load(Ordering::Relaxed),
        shared_coherent: CLASS_COUNTS[AuthorityClass::SharedCoherent as usize]
            .load(Ordering::Relaxed),
        sparse_tiled: CLASS_COUNTS[AuthorityClass::SparseTiled as usize].load(Ordering::Relaxed),
        transient_upload: CLASS_COUNTS[AuthorityClass::TransientUpload as usize]
            .load(Ordering::Relaxed),
        promoted_gpu_auth: PROMOTED_GPU_AUTH.load(Ordering::Relaxed),
        demoted_to_shared: DEMOTED_TO_SHARED.load(Ordering::Relaxed),
        skipped_before_twins: SKIPPED_BEFORE_TWINS.load(Ordering::Relaxed),
        skipped_card_compares: SKIPPED_CARD_COMPARES.load(Ordering::Relaxed),
        cpu_write_demotes: CPU_WRITE_DEMOTES.load(Ordering::Relaxed),
        epoch_invalidations: EPOCH_INVALIDATIONS.load(Ordering::Relaxed),
        partial_page_splits: PARTIAL_PAGE_SPLITS.load(Ordering::Relaxed),
        ambiguous_fallbacks: AMBIGUOUS_FALLBACKS.load(Ordering::Relaxed),
        current_epoch: CURRENT_EPOCH.load(Ordering::Acquire),
        generation_counter: NEXT_GENERATION.load(Ordering::Acquire),
    }
}

/// Returns the allocation generation of the matching entry `[start, start + len)`, if active.
pub fn range_generation(start: usize, len: usize) -> Option<u64> {
    let end = start.saturating_add(len);
    if spans_multiple_shards(start, end) {
        let span = TABLE.spanning.read().unwrap();
        let hi = span.entries.partition_point(|e| e.start <= start);
        for e in span.entries[..hi].iter().rev() {
            if e.start < start {
                break;
            }
            if e.end == end {
                return Some(e.generation);
            }
        }
        None
    } else {
        TABLE.shards[shard_index(start)]
            .entries
            .read()
            .unwrap()
            .iter()
            .find(|e| e.start == start && e.end == end)
            .map(|e| e.generation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_authority_reports_shared_coherent() {
        // Default test process has VICTORY_REA_GPU_AUTHORITY unset.
        register_range(0x1000_0000, 0x1000, AuthorityClass::GpuOnly, true);
        assert_eq!(
            class_for(0x1000_0000, 0x1000),
            AuthorityClass::SharedCoherent
        );
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

    #[test]
    fn gpu_authority_promotes_skips_and_demotes_cleanly_under_flag() {
        const CHILD: &str = "VICTORY_TEST_AUTHORITY_PROMOTE_CHILD";
        const OK: &str = "gpu authority promote/demote assertions passed";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "authority::tests::gpu_authority_promotes_skips_and_demotes_cleanly_under_flag",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .env("VICTORY_REA_GPU_AUTHORITY", "1")
                .output()
                .expect("spawn authority test child");
            assert!(
                output.status.success() && String::from_utf8_lossy(&output.stdout).contains(OK),
                "stdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }

        assert!(enabled());
        let unverified = 0x3000_0000usize;
        register_range(unverified, 0x10_000, AuthorityClass::SharedCoherent, false);
        for _ in 0..8 {
            observe_gpu_write(unverified, 0x10_000, false, false);
        }
        assert_eq!(
            class_for(unverified, 0x10_000),
            AuthorityClass::SharedCoherent,
            "must never promote before readback tagging is proven"
        );
        assert!(!can_skip_gpu_compare(unverified, 0x10_000, false));
        unregister_range(unverified, 0x10_000);

        let addr = 0x4000_0000usize;
        register_range(addr, 0x10_000, AuthorityClass::SharedCoherent, true);
        assert!(!can_skip_gpu_compare(addr, 0x10_000, false));

        // After sustained non-readback GPU writes, range promotes to GpuAuthoritative.
        observe_gpu_write(addr, 0x10_000, false, true);
        observe_gpu_write(addr, 0x10_000, false, true);
        assert_eq!(class_for(addr, 0x10_000), AuthorityClass::GpuAuthoritative);
        assert!(can_skip_gpu_compare(addr, 0x10_000, false));

        // Any subsequent CPU read/copy demotes permanently back to SharedCoherent.
        assert!(observe_cpu_read_or_copy(addr, 0x1000));
        assert_eq!(class_for(addr, 0x10_000), AuthorityClass::SharedCoherent);
        assert!(!can_skip_gpu_compare(addr, 0x10_000, false));
        observe_gpu_write(addr, 0x10_000, false, true);
        observe_gpu_write(addr, 0x10_000, false, true);
        assert_eq!(class_for(addr, 0x10_000), AuthorityClass::SharedCoherent);

        unregister_range(addr, 0x10_000);
        println!("{OK}");
    }

    #[test]
    fn cross_shard_demotion_and_epoch_reuse_are_safe() {
        const CHILD: &str = "VICTORY_TEST_AUTHORITY_CROSS_SHARD_CHILD";
        const OK: &str = "cross-shard authority and epoch reuse assertions passed";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "authority::tests::cross_shard_demotion_and_epoch_reuse_are_safe",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .env("VICTORY_REA_GPU_AUTHORITY", "1")
                .output()
                .expect("spawn cross-shard test child");
            assert!(
                output.status.success() && String::from_utf8_lossy(&output.stdout).contains(OK),
                "stdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }

        // Large spanning buffer (256 KiB across 4 64 KiB shards).
        let base = 0x5000_0000usize;
        let len = 0x4_0000usize;
        register_range(base, len, AuthorityClass::SharedCoherent, true);
        observe_gpu_write(base + 0x2_0000, 0x1000, false, true);
        observe_gpu_write(base + 0x2_0000, 0x1000, false, true);
        assert_eq!(
            class_for(base + 0x3_0000, 0x1000),
            AuthorityClass::GpuAuthoritative,
            "spanning entry must promote uniformly across all shards"
        );

        // A CPU write to offset +0x2_0000 (different 64K shard than `base`) MUST demote it!
        assert!(observe_cpu_write(base + 0x2_0000, 0x1000));
        assert_eq!(class_for(base, len), AuthorityClass::SharedCoherent);
        assert!(!can_skip_gpu_compare(base + 0x3_0000, 0x1000, false));

        // Ambiguous overlapping entries (GpuOnly + SharedCoherent) must stay SharedCoherent.
        let sub_a = base + 0x1_0000;
        register_range(sub_a, 0x1000, AuthorityClass::GpuOnly, true);
        assert_eq!(class_for(sub_a, 0x1000), AuthorityClass::SharedCoherent);

        // Unregistering frees the VA cleanly so reusing the VA in a new frame/generation works.
        let before_stats = stats();
        let old_gen = range_generation(base, len);
        unregister_range(sub_a, 0x1000);
        unregister_range(base, len);
        let after_stats = stats();
        assert!(after_stats.epoch_invalidations >= before_stats.epoch_invalidations + 2);
        assert_eq!(range_generation(base, len), None);

        register_range(base, 0x1000, AuthorityClass::SharedCoherent, true);
        assert_ne!(old_gen, range_generation(base, 0x1000));
        assert_eq!(class_for(base, 0x1000), AuthorityClass::SharedCoherent);
        observe_gpu_write(base, 0x1000, false, true);
        observe_gpu_write(base, 0x1000, false, true);
        assert_eq!(class_for(base, 0x1000), AuthorityClass::GpuAuthoritative);
        unregister_range(base, 0x1000);
        println!("{OK}");
    }

    #[test]
    fn spanning_index_scales_with_thousands_of_entries_and_read_first_demotion() {
        const CHILD: &str = "VICTORY_TEST_AUTHORITY_SPANNING_BENCH_CHILD";
        const OK: &str = "spanning index bench passed";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "authority::tests::spanning_index_scales_with_thousands_of_entries_and_read_first_demotion",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .env("VICTORY_REA_GPU_AUTHORITY", "1")
                .output()
                .expect("spawn spanning bench child");
            assert!(
                output.status.success() && String::from_utf8_lossy(&output.stdout).contains(OK),
                "stdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }

        // Register 4,000 multi-shard 1 MiB ranges into `SpanningIndex`.
        for i in 0..4_000usize {
            let start = 0x1_0000_0000 + i * 0x20_0000;
            register_range(start, 0x10_0000, AuthorityClass::SharedCoherent, true);
        }
        // Promote the 2,000th entry and keep it read-only under subsequent steady GPU writes.
        let target = 0x1_0000_0000 + 2000 * 0x20_0000;
        assert!(!observe_gpu_write_and_check(target, 4096, false, true));
        assert!(observe_gpu_write_and_check(target, 4096, false, true));

        let started = std::time::Instant::now();
        for i in 0..50_000usize {
            let probe = 0x1_0000_0000 + (i % 4_000) * 0x20_0000;
            let _ = observe_gpu_write_and_check(probe, 256, false, false);
        }
        let elapsed = started.elapsed();
        eprintln!(
            "50,000 SpanningIndex queries over 4,000 entries: {:?}",
            elapsed
        );
        assert!(elapsed < std::time::Duration::from_secs(2));

        // Clean up.
        for i in 0..4_000usize {
            let start = 0x1_0000_0000 + i * 0x20_0000;
            unregister_range(start, 0x10_0000);
        }
        println!("{OK}");
    }
}
