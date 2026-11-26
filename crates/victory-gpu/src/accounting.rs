//! O(1) incremental memory accounting and cached DXGI budget queries.
//!
//! Replaces full-table scans across all ~18k live textures every 5s when
//! `VICTORY_REA_INCREMENTAL_ACCOUNTING=1` is set. Can optionally cross-check
//! incremental totals against full enumeration under `VICTORY_REA_ACCOUNTING_AUDIT=1`.

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

#[repr(align(64))]
struct AlignedCounter(AtomicU64);

#[repr(align(64))]
struct AlignedSignedCounter(AtomicI64);

impl AlignedCounter {
    const fn new() -> Self {
        Self(AtomicU64::new(0))
    }

    #[inline]
    fn load(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }

    #[inline]
    fn add(&self, n: u64) {
        if n != 0 {
            self.0.fetch_add(n, Ordering::Relaxed);
        }
    }

    #[inline]
    fn sub_saturating(&self, n: u64) {
        if n == 0 {
            return;
        }
        let mut cur = self.0.load(Ordering::Relaxed);
        loop {
            let next = cur.saturating_sub(n);
            match self
                .0
                .compare_exchange_weak(cur, next, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => break,
                Err(actual) => cur = actual,
            }
        }
    }
}

/// Texture accounting categories mirrored into [`AccountingSnapshot`].
pub const KIND_SLOT_COUNT: usize = 8;

struct Metrics {
    committed_textures_bytes: AlignedCounter,
    reserved_textures_bytes: AlignedCounter,
    mapped_tile_bytes: AlignedCounter,
    tile_pool_allocated_bytes: AlignedCounter,
    staging_bytes: AlignedCounter,
    shadow_ram_bytes: AlignedCounter,
    shadow_vram_bytes: AlignedCounter,
    upload_ring_bytes: AlignedCounter,
    reclaimable_bytes: AlignedCounter,
    retired_pending_count: AlignedCounter,
    committed_texture_count: AlignedCounter,
    sparse_texture_count: AlignedCounter,
    dxgi_queries_actual: AlignedCounter,
    dxgi_queries_cached: AlignedCounter,
    dxgi_nonlocal_used_bytes: AlignedCounter,
    dxgi_nonlocal_budget_bytes: AlignedCounter,
    audit_runs: AlignedCounter,
    audit_mismatches: AlignedCounter,
    last_audit_drift_bytes: AlignedCounter,
}

impl Metrics {
    const fn new() -> Self {
        Self {
            committed_textures_bytes: AlignedCounter::new(),
            reserved_textures_bytes: AlignedCounter::new(),
            mapped_tile_bytes: AlignedCounter::new(),
            tile_pool_allocated_bytes: AlignedCounter::new(),
            staging_bytes: AlignedCounter::new(),
            shadow_ram_bytes: AlignedCounter::new(),
            shadow_vram_bytes: AlignedCounter::new(),
            upload_ring_bytes: AlignedCounter::new(),
            reclaimable_bytes: AlignedCounter::new(),
            retired_pending_count: AlignedCounter::new(),
            committed_texture_count: AlignedCounter::new(),
            sparse_texture_count: AlignedCounter::new(),
            dxgi_queries_actual: AlignedCounter::new(),
            dxgi_queries_cached: AlignedCounter::new(),
            dxgi_nonlocal_used_bytes: AlignedCounter::new(),
            dxgi_nonlocal_budget_bytes: AlignedCounter::new(),
            audit_runs: AlignedCounter::new(),
            audit_mismatches: AlignedCounter::new(),
            last_audit_drift_bytes: AlignedCounter::new(),
        }
    }
}

static METRICS: Metrics = Metrics::new();

#[derive(Clone, Copy)]
struct CachedBudget {
    captured_at: Instant,
    values: [u64; 4],
}

static BUDGET_CACHE: Mutex<Option<CachedBudget>> = Mutex::new(None);
static BUDGET_UNFLUSHED_LOCAL_DELTA: AlignedSignedCounter = AlignedSignedCounter(AtomicI64::new(0));
const DEFAULT_BUDGET_TTL: Duration = Duration::from_millis(15);
const MAX_UNFLUSHED_BUDGET_DELTA: i64 = 16 << 20;

/// Point-in-time O(1) snapshot of tracked host memory pools.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AccountingSnapshot {
    pub enabled: bool,
    pub audit_enabled: bool,
    pub committed_textures_bytes: u64,
    pub reserved_textures_bytes: u64,
    pub mapped_tile_bytes: u64,
    pub tile_pool_allocated_bytes: u64,
    pub staging_bytes: u64,
    pub shadow_ram_bytes: u64,
    pub shadow_vram_bytes: u64,
    pub upload_ring_bytes: u64,
    pub reclaimable_bytes: u64,
    pub retired_pending_count: u64,
    pub committed_texture_count: u64,
    pub sparse_texture_count: u64,
    pub dxgi_queries_actual: u64,
    pub dxgi_queries_cached: u64,
    pub dxgi_nonlocal_used_bytes: u64,
    pub dxgi_nonlocal_budget_bytes: u64,
    pub audit_runs: u64,
    pub audit_mismatches: u64,
    pub last_audit_drift_bytes: u64,
}

fn parse_flag(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| {
            let t = v.trim();
            t == "1" || t.eq_ignore_ascii_case("true") || t.eq_ignore_ascii_case("yes")
        })
        .unwrap_or(false)
}

/// Whether `VICTORY_REA_INCREMENTAL_ACCOUNTING=1` is enabled.
pub fn enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        let on = parse_flag("VICTORY_REA_INCREMENTAL_ACCOUNTING");
        if on {
            tracing::info!("REA-002 incremental memory accounting enabled");
        }
        on
    })
}

/// Whether `VICTORY_REA_ACCOUNTING_AUDIT=1` verification against full scans is active.
pub fn audit_enabled() -> bool {
    static AUDIT: OnceLock<bool> = OnceLock::new();
    *AUDIT.get_or_init(|| parse_flag("VICTORY_REA_ACCOUNTING_AUDIT"))
}

/// Track creation of a committed host texture (`committed_bytes > 0`) or
/// sparse reserved host texture (`reserved_bytes > 0`).
pub fn on_texture_created(committed_bytes: u64, reserved_bytes: u64, is_sparse: bool) {
    METRICS.committed_textures_bytes.add(committed_bytes);
    METRICS.reserved_textures_bytes.add(reserved_bytes);
    if committed_bytes > 0 {
        BUDGET_UNFLUSHED_LOCAL_DELTA
            .0
            .fetch_add(committed_bytes as i64, Ordering::Relaxed);
    }
    if is_sparse {
        METRICS.sparse_texture_count.add(1);
    } else {
        METRICS.committed_texture_count.add(1);
    }
}

/// Track destruction of a host texture.
pub fn on_texture_destroyed(committed_bytes: u64, reserved_bytes: u64, is_sparse: bool) {
    METRICS
        .committed_textures_bytes
        .sub_saturating(committed_bytes);
    METRICS
        .reserved_textures_bytes
        .sub_saturating(reserved_bytes);
    if committed_bytes > 0 {
        BUDGET_UNFLUSHED_LOCAL_DELTA
            .0
            .fetch_sub(committed_bytes as i64, Ordering::Relaxed);
    }
    if is_sparse {
        METRICS.sparse_texture_count.sub_saturating(1);
    } else {
        METRICS.committed_texture_count.sub_saturating(1);
    }
}

/// Adjust mapped sparse tile bytes (`delta > 0` on map, `< 0` on unmap/free).
pub fn on_sparse_tiles_delta(delta_bytes: i64) {
    if delta_bytes >= 0 {
        METRICS.mapped_tile_bytes.add(delta_bytes as u64);
    } else {
        METRICS
            .mapped_tile_bytes
            .sub_saturating(delta_bytes.unsigned_abs());
    }
}

/// Adjust sparse tile pool backing heap bytes in VRAM.
pub fn on_tile_pool_delta(delta_bytes: i64) {
    if delta_bytes >= 0 {
        METRICS.tile_pool_allocated_bytes.add(delta_bytes as u64);
    } else {
        METRICS
            .tile_pool_allocated_bytes
            .sub_saturating(delta_bytes.unsigned_abs());
    }
    if delta_bytes != 0 {
        BUDGET_UNFLUSHED_LOCAL_DELTA
            .0
            .fetch_add(delta_bytes, Ordering::Relaxed);
    }
}

/// Adjust CPU RAM and GPU VRAM attributed to shadow copies / card mirrors.
pub fn on_shadow_delta(ram_delta: i64, vram_delta: i64) {
    if ram_delta >= 0 {
        METRICS.shadow_ram_bytes.add(ram_delta as u64);
    } else {
        METRICS
            .shadow_ram_bytes
            .sub_saturating(ram_delta.unsigned_abs());
    }
    if vram_delta >= 0 {
        METRICS.shadow_vram_bytes.add(vram_delta as u64);
    } else {
        METRICS
            .shadow_vram_bytes
            .sub_saturating(vram_delta.unsigned_abs());
    }
    if vram_delta != 0 {
        BUDGET_UNFLUSHED_LOCAL_DELTA
            .0
            .fetch_add(vram_delta, Ordering::Relaxed);
    }
}

/// Update pooled upload ring footprint in host-visible staging memory.
pub fn on_upload_ring_delta(delta_bytes: i64) {
    if delta_bytes >= 0 {
        METRICS.upload_ring_bytes.add(delta_bytes as u64);
    } else {
        METRICS
            .upload_ring_bytes
            .sub_saturating(delta_bytes.unsigned_abs());
    }
}

/// Update staging pool footprint in host-visible memory.
pub fn on_staging_delta(delta_bytes: i64) {
    if delta_bytes >= 0 {
        METRICS.staging_bytes.add(delta_bytes as u64);
    } else {
        METRICS
            .staging_bytes
            .sub_saturating(delta_bytes.unsigned_abs());
    }
}

/// Adjust count of pending host resources queued for retirement behind fences.
pub fn on_retired_pending_delta(delta: i64) {
    if delta >= 0 {
        METRICS.retired_pending_count.add(delta as u64);
    } else {
        METRICS
            .retired_pending_count
            .sub_saturating(delta.unsigned_abs());
    }
}

/// Adjust bytes marked reclaimable under VRAM pressure.
pub fn on_reclaimable_delta(delta_bytes: i64) {
    if delta_bytes >= 0 {
        METRICS.reclaimable_bytes.add(delta_bytes as u64);
    } else {
        METRICS
            .reclaimable_bytes
            .sub_saturating(delta_bytes.unsigned_abs());
    }
}

#[inline]
fn record_nonlocal_video_memory(values: [u64; 4]) {
    METRICS
        .dxgi_nonlocal_budget_bytes
        .0
        .store(values[2], Ordering::Relaxed);
    METRICS
        .dxgi_nonlocal_used_bytes
        .0
        .store(values[3], Ordering::Relaxed);
}

/// Return cached `[local_budget, local_usage, nonlocal_budget, nonlocal_usage]`
/// when within TTL and `|unflushed_vram_delta| < 16 MiB`, adjusting `local_usage`
/// by the incremental delta; otherwise run `query_fn` and cache its result.
pub fn cached_video_memory_with<F>(ttl: Duration, query_fn: F) -> Option<[u64; 4]>
where
    F: FnOnce() -> Option<[u64; 4]>,
{
    let delta = BUDGET_UNFLUSHED_LOCAL_DELTA.0.load(Ordering::Relaxed);
    if let Ok(guard) = BUDGET_CACHE.lock()
        && let Some(cached) = *guard
        && cached.captured_at.elapsed() < ttl
        && delta.abs() < MAX_UNFLUSHED_BUDGET_DELTA
    {
        METRICS.dxgi_queries_cached.add(1);
        let mut vals = cached.values;
        vals[1] = vals[1].saturating_add_signed(delta);
        return Some(vals);
    }

    BUDGET_UNFLUSHED_LOCAL_DELTA.0.store(0, Ordering::Release);
    let values = query_fn()?;
    METRICS.dxgi_queries_actual.add(1);
    record_nonlocal_video_memory(values);
    if let Ok(mut guard) = BUDGET_CACHE.lock() {
        *guard = Some(CachedBudget {
            captured_at: Instant::now(),
            values,
        });
    }
    Some(values)
}

/// Query DXGI adapter video memory through [`cached_video_memory_with`] (15 ms TTL
/// + 16 MiB burst invalidation) when incremental accounting is enabled, or query directly otherwise.
pub(crate) fn video_memory_budget(host: &crate::host::Host) -> Option<[u64; 4]> {
    if enabled() {
        cached_video_memory_with(DEFAULT_BUDGET_TTL, || host.video_memory())
    } else {
        let values = host.video_memory()?;
        record_nonlocal_video_memory(values);
        Some(values)
    }
}

/// Force-invalidate the DXGI budget cache (e.g. immediately after large evictions).
pub fn invalidate_budget_cache() {
    BUDGET_UNFLUSHED_LOCAL_DELTA.0.store(0, Ordering::Release);
    if let Ok(mut guard) = BUDGET_CACHE.lock() {
        *guard = None;
    }
}

/// Compare incremental live texture totals against an independent full-walk enumeration.
pub fn record_audit(
    enumerated_committed_bytes: u64,
    enumerated_mapped_bytes: u64,
    tolerance: u64,
) -> u64 {
    METRICS.audit_runs.add(1);
    let expected = enumerated_committed_bytes.saturating_add(enumerated_mapped_bytes);
    let tracked = METRICS
        .committed_textures_bytes
        .load()
        .saturating_add(METRICS.mapped_tile_bytes.load());
    let drift = tracked.abs_diff(expected);
    METRICS
        .last_audit_drift_bytes
        .0
        .store(drift, Ordering::Relaxed);
    if drift > tolerance {
        METRICS.audit_mismatches.add(1);
        tracing::warn!(
            tracked,
            expected,
            drift,
            "REA-002 accounting audit detected drift"
        );
    }
    drift
}

/// O(1) read of all incremental accounting counters.
pub fn snapshot() -> AccountingSnapshot {
    AccountingSnapshot {
        enabled: enabled(),
        audit_enabled: audit_enabled(),
        committed_textures_bytes: METRICS.committed_textures_bytes.load(),
        reserved_textures_bytes: METRICS.reserved_textures_bytes.load(),
        mapped_tile_bytes: METRICS.mapped_tile_bytes.load(),
        tile_pool_allocated_bytes: METRICS.tile_pool_allocated_bytes.load(),
        staging_bytes: METRICS.staging_bytes.load(),
        shadow_ram_bytes: METRICS.shadow_ram_bytes.load(),
        shadow_vram_bytes: METRICS.shadow_vram_bytes.load(),
        upload_ring_bytes: METRICS.upload_ring_bytes.load(),
        reclaimable_bytes: METRICS.reclaimable_bytes.load(),
        retired_pending_count: METRICS.retired_pending_count.load(),
        committed_texture_count: METRICS.committed_texture_count.load(),
        sparse_texture_count: METRICS.sparse_texture_count.load(),
        dxgi_queries_actual: METRICS.dxgi_queries_actual.load(),
        dxgi_queries_cached: METRICS.dxgi_queries_cached.load(),
        dxgi_nonlocal_used_bytes: METRICS.dxgi_nonlocal_used_bytes.load(),
        dxgi_nonlocal_budget_bytes: METRICS.dxgi_nonlocal_budget_bytes.load(),
        audit_runs: METRICS.audit_runs.load(),
        audit_mismatches: METRICS.audit_mismatches.load(),
        last_audit_drift_bytes: METRICS.last_audit_drift_bytes.load(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incremental_deltas_saturate_and_audit_matches() {
        let base = snapshot();

        on_texture_created(4096, 65536, true);
        on_sparse_tiles_delta(65536);
        on_tile_pool_delta(131072);
        on_shadow_delta(1024, 2048);

        let mid = snapshot();
        assert_eq!(
            mid.committed_textures_bytes,
            base.committed_textures_bytes + 4096
        );
        assert_eq!(
            mid.reserved_textures_bytes,
            base.reserved_textures_bytes + 65536
        );
        assert_eq!(mid.mapped_tile_bytes, base.mapped_tile_bytes + 65536);
        assert_eq!(
            mid.tile_pool_allocated_bytes,
            base.tile_pool_allocated_bytes + 131072
        );
        assert_eq!(mid.shadow_ram_bytes, base.shadow_ram_bytes + 1024);
        assert_eq!(mid.shadow_vram_bytes, base.shadow_vram_bytes + 2048);

        let drift = record_audit(mid.committed_textures_bytes, mid.mapped_tile_bytes, 0);
        assert_eq!(drift, 0);

        on_shadow_delta(-1024, -2048);
        on_tile_pool_delta(-131072);
        on_sparse_tiles_delta(-65536);
        on_texture_destroyed(4096, 65536, true);

        let end = snapshot();
        assert_eq!(end.committed_textures_bytes, base.committed_textures_bytes);
        assert_eq!(end.reserved_textures_bytes, base.reserved_textures_bytes);
        assert_eq!(end.mapped_tile_bytes, base.mapped_tile_bytes);
        assert_eq!(end.shadow_ram_bytes, base.shadow_ram_bytes);
        assert_eq!(end.shadow_vram_bytes, base.shadow_vram_bytes);
    }

    #[test]
    fn budget_cache_deduplicates_queries_within_ttl() {
        invalidate_budget_cache();
        let before = snapshot();
        let mut calls = 0usize;
        let first = cached_video_memory_with(Duration::from_secs(10), || {
            calls += 1;
            Some([100, 200, 300, 400])
        });
        let second = cached_video_memory_with(Duration::from_secs(10), || {
            calls += 1;
            Some([999, 999, 999, 999])
        });
        assert_eq!(calls, 1);
        assert_eq!(first, Some([100, 200, 300, 400]));
        assert_eq!(second, Some([100, 200, 300, 400]));
        let after = snapshot();
        assert_eq!(after.dxgi_queries_actual, before.dxgi_queries_actual + 1);
        assert_eq!(after.dxgi_queries_cached, before.dxgi_queries_cached + 1);
        invalidate_budget_cache();
    }

    #[test]
    fn budget_cache_tracks_unflushed_vram_delta_and_invalidates_on_burst() {
        invalidate_budget_cache();
        let mut calls = 0usize;
        let first = cached_video_memory_with(Duration::from_secs(10), || {
            calls += 1;
            Some([7000 << 20, 4000 << 20, 8000 << 20, 3000 << 20])
        });
        assert_eq!(calls, 1);
        assert_eq!(
            first,
            Some([7000 << 20, 4000 << 20, 8000 << 20, 3000 << 20])
        );

        on_texture_created(4 << 20, 0, false);
        let adjusted = cached_video_memory_with(Duration::from_secs(10), || {
            calls += 1;
            Some([7000 << 20, 4004 << 20, 8000 << 20, 3000 << 20])
        });
        assert_eq!(calls, 1, "small delta (<16 MiB) must serve adjusted cache");
        assert_eq!(
            adjusted,
            Some([7000 << 20, 4004 << 20, 8000 << 20, 3000 << 20])
        );

        on_texture_created(20 << 20, 0, false);
        let refreshed = cached_video_memory_with(Duration::from_secs(10), || {
            calls += 1;
            Some([7000 << 20, 4024 << 20, 8000 << 20, 3000 << 20])
        });
        assert_eq!(
            calls, 2,
            "burst >=16 MiB must bypass TTL and force fresh DXGI query"
        );
        assert_eq!(
            refreshed,
            Some([7000 << 20, 4024 << 20, 8000 << 20, 3000 << 20])
        );

        on_texture_destroyed(24 << 20, 0, false);
        invalidate_budget_cache();
    }
}
