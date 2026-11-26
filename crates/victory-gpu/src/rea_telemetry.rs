//! Expanded REA-002 runtime telemetry: frame-pacing percentiles, stall
//! buckets, memory/residency counters, coherence/readback stats, native CPU
//! fast-path stats, and optional out-of-repo NDJSON logging.

use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

const PACER_WINDOW: usize = 300;
const JSON_ENV: &str = "VICTORY_REA_TELEMETRY_JSON";

/// Host CPU fast-path statistics mirrored from `victory_loader::fastpath`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CpuTelemetry {
    pub fastpath_enabled: bool,
    pub patched_clzero_loops: u64,
    pub patched_clzero_alt_loops: u64,
    pub patched_monitorx: u64,
    pub patched_mwaitx: u64,
    pub fast_clzero_calls: u64,
    pub fast_clzero_bytes: u64,
    pub fast_monitorx_calls: u64,
    pub fast_mwaitx_calls: u64,
    pub emu_total: u64,
    pub emu_zeroed_bytes: u64,
}

/// DirectStorage streaming statistics mirrored from `victory_hle::modules::dstorage`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StreamingTelemetry {
    pub enabled: bool,
    pub scratch_reuses: u64,
    pub scratch_bytes_saved: u64,
    pub low_ram_throttles: u64,
    pub page_dest_notifies: u64,
}

static CPU_PROVIDER: OnceLock<fn() -> CpuTelemetry> = OnceLock::new();
static STREAMING_PROVIDER: OnceLock<fn() -> StreamingTelemetry> = OnceLock::new();

/// Registers the CPU fast-path telemetry provider (typically wired by `victory-hle`).
pub fn register_cpu_stats_provider(provider: fn() -> CpuTelemetry) {
    let _ = CPU_PROVIDER.set(provider);
}

/// Registers the DirectStorage streaming telemetry provider (typically wired by `victory-hle`).
pub fn register_streaming_stats_provider(provider: fn() -> StreamingTelemetry) {
    let _ = STREAMING_PROVIDER.set(provider);
}

/// Frame-pacing percentiles and cumulative frame-time stall counters.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PacingSnapshot {
    pub samples: usize,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub low1_fps: f64,
    pub stalls_100ms: u64,
    pub stalls_250ms: u64,
    pub stalls_500ms: u64,
    pub stalls_1000ms: u64,
}

#[derive(Debug)]
struct FramePacer {
    samples: VecDeque<f64>,
    stalls_100ms: u64,
    stalls_250ms: u64,
    stalls_500ms: u64,
    stalls_1000ms: u64,
}

impl FramePacer {
    const fn new() -> Self {
        Self {
            samples: VecDeque::new(),
            stalls_100ms: 0,
            stalls_250ms: 0,
            stalls_500ms: 0,
            stalls_1000ms: 0,
        }
    }

    fn record(&mut self, dt_ms: f64) {
        if !dt_ms.is_finite() || dt_ms <= 0.0 {
            return;
        }
        if self.samples.len() >= PACER_WINDOW {
            self.samples.pop_front();
        }
        self.samples.push_back(dt_ms);
        if dt_ms > 100.0 {
            self.stalls_100ms += 1;
        }
        if dt_ms > 250.0 {
            self.stalls_250ms += 1;
        }
        if dt_ms > 500.0 {
            self.stalls_500ms += 1;
        }
        if dt_ms > 1000.0 {
            self.stalls_1000ms += 1;
        }
    }

    fn snapshot(&self) -> PacingSnapshot {
        if self.samples.is_empty() {
            return PacingSnapshot {
                samples: 0,
                p50_ms: 0.0,
                p95_ms: 0.0,
                p99_ms: 0.0,
                low1_fps: 0.0,
                stalls_100ms: self.stalls_100ms,
                stalls_250ms: self.stalls_250ms,
                stalls_500ms: self.stalls_500ms,
                stalls_1000ms: self.stalls_1000ms,
            };
        }
        let mut sorted: Vec<f64> = self.samples.iter().copied().collect();
        sorted.sort_by(f64::total_cmp);
        let n = sorted.len();
        let pct = |q: f64| -> f64 {
            let idx = ((n - 1) as f64 * q).round() as usize;
            sorted[idx.min(n - 1)]
        };
        let tail_count = ((n as f64) * 0.01).ceil().max(1.0) as usize;
        let tail_start = n.saturating_sub(tail_count);
        let tail_mean_ms =
            sorted[tail_start..].iter().sum::<f64>() / (n - tail_start).max(1) as f64;
        let low1_fps = if tail_mean_ms > 0.0 {
            1000.0 / tail_mean_ms
        } else {
            0.0
        };
        PacingSnapshot {
            samples: n,
            p50_ms: pct(0.50),
            p95_ms: pct(0.95),
            p99_ms: pct(0.99),
            low1_fps,
            stalls_100ms: self.stalls_100ms,
            stalls_250ms: self.stalls_250ms,
            stalls_500ms: self.stalls_500ms,
            stalls_1000ms: self.stalls_1000ms,
        }
    }
}

static PACER: Mutex<FramePacer> = Mutex::new(FramePacer::new());
static WINDOWS_LOGGED: AtomicU64 = AtomicU64::new(0);

/// Records a single frame-to-frame interval in milliseconds.
pub fn record_frame(dt_ms: f64) {
    PACER
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .record(dt_ms);
}

/// Returns the current rolling frame-pacing snapshot.
pub fn pacing_snapshot() -> PacingSnapshot {
    PACER.lock().unwrap_or_else(|e| e.into_inner()).snapshot()
}

enum JsonTarget {
    Disabled,
    Rejected,
    Path(PathBuf),
}

fn json_target() -> &'static JsonTarget {
    static TARGET: OnceLock<JsonTarget> = OnceLock::new();
    TARGET.get_or_init(|| {
        let Some(raw) = std::env::var_os(JSON_ENV) else {
            return JsonTarget::Disabled;
        };
        if raw.is_empty() {
            return JsonTarget::Disabled;
        }
        let path = PathBuf::from(raw);
        match victory_core::ensure_outside_repo(&path) {
            Ok(()) => JsonTarget::Path(path),
            Err(err) => {
                tracing::error!(
                    path = %path.display(),
                    error = %err,
                    "{JSON_ENV} points inside the repository and was rejected by policy"
                );
                JsonTarget::Rejected
            }
        }
    })
}

pub(crate) fn append_json_line_to(path: &Path, line: &str) -> Result<(), String> {
    victory_core::ensure_outside_repo(path).map_err(|e| e.to_string())?;
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|e| format!("create_dir_all: {e}"))?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("open {}: {e}", path.display()))?;
    writeln!(file, "{line}").map_err(|e| format!("write {}: {e}", path.display()))
}

/// Emits the expanded 5-second REA-002 telemetry summary and optional NDJSON line.
pub(crate) fn log_window(
    fps: f64,
    avg_ms: f64,
    cpu_prep_ms: f64,
    gpu_ms: f64,
    back_ms: f64,
    back_mb: f64,
    draws: f64,
) {
    let seq = WINDOWS_LOGGED.fetch_add(1, Ordering::Relaxed) + 1;
    let pacing = pacing_snapshot();
    let sched = victory_core::scheduler::stats();
    let acc = crate::accounting::snapshot();
    let [vram_budget, vram_usage, _, _] = crate::host::host()
        .and_then(crate::accounting::video_memory_budget)
        .unwrap_or([0; 4]);
    let sparse = crate::tiles::stats();
    let rb = crate::readback::stats();
    let dirty = crate::shadow::dirty_stats();
    let (tls_hits, tls_misses) = crate::shadow::tls_cache_stats();
    let (dyn_allocs, dyn_wraps, dyn_refusals) = crate::descriptor::host::dynamic_slot_stats();
    let barriers_elided = crate::host::barrier_fastpath_stats();
    let auth = crate::authority::stats();
    let upload = crate::upload_ring::stats();
    let dxr = crate::dxr_proto::stats();
    let cpu = CPU_PROVIDER.get().map(|f| f()).unwrap_or_default();
    let stream = STREAMING_PROVIDER.get().map(|f| f()).unwrap_or_default();

    let vram_used_mb = vram_usage >> 20;
    let vram_budget_mb = vram_budget >> 20;
    let acc_comm_mb = acc.committed_textures_bytes >> 20;
    let acc_res_mb = acc.reserved_textures_bytes >> 20;
    let shadow_ram_mb = acc.shadow_ram_bytes >> 20;
    let shadow_vram_mb = acc.shadow_vram_bytes >> 20;
    let reclaim_mb = acc.reclaimable_bytes >> 20;
    let dirty_mb = dirty.diff_bytes as f64 / (1024.0 * 1024.0);
    let clz_mb = cpu.fast_clzero_bytes as f64 / (1024.0 * 1024.0);
    let saved_mb = stream.scratch_bytes_saved as f64 / (1024.0 * 1024.0);
    let ring_mb = upload.ring_bytes_suballoc as f64 / (1024.0 * 1024.0);
    let cq_mb = upload.copy_queue_bytes as f64 / (1024.0 * 1024.0);

    tracing::info!(
        "REA-002 telemetry: seq={seq} fps={fps:.1} avg_ms={avg_ms:.2} p50_ms={:.2} p95_ms={:.2} p99_ms={:.2} low1_fps={:.1} \
         stalls_100={} stalls_250={} stalls_500={} stalls_1000={} cpu_prep_ms={cpu_prep_ms:.2} gpu_ms={gpu_ms:.2} \
         back_ms={back_ms:.2} back_mb={back_mb:.2} draws={draws:.0} \
         vram_used_mb={vram_used_mb} vram_budget_mb={vram_budget_mb} acc_comm_mb={acc_comm_mb} acc_res_mb={acc_res_mb} \
         sparse_mapped_mb={} sparse_virt_mb={} sparse_pool_mb={} sparse_reclaimed_tiles={} reclaim_mb={reclaim_mb} \
         shadow_ram_mb={shadow_ram_mb} shadow_vram_mb={shadow_vram_mb} \
         dxgi_act={} dxgi_hit={} audit_runs={} audit_err={} \
         rb_tagged={} rb_hit={} rb_miss={} rb_fb={} rb_val={} \
         dirty_calls={} dirty_mb={dirty_mb:.1} dirty_avx2={} dirty_subsumed={} dirty_coal={}/{} \
         auth_gpu={} auth_cpu={} auth_coh={} auth_skip_cmp={} auth_prom={} \
         tls_hit={tls_hits} tls_miss={tls_misses} dyn_alloc={dyn_allocs} dyn_wrap={dyn_wraps} dyn_ref={dyn_refusals} bar_elided={barriers_elided} \
         sched_on={} cores={}/{} workers={}/{}/{}/{} \
         cpu_fast={} clz_patch={}/{} mon_patch={}/{} clz_fast={} clz_mb={clz_mb:.1} wait_fast={}/{} emu_total={} \
         stream_on={} scratch_reuse={} saved_mb={saved_mb:.1} low_ram_throttle={} \
         ring_on={} pool_hit={} pool_miss={} ring_mb={ring_mb:.1} cq_on={} cq_batches={} cq_mb={cq_mb:.1} \
         dxr_tier={} dxr_blas={} dxr_tlas={} ffx_0x218={}",
        pacing.p50_ms,
        pacing.p95_ms,
        pacing.p99_ms,
        pacing.low1_fps,
        pacing.stalls_100ms,
        pacing.stalls_250ms,
        pacing.stalls_500ms,
        pacing.stalls_1000ms,
        sparse.physical_mapped_mb,
        sparse.virtual_reserved_mb,
        sparse.pool_capacity_mb,
        sparse.tiles_reclaimed_total,
        acc.dxgi_queries_actual,
        acc.dxgi_queries_cached,
        acc.audit_runs,
        acc.audit_mismatches,
        rb.tagged_buffers,
        rb.scan_hits,
        rb.scan_misses,
        rb.semantic_fallbacks,
        rb.validate_samples,
        dirty.diff_calls,
        dirty.avx2_calls,
        dirty.subsumed_skips,
        dirty.copy_regions_coalesced,
        dirty.copy_regions_raw,
        auth.gpu_authoritative,
        auth.cpu_authoritative,
        auth.shared_coherent,
        auth.skipped_card_compares,
        auth.promoted_gpu_auth,
        sched.enabled,
        sched.physical_cores,
        sched.logical_processors,
        sched.streaming_workers,
        sched.shader_workers,
        sched.shadow_workers,
        sched.memcopy_workers,
        cpu.fastpath_enabled,
        cpu.patched_clzero_loops,
        cpu.patched_clzero_alt_loops,
        cpu.patched_monitorx,
        cpu.patched_mwaitx,
        cpu.fast_clzero_calls,
        cpu.fast_monitorx_calls,
        cpu.fast_mwaitx_calls,
        cpu.emu_total,
        stream.enabled,
        stream.scratch_reuses,
        stream.low_ram_throttles,
        upload.enabled,
        upload.pool_hits,
        upload.pool_misses,
        upload.copy_queue_enabled,
        upload.copy_queue_batches,
        dxr.tier,
        dxr.blas_builds,
        dxr.tlas_builds,
        dxr.ffx_0x218_calls,
    );

    if let JsonTarget::Path(path) = json_target() {
        let json = format!(
            "{{\"seq\":{seq},\"fps\":{fps:.3},\"avg_ms\":{avg_ms:.3},\"p50_ms\":{:.3},\"p95_ms\":{:.3},\"p99_ms\":{:.3},\"low1_fps\":{:.3},\
             \"stalls_100ms\":{},\"stalls_250ms\":{},\"stalls_500ms\":{},\"stalls_1000ms\":{},\
             \"cpu_prep_ms\":{cpu_prep_ms:.3},\"gpu_ms\":{gpu_ms:.3},\"back_ms\":{back_ms:.3},\"back_mb\":{back_mb:.3},\"draws\":{draws:.2},\
             \"vram_used_mb\":{vram_used_mb},\"vram_budget_mb\":{vram_budget_mb},\"acc_comm_mb\":{acc_comm_mb},\"acc_res_mb\":{acc_res_mb},\
             \"sparse_mapped_mb\":{},\"sparse_virt_mb\":{},\"sparse_reclaimed_tiles\":{},\"shadow_ram_mb\":{shadow_ram_mb},\"shadow_vram_mb\":{shadow_vram_mb},\
             \"dxgi_act\":{},\"dxgi_hit\":{},\"audit_runs\":{},\"audit_mismatches\":{},\
             \"rb_tagged\":{},\"rb_hit\":{},\"rb_miss\":{},\"rb_semantic_fallbacks\":{},\"rb_validate_warnings\":{},\
             \"dirty_calls\":{},\"dirty_bytes\":{},\"dirty_subsumed\":{},\"dirty_coalesced\":{},\"dirty_raw\":{},\
             \"auth_gpu\":{},\"auth_skipped_cmp\":{},\"tls_hits\":{tls_hits},\"tls_misses\":{tls_misses},\"barriers_elided\":{barriers_elided},\
             \"dyn_allocs\":{dyn_allocs},\"dyn_wraps\":{dyn_wraps},\"dyn_refusals\":{dyn_refusals},\
             \"clzero_fast_calls\":{},\"clzero_fast_bytes\":{},\"wait_fast_calls\":{},\"emu_total\":{},\
             \"scratch_reuses\":{},\"scratch_bytes_saved\":{},\"copy_queue_batches\":{},\"copy_queue_bytes\":{},\
             \"dxr_tier\":{},\"ffx_0x218_calls\":{}}}",
            pacing.p50_ms,
            pacing.p95_ms,
            pacing.p99_ms,
            pacing.low1_fps,
            pacing.stalls_100ms,
            pacing.stalls_250ms,
            pacing.stalls_500ms,
            pacing.stalls_1000ms,
            sparse.physical_mapped_mb,
            sparse.virtual_reserved_mb,
            sparse.tiles_reclaimed_total,
            acc.dxgi_queries_actual,
            acc.dxgi_queries_cached,
            acc.audit_runs,
            acc.audit_mismatches,
            rb.tagged_buffers,
            rb.scan_hits,
            rb.scan_misses,
            rb.semantic_fallbacks,
            rb.validate_warnings,
            dirty.diff_calls,
            dirty.diff_bytes,
            dirty.subsumed_skips,
            dirty.copy_regions_coalesced,
            dirty.copy_regions_raw,
            auth.gpu_authoritative,
            auth.skipped_card_compares,
            cpu.fast_clzero_calls,
            cpu.fast_clzero_bytes,
            cpu.fast_monitorx_calls + cpu.fast_mwaitx_calls,
            cpu.emu_total,
            stream.scratch_reuses,
            stream.scratch_bytes_saved,
            upload.copy_queue_batches,
            upload.copy_queue_bytes,
            dxr.tier,
            dxr.ffx_0x218_calls,
        );
        if let Err(err) = append_json_line_to(path, &json) {
            tracing::warn!(error = %err, "failed to append REA-002 telemetry JSON line");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{FramePacer, append_json_line_to};

    #[test]
    fn percentiles_and_stall_buckets_compute_accurately() {
        let mut pacer = FramePacer::new();
        for i in 1..=100 {
            let ms = match i {
                99 => 150.0,
                100 => 1200.0,
                v => f64::from(v),
            };
            pacer.record(ms);
        }
        let snap = pacer.snapshot();
        assert_eq!(snap.samples, 100);
        assert!((snap.p50_ms - 50.0).abs() < 2.0);
        assert!((snap.p95_ms - 95.0).abs() < 2.0);
        assert!((snap.p99_ms - 150.0).abs() < 5.0);
        assert_eq!(snap.stalls_100ms, 2);
        assert_eq!(snap.stalls_250ms, 1);
        assert_eq!(snap.stalls_500ms, 1);
        assert_eq!(snap.stalls_1000ms, 1);
        assert!(snap.low1_fps > 0.0 && snap.low1_fps < 1.5);
    }

    #[test]
    fn telemetry_json_rejects_in_repo_path_and_writes_temp_path() {
        let in_repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("illegal-telemetry.jsonl");
        let err = append_json_line_to(&in_repo, "{}").unwrap_err();
        assert!(err.contains("inside") || err.contains("repository") || err.contains("repo"));

        let temp_file = std::env::temp_dir().join(format!(
            "victory-rea002-telemetry-test-{}.jsonl",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&temp_file);
        append_json_line_to(&temp_file, "{\"ok\":true}").expect("temp write succeeds");
        let contents = std::fs::read_to_string(&temp_file).unwrap();
        assert!(contents.contains("{\"ok\":true}"));
        let _ = std::fs::remove_file(&temp_file);
    }
}
