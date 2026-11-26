//! Expanded REA-002 / REA-003.1 runtime telemetry: frame-pacing percentiles,
//! stall buckets, memory/residency counters, coherence/readback stats, native
//! CPU fast-path stats, CPU prep phase timers, unique-session NDJSON logging,
//! resolved configuration reporting, and honest backend lifecycle status.

use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, Once, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

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
    pub wait_yields: u64,
    pub wait_sleeps: u64,
    pub wait_early_wakes: u64,
    pub wait_total_us: u64,
    pub wait_sleep_us: u64,
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

/// FidelityFX HLE statistics mirrored from `victory_hle::modules::ffx`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FfxHleTelemetry {
    pub mode: &'static str,
    pub create_calls: u64,
    pub dispatch_calls: u64,
}

/// Honest lifecycle state of an experimental or runtime subsystem.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendState {
    Disabled,
    Requested,
    Unsupported,
    LibraryMissing,
    Initialized,
    ContextCreated,
    Active,
    Fallback,
    Failed,
}

impl BackendState {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Requested => "requested",
            Self::Unsupported => "unsupported",
            Self::LibraryMissing => "library_missing",
            Self::Initialized => "initialized",
            Self::ContextCreated => "context_created",
            Self::Active => "active",
            Self::Fallback => "fallback",
            Self::Failed => "failed",
        }
    }
}

/// Lifecycle & activity metrics reported for each subsystem in telemetry JSONL.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackendStatus {
    pub state: BackendState,
    pub requested: bool,
    pub enabled: bool,
    pub initialized: bool,
    pub active: bool,
    pub executed_calls: u64,
    pub fallback_count: u64,
    pub failure_count: u64,
}

impl BackendStatus {
    fn json_object(self) -> String {
        format!(
            "{{\"state\":\"{}\",\"requested\":{},\"enabled\":{},\"initialized\":{},\
             \"active\":{},\"executed_calls\":{},\"fallback_count\":{},\"failure_count\":{}}}",
            self.state.as_str(),
            self.requested,
            self.enabled,
            self.initialized,
            self.active,
            self.executed_calls,
            self.fallback_count,
            self.failure_count,
        )
    }
}

/// CPU preparation sub-phase buckets accumulated in microseconds and reported per frame.
#[repr(usize)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PrepPhase {
    AskOnce = 0,
    Copy = 1,
    Fill = 2,
    Materialize = 3,
    Authority = 4,
    NoteGpuWrite = 5,
    ShadowUpdate = 6,
    SubpageSplit = 7,
    TileLock = 8,
}

const PREP_PHASE_COUNT: usize = 9;
static PREP_PHASE_US: [AtomicU64; PREP_PHASE_COUNT] =
    [const { AtomicU64::new(0) }; PREP_PHASE_COUNT];

/// Adds elapsed microseconds to a CPU preparation sub-phase accumulator.
#[inline]
pub(crate) fn record_prep_phase_us(phase: PrepPhase, us: u64) {
    if us > 0 {
        PREP_PHASE_US[phase as usize].fetch_add(us, Ordering::Relaxed);
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct PrepBreakdown {
    ask_once_ms: f64,
    copy_ms: f64,
    fill_ms: f64,
    materialize_ms: f64,
    authority_ms: f64,
    note_gpu_write_ms: f64,
    shadow_update_ms: f64,
    subpage_split_ms: f64,
    tile_lock_ms: f64,
}

fn take_prep_breakdown(window_frames: f64) -> PrepBreakdown {
    let divisor = window_frames.max(1.0) * 1000.0;
    let ms = |p: PrepPhase| PREP_PHASE_US[p as usize].swap(0, Ordering::Relaxed) as f64 / divisor;
    PrepBreakdown {
        ask_once_ms: ms(PrepPhase::AskOnce),
        copy_ms: ms(PrepPhase::Copy),
        fill_ms: ms(PrepPhase::Fill),
        materialize_ms: ms(PrepPhase::Materialize),
        authority_ms: ms(PrepPhase::Authority),
        note_gpu_write_ms: ms(PrepPhase::NoteGpuWrite),
        shadow_update_ms: ms(PrepPhase::ShadowUpdate),
        subpage_split_ms: ms(PrepPhase::SubpageSplit),
        tile_lock_ms: ms(PrepPhase::TileLock),
    }
}

static CPU_PROVIDER: OnceLock<fn() -> CpuTelemetry> = OnceLock::new();
static STREAMING_PROVIDER: OnceLock<fn() -> StreamingTelemetry> = OnceLock::new();
static FFX_PROVIDER: OnceLock<fn() -> FfxHleTelemetry> = OnceLock::new();

/// Registers the CPU fast-path telemetry provider (typically wired by `victory-hle`).
pub fn register_cpu_stats_provider(provider: fn() -> CpuTelemetry) {
    let _ = CPU_PROVIDER.set(provider);
}

/// Registers the DirectStorage streaming telemetry provider (typically wired by `victory-hle`).
pub fn register_streaming_stats_provider(provider: fn() -> StreamingTelemetry) {
    let _ = STREAMING_PROVIDER.set(provider);
}

/// Registers the FidelityFX HLE telemetry provider (typically wired by `victory-hle`).
pub fn register_ffx_stats_provider(provider: fn() -> FfxHleTelemetry) {
    let _ = FFX_PROVIDER.set(provider);
}

struct SessionIdentity {
    run_id: String,
    pid: u32,
    started_instant: Instant,
    preset: String,
    commit: String,
    runtime_hash: String,
}

fn utc_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

fn non_empty_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
}

fn detect_runtime_hash() -> String {
    if let Some(sha) = non_empty_env("VICTORY_RUNTIME_SHA256") {
        return if sha.contains(':') {
            sha
        } else {
            format!("sha256:{sha}")
        };
    }
    let Ok(exe) = std::env::current_exe() else {
        return "unknown".to_owned();
    };
    match std::fs::read(exe) {
        Ok(bytes) => format!("blake3:{}", blake3::hash(&bytes).to_hex()),
        Err(_) => "unknown".to_owned(),
    }
}

fn session_identity() -> &'static SessionIdentity {
    static ID: OnceLock<SessionIdentity> = OnceLock::new();
    ID.get_or_init(|| {
        let pid = std::process::id();
        let run_id = non_empty_env("VICTORY_RUN_ID")
            .unwrap_or_else(|| format!("{}-pid{pid}", utc_unix_ms()));
        let preset = non_empty_env("VICTORY_PRESET").unwrap_or_else(|| "unspecified".to_owned());
        let commit = non_empty_env("VICTORY_GIT_COMMIT")
            .or_else(|| option_env!("VICTORY_GIT_COMMIT").map(ToOwned::to_owned))
            .unwrap_or_else(|| "unknown".to_owned());
        let runtime_hash = detect_runtime_hash();
        SessionIdentity {
            run_id,
            pid,
            started_instant: Instant::now(),
            preset,
            commit,
            runtime_hash,
        }
    })
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                use std::fmt::Write as _;
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

/// Logs the resolved effective configuration (env vars vs runtime defaults and
/// implicit dependencies) once at host D3D12 initialization.
pub(crate) fn log_effective_config() {
    static LOGGED: Once = Once::new();
    LOGGED.call_once(|| {
        let id = session_identity();
        let own_upscaler = non_empty_env("VICTORY_OWN_UPSCALER").is_some_and(|v| v == "1");
        let render_scale = non_empty_env("VICTORY_RENDER_SCALE");
        let upscaler = crate::upscaler::UpscalerMode::from_env();
        let relay = crate::host::relay_level();
        let shader_recovery = victory_shader::emit::rea003_shader_recovery();
        let shader_opt_mask = victory_shader::emit::shader_opt();
        let rea003_shader_opt = victory_shader::emit::rea003_shader_opt();
        let ambient_native = crate::shader_cache::rea003_ambient_native();
        let shadowmask_native = crate::shader_cache::rea003_shadowmask_native();
        let sparse = crate::tiles::enabled();
        let sparse_reclaim = crate::tiles::reclaim_enabled();
        let authority = crate::authority::enabled();
        let rb_validate = crate::readback::validate_enabled();
        let incremental_acc = crate::accounting::enabled();
        let acc_audit = crate::accounting::audit_enabled();
        let gpu_driven = crate::indirect::rea003_gpu_driven();
        let gpu_detile = crate::gpu_detile::gpu_detile_enabled();
        let rg_fastpath = crate::render_graph_opt::rendergraph_fastpath_enabled();
        let dxr = crate::dxr_backend::dxr_enabled();
        let dxr_proto = crate::dxr_proto::dxr_proto_enabled();
        let async_stream = crate::upload_ring::enabled();
        let copy_queue = crate::upload_ring::copy_queue_enabled();

        tracing::info!(
            run_id = %id.run_id,
            pid = id.pid,
            preset = %id.preset,
            commit = %id.commit,
            runtime_hash = %id.runtime_hash,
            own_upscaler,
            render_scale = ?render_scale,
            upscaler = upscaler.as_str(),
            relay,
            dred = crate::host::dred_on(),
            authority,
            rb_validate,
            sparse,
            sparse_reclaim,
            incremental_acc,
            acc_audit,
            async_stream,
            copy_queue,
            shader_recovery,
            rea003_shader_opt,
            shader_opt_mask = format_args!("{shader_opt_mask:#x}"),
            ambient_native,
            shadowmask_native,
            gpu_driven,
            gpu_detile,
            rg_fastpath,
            dxr,
            dxr_proto,
            host_wave = ?non_empty_env("VICTORY_HOST_WAVE"),
            wave_local = non_empty_env("VICTORY_WAVE_LOCAL").is_some(),
            firstlane_own = non_empty_env("VICTORY_FIRSTLANE_OWN").is_some(),
            "REA-003.1 effective config resolved"
        );
    });
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

fn sanitize_run_id(run_id: &str) -> String {
    let cleaned: String = run_id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if cleaned.is_empty() {
        format!("pid{}", std::process::id())
    } else {
        cleaned
    }
}

/// Ensures two processes or relaunches never interleave duplicate `seq` series into
/// an existing non-empty `.jsonl` file: forks to `<stem>-run-<run_id>.<ext>` when needed.
pub(crate) fn resolve_unique_json_path(
    path: &Path,
    run_id: &str,
) -> Result<(PathBuf, bool), String> {
    victory_core::ensure_outside_repo(path).map_err(|e| e.to_string())?;
    let already_non_empty =
        path.exists() && std::fs::metadata(path).is_ok_and(|meta| meta.is_file() && meta.len() > 0);
    if !already_non_empty {
        return Ok((path.to_path_buf(), false));
    }

    let safe_id = sanitize_run_id(run_id);
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("telemetry");
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("jsonl");
    let sibling_name = format!("{stem}-run-{safe_id}.{ext}");
    let unique = match path.parent() {
        Some(parent) => parent.join(sibling_name),
        None => PathBuf::from(sibling_name),
    };
    victory_core::ensure_outside_repo(&unique).map_err(|e| e.to_string())?;
    Ok((unique, true))
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
        let run_id = &session_identity().run_id;
        match resolve_unique_json_path(&path, run_id) {
            Ok((resolved, redirected)) => {
                if redirected {
                    tracing::warn!(
                        requested = %path.display(),
                        resolved = %resolved.display(),
                        run_id = %run_id,
                        "{JSON_ENV} pointed to an existing non-empty file; forked to unique run file"
                    );
                }
                JsonTarget::Path(resolved)
            }
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

#[allow(clippy::too_many_arguments)]
fn collect_backends_json(
    acc: &crate::accounting::AccountingSnapshot,
    auth: &crate::authority::AuthorityStats,
    rb: &crate::readback::ReadbackStats,
    sparse: &crate::tiles::SparseStats,
    cpu: &CpuTelemetry,
    upload: &crate::upload_ring::UploadRingStats,
    dxr: &crate::dxr_proto::DxrStats,
    ffx: &FfxHleTelemetry,
    rg: &crate::render_graph::RenderGraphSnapshot,
) -> String {
    let status_bool = |requested: bool, active: bool, calls: u64, fallbacks: u64, fails: u64| {
        let state = if !requested {
            BackendState::Disabled
        } else if fails > 0 && calls == 0 {
            BackendState::Failed
        } else if fallbacks > 0 && calls == 0 {
            BackendState::Fallback
        } else if active || calls > 0 {
            BackendState::Active
        } else {
            BackendState::Initialized
        };
        BackendStatus {
            state,
            requested,
            enabled: requested,
            initialized: requested,
            active: active || calls > 0,
            executed_calls: calls,
            fallback_count: fallbacks,
            failure_count: fails,
        }
    };

    let auth_req = crate::authority::enabled();
    let gpu_authority = status_bool(
        auth_req,
        auth_req && (auth.gpu_authoritative + auth.skipped_card_compares > 0),
        auth.skipped_card_compares,
        auth.shared_coherent,
        0,
    );

    let rbv_req = rb.validate_enabled;
    let readback_validate = BackendStatus {
        state: if !rbv_req {
            BackendState::Disabled
        } else if !auth_req && !rb.semantic_enabled && !rb.trace_enabled {
            BackendState::Requested
        } else if rb.validate_samples > 0 {
            BackendState::Active
        } else {
            BackendState::Initialized
        },
        requested: rbv_req,
        enabled: rbv_req,
        initialized: rbv_req && (auth_req || rb.semantic_enabled || rb.trace_enabled),
        active: rb.validate_samples > 0,
        executed_calls: rb.validate_samples,
        fallback_count: rb.semantic_fallbacks,
        failure_count: rb.validate_warnings,
    };

    let sp_req = crate::tiles::enabled();
    let sparse_residency = status_bool(
        sp_req,
        sp_req && (sparse.physical_mapped_mb > 0 || sparse.virtual_reserved_mb > 0),
        sparse.virtual_reserved_mb,
        0,
        0,
    );

    let sr_req = crate::tiles::reclaim_enabled();
    let sparse_reclaim = status_bool(
        sr_req,
        sr_req && sparse.tiles_reclaimed_total > 0,
        sparse.tiles_reclaimed_total,
        0,
        0,
    );

    let mw_req = cpu.fastpath_enabled;
    let mw_calls = cpu.fast_clzero_calls + cpu.fast_monitorx_calls + cpu.fast_mwaitx_calls;
    let mwaitx_fastpath = status_bool(mw_req, mw_req && mw_calls > 0, mw_calls, cpu.emu_total, 0);

    let srec_req = victory_shader::emit::rea003_shader_recovery();
    let srec_state = if !srec_req {
        BackendState::Disabled
    } else if rg.submitted_draws + rg.submitted_dispatches > 0 {
        BackendState::Active
    } else {
        BackendState::Initialized
    };
    let shader_recovery = BackendStatus {
        state: srec_state,
        requested: srec_req,
        enabled: srec_req,
        initialized: srec_req,
        active: matches!(srec_state, BackendState::Active),
        executed_calls: rg.submitted_draws + rg.submitted_dispatches,
        fallback_count: 0,
        failure_count: 0,
    };

    let sopt_req = victory_shader::emit::rea003_shader_opt();
    let shader_opt = BackendStatus {
        state: if !sopt_req {
            BackendState::Disabled
        } else if (victory_shader::emit::shader_opt() & 0xC0) != 0 {
            BackendState::Active
        } else {
            BackendState::Unsupported
        },
        requested: sopt_req,
        enabled: sopt_req,
        initialized: sopt_req,
        active: sopt_req && (victory_shader::emit::shader_opt() & 0xC0) != 0,
        executed_calls: 0,
        fallback_count: 0,
        failure_count: 0,
    };

    let amb_req = crate::shader_cache::rea003_ambient_native();
    let ambient_native = BackendStatus {
        state: if !amb_req {
            BackendState::Disabled
        } else if rg.standin_ambient > 0 {
            BackendState::Fallback
        } else if rg.submitted_dispatches > 0 {
            BackendState::Active
        } else {
            BackendState::Initialized
        },
        requested: amb_req,
        enabled: amb_req,
        initialized: amb_req,
        active: amb_req && rg.standin_ambient == 0,
        executed_calls: rg.submitted_dispatches,
        fallback_count: rg.standin_ambient,
        failure_count: 0,
    };

    let sm_req = crate::shader_cache::rea003_shadowmask_native();
    let shadowmask_native = BackendStatus {
        state: if !sm_req {
            BackendState::Disabled
        } else if rg.standin_shadowmask > 0 {
            BackendState::Fallback
        } else if rg.submitted_dispatches > 0 {
            BackendState::Active
        } else {
            BackendState::Initialized
        },
        requested: sm_req,
        enabled: sm_req,
        initialized: sm_req,
        active: sm_req && rg.standin_shadowmask == 0,
        executed_calls: rg.submitted_dispatches,
        fallback_count: rg.standin_shadowmask,
        failure_count: 0,
    };

    let gd_stats = crate::indirect::gpu_driven_stats();
    let gpu_driven = status_bool(
        gd_stats.gpu_driven_enabled,
        gd_stats.gpu_driven_enabled && gd_stats.indirect_dispatches > 0,
        gd_stats.indirect_dispatches,
        0,
        0,
    );

    let detile_on = crate::gpu_detile::gpu_detile_enabled();
    let detile_stats = crate::gpu_detile::stats();
    let gpu_detile = status_bool(
        detile_on,
        detile_on && detile_stats.dispatches > 0,
        detile_stats.dispatches,
        detile_stats.fallback_cpu,
        0,
    );

    let rg_stats = crate::render_graph_opt::stats();
    let rg_calls = rg_stats.transitions_elided + rg_stats.twin_copies_elided;
    let rendergraph_fastpath = status_bool(
        rg_stats.enabled,
        rg_stats.enabled && rg_calls > 0,
        rg_calls,
        0,
        0,
    );

    let up_mode = crate::upscaler::UpscalerMode::from_env();
    let up_req = up_mode != crate::upscaler::UpscalerMode::Original;
    let upscaler = BackendStatus {
        state: if !up_req {
            BackendState::Disabled
        } else if ffx.dispatch_calls > 0 {
            BackendState::ContextCreated
        } else {
            BackendState::Requested
        },
        requested: up_req,
        enabled: up_req,
        initialized: ffx.create_calls > 0,
        active: false,
        executed_calls: ffx.dispatch_calls,
        fallback_count: 0,
        failure_count: 0,
    };

    let dxr_req = crate::dxr_backend::dxr_enabled() || crate::dxr_proto::dxr_proto_enabled();
    let dxr_calls = dxr.blas_builds + dxr.tlas_builds;
    let dxr_status = BackendStatus {
        state: if !dxr_req {
            BackendState::Disabled
        } else if dxr.tier < 10 {
            BackendState::Unsupported
        } else if dxr_calls > 0 {
            BackendState::Active
        } else {
            BackendState::Requested
        },
        requested: dxr_req,
        enabled: dxr_req,
        initialized: dxr.tier >= 10 && dxr_calls > 0,
        active: dxr_calls > 0,
        executed_calls: dxr_calls,
        fallback_count: 0,
        failure_count: 0,
    };

    let cq_req = upload.copy_queue_enabled;
    let copy_queue = BackendStatus {
        state: if !cq_req {
            BackendState::Disabled
        } else if upload.copy_queue_batches > 0 {
            BackendState::Active
        } else {
            BackendState::Requested
        },
        requested: cq_req,
        enabled: cq_req,
        initialized: upload.copy_queue_batches > 0,
        active: upload.copy_queue_batches > 0,
        executed_calls: upload.copy_queue_batches,
        fallback_count: 0,
        failure_count: 0,
    };

    let audit_req = acc.audit_enabled;
    let accounting_audit = BackendStatus {
        state: if !audit_req && acc.audit_runs == 0 {
            BackendState::Disabled
        } else if acc.audit_runs > 0 {
            BackendState::Active
        } else {
            BackendState::Requested
        },
        requested: audit_req || acc.audit_runs > 0,
        enabled: audit_req || acc.audit_runs > 0,
        initialized: acc.enabled || acc.audit_runs > 0,
        active: acc.audit_runs > 0,
        executed_calls: acc.audit_runs,
        fallback_count: 0,
        failure_count: acc.audit_mismatches,
    };

    format!(
        "\"gpu_authority\":{},\"readback_validate\":{},\"sparse_residency\":{},\"sparse_reclaim\":{},\
         \"mwaitx_fastpath\":{},\"shader_recovery\":{},\"shader_opt\":{},\"ambient_native\":{},\
         \"shadowmask_native\":{},\"gpu_driven\":{},\"gpu_detile\":{},\"rendergraph_fastpath\":{},\
         \"upscaler\":{},\"dxr\":{},\"copy_queue\":{},\"accounting_audit\":{}",
        gpu_authority.json_object(),
        readback_validate.json_object(),
        sparse_residency.json_object(),
        sparse_reclaim.json_object(),
        mwaitx_fastpath.json_object(),
        shader_recovery.json_object(),
        shader_opt.json_object(),
        ambient_native.json_object(),
        shadowmask_native.json_object(),
        gpu_driven.json_object(),
        gpu_detile.json_object(),
        rendergraph_fastpath.json_object(),
        upscaler.json_object(),
        dxr_status.json_object(),
        copy_queue.json_object(),
        accounting_audit.json_object(),
    )
}

#[allow(clippy::too_many_arguments)]
fn format_window_json(
    id: &SessionIdentity,
    seq: u64,
    frame_count: u64,
    fps: f64,
    avg_ms: f64,
    cpu_prep_ms: f64,
    gpu_ms: f64,
    back_ms: f64,
    back_mb: f64,
    draws: f64,
    window_secs: f64,
    pacing: &PacingSnapshot,
    rg: &crate::render_graph::RenderGraphSnapshot,
    vram_used_mb: u64,
    vram_budget_mb: u64,
    nonlocal_used_mb: u64,
    nonlocal_budget_mb: u64,
    acc: &crate::accounting::AccountingSnapshot,
    sparse: &crate::tiles::SparseStats,
    rb: &crate::readback::ReadbackStats,
    dirty: &crate::shadow::DirtyRangeStats,
    (tls_hits, tls_misses): (u64, u64),
    (dyn_allocs, dyn_wraps, dyn_refusals): (u64, u64, u64),
    barriers_elided: u64,
    auth: &crate::authority::AuthorityStats,
    cpu: &CpuTelemetry,
    stream: &StreamingTelemetry,
    upload: &crate::upload_ring::UploadRingStats,
    dxr: &crate::dxr_proto::DxrStats,
    ffx: &FfxHleTelemetry,
    prep: &PrepBreakdown,
) -> String {
    let monotonic_ms = id.started_instant.elapsed().as_millis() as u64;
    let now_unix_ms = utc_unix_ms();
    let acc_comm_mb = acc.committed_textures_bytes >> 20;
    let acc_res_mb = acc.reserved_textures_bytes >> 20;
    let shadow_ram_mb = acc.shadow_ram_bytes >> 20;
    let shadow_vram_mb = acc.shadow_vram_bytes >> 20;
    let staging_mb = acc.staging_bytes >> 20;
    let upload_ring_mb = acc.upload_ring_bytes >> 20;
    let reclaimable_mb = acc.reclaimable_bytes >> 20;
    let backends_json = collect_backends_json(acc, auth, rb, sparse, cpu, upload, dxr, ffx, rg);

    format!(
        "{{\"run_id\":\"{}\",\"pid\":{},\"monotonic_ms\":{},\"utc_unix_ms\":{},\"commit\":\"{}\",\
         \"runtime_hash\":\"{}\",\"preset\":\"{}\",\"seq\":{seq},\"frame_count\":{frame_count},\
         \"window_secs\":{window_secs:.3},\"fps\":{fps:.3},\"avg_ms\":{avg_ms:.3},\
         \"p50_ms\":{:.3},\"p95_ms\":{:.3},\"p99_ms\":{:.3},\"low1_fps\":{:.3},\
         \"stalls_100ms\":{},\"stalls_250ms\":{},\"stalls_500ms\":{},\"stalls_1000ms\":{},\
         \"cpu_prep_ms\":{cpu_prep_ms:.3},\"gpu_ms\":{gpu_ms:.3},{},\
         \"back_ms\":{back_ms:.3},\"back_mb\":{back_mb:.3},\"draws\":{draws:.2},\
         \"vram_used_mb\":{vram_used_mb},\"vram_budget_mb\":{vram_budget_mb},\
         \"dxgi_nonlocal_used_mb\":{nonlocal_used_mb},\"dxgi_nonlocal_budget_mb\":{nonlocal_budget_mb},\
         \"acc_comm_mb\":{acc_comm_mb},\"acc_res_mb\":{acc_res_mb},\"staging_mb\":{staging_mb},\
         \"upload_ring_mb\":{upload_ring_mb},\"reclaimable_mb\":{reclaimable_mb},\
         \"retired_pending\":{},\"sparse_mapped_mb\":{},\"sparse_virt_mb\":{},\
         \"sparse_reclaimed_tiles\":{},\"shadow_ram_mb\":{shadow_ram_mb},\"shadow_vram_mb\":{shadow_vram_mb},\
         \"dxgi_act\":{},\"dxgi_hit\":{},\"audit_runs\":{},\"audit_mismatches\":{},\
         \"rb_tagged\":{},\"rb_hit\":{},\"rb_miss\":{},\"rb_semantic_fallbacks\":{},\"rb_validate_samples\":{},\"rb_validate_warnings\":{},\
         \"dirty_calls\":{},\"dirty_bytes\":{},\"dirty_subsumed\":{},\"dirty_coalesced\":{},\"dirty_raw\":{},\
         \"auth_gpu\":{},\"auth_skipped_cmp\":{},\"tls_hits\":{tls_hits},\"tls_misses\":{tls_misses},\"barriers_elided\":{barriers_elided},\
         \"dyn_allocs\":{dyn_allocs},\"dyn_wraps\":{dyn_wraps},\"dyn_refusals\":{dyn_refusals},\
         \"clzero_fast_calls\":{},\"clzero_fast_bytes\":{},\"wait_fast_calls\":{},\"wait_yields\":{},\"wait_sleeps\":{},\"wait_early_wakes\":{},\
         \"wait_total_us\":{},\"wait_sleep_us\":{},\"emu_total\":{},\
         \"scratch_reuses\":{},\"scratch_bytes_saved\":{},\"copy_queue_batches\":{},\"copy_queue_bytes\":{},\
         \"dxr_tier\":{},\"ffx_0x218_calls\":{},\"ffx_hle_mode\":\"{}\",\"ffx_hle_creates\":{},\"ffx_hle_dispatches\":{},\
         \"prep_ask_once_ms\":{:.3},\"prep_copy_ms\":{:.3},\"prep_fill_ms\":{:.3},\"prep_materialize_ms\":{:.3},\
         \"prep_authority_ms\":{:.3},\"prep_note_gpu_write_ms\":{:.3},\"prep_shadow_update_ms\":{:.3},\
         \"prep_subpage_split_ms\":{:.3},\"prep_tile_lock_ms\":{:.3},\
         \"backends\":{{{}}}}}",
        json_escape(&id.run_id),
        id.pid,
        monotonic_ms,
        now_unix_ms,
        json_escape(&id.commit),
        json_escape(&id.runtime_hash),
        json_escape(&id.preset),
        pacing.p50_ms,
        pacing.p95_ms,
        pacing.p99_ms,
        pacing.low1_fps,
        pacing.stalls_100ms,
        pacing.stalls_250ms,
        pacing.stalls_500ms,
        pacing.stalls_1000ms,
        rg.json_fields(),
        acc.retired_pending_count,
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
        rb.validate_samples,
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
        cpu.wait_yields,
        cpu.wait_sleeps,
        cpu.wait_early_wakes,
        cpu.wait_total_us,
        cpu.wait_sleep_us,
        cpu.emu_total,
        stream.scratch_reuses,
        stream.scratch_bytes_saved,
        upload.copy_queue_batches,
        upload.copy_queue_bytes,
        dxr.tier,
        dxr.ffx_0x218_calls,
        ffx.mode,
        ffx.create_calls,
        ffx.dispatch_calls,
        prep.ask_once_ms,
        prep.copy_ms,
        prep.fill_ms,
        prep.materialize_ms,
        prep.authority_ms,
        prep.note_gpu_write_ms,
        prep.shadow_update_ms,
        prep.subpage_split_ms,
        prep.tile_lock_ms,
        backends_json,
    )
}

/// Emits the expanded 5-second REA-002 / REA-003.1 telemetry summary and optional NDJSON line.
#[allow(clippy::too_many_arguments)]
pub(crate) fn log_window(
    fps: f64,
    avg_ms: f64,
    cpu_prep_ms: f64,
    gpu_ms: f64,
    back_ms: f64,
    back_mb: f64,
    draws: f64,
    window_frames: f64,
    window_secs: f64,
    rg: &crate::render_graph::RenderGraphSnapshot,
) {
    let id = session_identity();
    let seq = WINDOWS_LOGGED.fetch_add(1, Ordering::Relaxed) + 1;
    let frame_count = crate::frame::current();
    let pacing = pacing_snapshot();
    let sched = victory_core::scheduler::stats();
    let acc = crate::accounting::snapshot();
    let [vram_budget, vram_usage, nonlocal_budget, nonlocal_usage] = crate::host::host()
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
    let ffx = FFX_PROVIDER.get().map(|f| f()).unwrap_or(FfxHleTelemetry {
        mode: crate::upscaler::UpscalerMode::from_env().as_str(),
        create_calls: 0,
        dispatch_calls: 0,
    });
    let prep = take_prep_breakdown(window_frames);

    let vram_used_mb = vram_usage >> 20;
    let vram_budget_mb = vram_budget >> 20;
    let nonlocal_used_mb = nonlocal_usage >> 20;
    let nonlocal_budget_mb = nonlocal_budget >> 20;
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
    let gpu_status = rg.gpu_time_status.as_str();

    tracing::info!(
        "REA-002 telemetry: run_id={} seq={seq} frame={frame_count} win_frames={} win_s={window_secs:.2} \
         fps={fps:.1} avg_ms={avg_ms:.2} p50_ms={:.2} p95_ms={:.2} p99_ms={:.2} low1_fps={:.1} \
         stalls_100={} stalls_250={} stalls_500={} stalls_1000={} cpu_prep_ms={cpu_prep_ms:.2} gpu_ms={gpu_ms:.2} \
         gpu_time_status={gpu_status} gpu_hw_ms={:.2} gpu_hw_total_ms={:.2} gpu_hw_samples={} \
         back_ms={back_ms:.2} back_mb={back_mb:.2} draws={draws:.0} \
         standin_draws={:.1} standin_brk={}/{}/{} dropped_pso_draws={:.1} dropped_brk={}/{}/{}/{} dropped_ind={} \
         rg_passes=gfx:{}/depth:{}/mesh:{}/rect:{}/cs:{}/idraw:{}/idisp:{} \
         vram_used_mb={vram_used_mb} vram_budget_mb={vram_budget_mb} nonloc_mb={nonlocal_used_mb}/{nonlocal_budget_mb} \
         acc_comm_mb={acc_comm_mb} acc_res_mb={acc_res_mb} \
         sparse_mapped_mb={} sparse_virt_mb={} sparse_pool_mb={} sparse_reclaimed_tiles={} reclaim_mb={reclaim_mb} \
         shadow_ram_mb={shadow_ram_mb} shadow_vram_mb={shadow_vram_mb} \
         dxgi_act={} dxgi_hit={} audit_runs={} audit_err={} \
         rb_tagged={} rb_hit={} rb_miss={} rb_fb={} rb_val={}/{} \
         dirty_calls={} dirty_mb={dirty_mb:.1} dirty_avx2={} dirty_subsumed={} dirty_coal={}/{} \
         auth_gpu={} auth_cpu={} auth_coh={} auth_skip_cmp={} auth_prom={} \
         tls_hit={tls_hits} tls_miss={tls_misses} dyn_alloc={dyn_allocs} dyn_wrap={dyn_wraps} dyn_ref={dyn_refusals} bar_elided={barriers_elided} \
         sched_on={} cores={}/{} workers={}/{}/{}/{} \
         cpu_fast={} clz_patch={}/{} mon_patch={}/{} clz_fast={} clz_mb={clz_mb:.1} wait_fast={}/{} emu_total={} \
         stream_on={} scratch_reuse={} saved_mb={saved_mb:.1} low_ram_throttle={} \
         ring_on={} pool_hit={} pool_miss={} ring_mb={ring_mb:.1} cq_on={} cq_batches={} cq_mb={cq_mb:.1} \
         dxr_tier={} dxr_blas={} dxr_tlas={} ffx_0x218={} ffx_hle={}:{}/{}",
        id.run_id,
        rg.window_frames,
        pacing.p50_ms,
        pacing.p95_ms,
        pacing.p99_ms,
        pacing.low1_fps,
        pacing.stalls_100ms,
        pacing.stalls_250ms,
        pacing.stalls_500ms,
        pacing.stalls_1000ms,
        rg.gpu_hw_ms,
        rg.gpu_hw_total_ms,
        rg.gpu_hw_samples,
        rg.standin_draws,
        rg.standin_ambient,
        rg.standin_shadowmask,
        rg.standin_exposure,
        rg.dropped_pso_draws,
        rg.dropped_null,
        rg.dropped_pending,
        rg.dropped_failed,
        rg.dropped_other,
        rg.dropped_indirect_only,
        rg.graphics_color,
        rg.depth_prepass,
        rg.mesh_geometry,
        rg.rect_pass,
        rg.compute,
        rg.indirect_draw,
        rg.indirect_dispatch,
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
        rb.validate_warnings,
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
        ffx.mode,
        ffx.create_calls,
        ffx.dispatch_calls,
    );

    if let JsonTarget::Path(path) = json_target() {
        let json = format_window_json(
            id,
            seq,
            frame_count,
            fps,
            avg_ms,
            cpu_prep_ms,
            gpu_ms,
            back_ms,
            back_mb,
            draws,
            window_secs,
            &pacing,
            rg,
            vram_used_mb,
            vram_budget_mb,
            nonlocal_used_mb,
            nonlocal_budget_mb,
            &acc,
            &sparse,
            &rb,
            &dirty,
            (tls_hits, tls_misses),
            (dyn_allocs, dyn_wraps, dyn_refusals),
            barriers_elided,
            &auth,
            &cpu,
            &stream,
            &upload,
            &dxr,
            &ffx,
            &prep,
        );
        if let Err(err) = append_json_line_to(path, &json) {
            tracing::warn!(error = %err, "failed to append REA-002 telemetry JSON line");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let in_repo =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("illegal-telemetry.jsonl");
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

    #[test]
    fn json_target_avoids_colliding_with_existing_non_empty_jsonl() {
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let base = dir.join(format!("victory-rea0031-collision-{pid}.jsonl"));
        let _ = std::fs::remove_file(&base);

        // Fresh/non-existent path uses exact requested path.
        let (first, redirected_first) =
            resolve_unique_json_path(&base, "runA-123").expect("fresh path resolves");
        assert!(!redirected_first);
        assert_eq!(first, base);

        // Once non-empty data exists, a second run gets a distinct `-run-<id>.jsonl` file.
        append_json_line_to(&base, "{\"seq\":1}").expect("write first run");
        let (second, redirected_second) =
            resolve_unique_json_path(&base, "runB/456").expect("collision avoids overwrite");
        assert!(redirected_second);
        assert_ne!(second, base);
        assert!(
            second
                .file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|n| n.contains("-run-runB_456.jsonl"))
        );

        let _ = std::fs::remove_file(&base);
        let _ = std::fs::remove_file(&second);
    }

    #[test]
    fn json_window_includes_gpu_time_status_and_render_graph_and_backend_fields() {
        let _g = crate::render_graph::tests::TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _ = crate::render_graph::take_window_snapshot(1.0, 0.0);
        crate::render_graph::set_ring_available(true);
        crate::render_graph::record_pipeline_exec(
            crate::render_graph::PassKind::StandInAmbient,
            Some(crate::render_graph::StandInKind::Ambient),
            false,
            false,
        );
        crate::render_graph::record_dropped_pso(crate::pipeline::PipelineMiss::Failed);
        crate::render_graph::record_dropped_pso(crate::pipeline::PipelineMiss::Other);
        crate::render_graph::record_dropped_indirect_only();
        let rg = crate::render_graph::take_window_snapshot(5.0, 0.0);

        let id = SessionIdentity {
            run_id: "test-run-1".to_owned(),
            pid: 4242,
            started_instant: Instant::now(),
            preset: "rea0031-test".to_owned(),
            commit: "de79c6c".to_owned(),
            runtime_hash: "sha256:abc".to_owned(),
        };
        let json = format_window_json(
            &id,
            1,
            100,
            30.0,
            33.33,
            5.0,
            0.0,
            1.0,
            0.5,
            1200.0,
            5.0,
            &PacingSnapshot::default(),
            &rg,
            1000,
            7000,
            50,
            4000,
            &crate::accounting::AccountingSnapshot::default(),
            &crate::tiles::SparseStats::default(),
            &crate::readback::ReadbackStats::default(),
            &crate::shadow::DirtyRangeStats::default(),
            (0, 0),
            (0, 0, 0),
            0,
            &crate::authority::AuthorityStats::default(),
            &CpuTelemetry::default(),
            &StreamingTelemetry::default(),
            &crate::upload_ring::UploadRingStats::default(),
            &crate::dxr_proto::DxrStats::default(),
            &FfxHleTelemetry::default(),
            &PrepBreakdown::default(),
        );
        assert!(json.contains("\"run_id\":\"test-run-1\""));
        assert!(json.contains("\"window_frames\":5"));
        assert!(json.contains("\"gpu_time_status\":\"unavailable\""));
        assert!(json.contains("\"standin_ambient\":1"));
        assert!(json.contains("\"dropped_failed\":1"));
        assert!(json.contains("\"dropped_other\":1"));
        assert!(json.contains("\"dropped_window_total\":2"));
        assert!(json.contains("\"dropped_indirect_only\":1"));
        assert!(json.contains("\"submitted_draws\":1"));
        assert!(json.contains("\"gpu_hw_ms\":0.000"));
        assert!(json.contains("\"backends\":{\"gpu_authority\":"));
        assert!(json.contains("\"dxr\":{"));
        assert!(json.contains("\"upscaler\":{"));
        assert!(json.starts_with('{') && json.ends_with("}}"));
    }
}
