//! Desktop-aware CPU topology probe, worker sizing, and scheduling hints.
//!
//! All policy changes stay inert unless `VICTORY_REA_HOST_SCHEDULER=1` is set,
//! preserving bit-exact baseline behavior when running without REA-002 flags.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// Functional role of a host worker or title thread.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThreadRole {
    /// Main frame / D3D12 command submission thread.
    RenderSubmit,
    /// DirectStorage / I/O decompression worker.
    StreamingIo,
    /// Background DXBC -> DXIL / PSO compilation worker.
    ShaderCompile,
    /// Software DSM shadow compare / merge / copy worker.
    ShadowWorker,
    /// Periodic memory accounting / telemetry thread.
    BackgroundTelemetry,
}

/// Detected or synthetic host CPU topology summary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuTopology {
    pub physical_cores: u16,
    pub logical_processors: u16,
    pub numa_nodes: u16,
    pub l3_domains: u16,
    pub has_hybrid_cores: bool,
    pub cpu_set_count: u16,
    /// Bitmask of logical processors belonging to the highest efficiency class (P-cores),
    /// or all logical processors on homogeneous CPUs.
    pub preferred_mask: usize,
}

impl Default for CpuTopology {
    fn default() -> Self {
        Self {
            physical_cores: 4,
            logical_processors: 8,
            numa_nodes: 1,
            l3_domains: 1,
            has_hybrid_cores: false,
            cpu_set_count: 8,
            preferred_mask: 0xFF,
        }
    }
}

/// Optional env overrides parsed from `VICTORY_REA_*` variables.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SchedulerOverrides {
    pub enabled: bool,
    pub streaming_workers: Option<u16>,
    pub shader_workers: Option<u16>,
    pub shadow_workers: Option<u16>,
    pub memcopy_workers: Option<u16>,
    pub max_affinity_cores: Option<u16>,
}

/// Resolved worker sizing & title scheduling policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkerPlan {
    pub enabled: bool,
    pub topology: CpuTopology,
    pub streaming_workers: u16,
    pub shader_workers: u16,
    pub shadow_workers: u16,
    pub memcopy_workers: u16,
    pub active_mask: usize,
    pub widen_single_core_masks: bool,
    pub clamp_excessive_priorities: bool,
}

/// Snapshot of runtime scheduler decisions and telemetry counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SchedulerStats {
    pub enabled: bool,
    pub physical_cores: u16,
    pub logical_processors: u16,
    pub l3_domains: u16,
    pub numa_nodes: u16,
    pub has_hybrid_cores: bool,
    pub streaming_workers: u16,
    pub shader_workers: u16,
    pub shadow_workers: u16,
    pub memcopy_workers: u16,
    pub affinity_widened: u64,
    pub affinity_clamped: u64,
    pub priority_clamped: u64,
    pub host_hints_applied: u64,
}

static TOPOLOGY: OnceLock<CpuTopology> = OnceLock::new();
static PLAN: OnceLock<WorkerPlan> = OnceLock::new();

static AFFINITY_WIDENED: AtomicU64 = AtomicU64::new(0);
static AFFINITY_CLAMPED: AtomicU64 = AtomicU64::new(0);
static PRIORITY_CLAMPED: AtomicU64 = AtomicU64::new(0);
static HOST_HINTS_APPLIED: AtomicU64 = AtomicU64::new(0);

fn parse_bool_env(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| {
            let trimmed = v.trim();
            trimmed == "1"
                || trimmed.eq_ignore_ascii_case("true")
                || trimmed.eq_ignore_ascii_case("yes")
        })
        .unwrap_or(false)
}

fn parse_u16_env(name: &str) -> Option<u16> {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<u16>().ok())
        .filter(|&n| n > 0)
}

fn load_overrides_from_env() -> SchedulerOverrides {
    SchedulerOverrides {
        enabled: parse_bool_env("VICTORY_REA_HOST_SCHEDULER"),
        streaming_workers: parse_u16_env("VICTORY_REA_STREAMING_WORKERS"),
        shader_workers: parse_u16_env("VICTORY_REA_SHADER_WORKERS"),
        shadow_workers: parse_u16_env("VICTORY_REA_SHADOW_WORKERS"),
        memcopy_workers: parse_u16_env("VICTORY_REA_MEMCOPY_WORKERS"),
        max_affinity_cores: parse_u16_env("VICTORY_REA_MAX_AFFINITY_CORES"),
    }
}

/// Probe host CPU topology once (uses `GetLogicalProcessorInformationEx` +
/// `GetSystemCpuSetInformation` on Windows).
pub fn topology() -> CpuTopology {
    *TOPOLOGY.get_or_init(probe_topology)
}

/// Compute a deterministic worker plan for the given topology and overrides.
pub fn plan_for_topology(topo: CpuTopology, overrides: SchedulerOverrides) -> WorkerPlan {
    let logical = topo.logical_processors.max(1);
    let physical = topo.physical_cores.max(1);

    // Baseline defaults mirror existing VIctory behavior when scheduler is disabled.
    let baseline_streaming = logical.max(1);
    let baseline_shader = logical.max(1);
    let baseline_shadow = logical.clamp(1, 8);
    let baseline_memcopy = logical.clamp(1, 6);

    if !overrides.enabled {
        return WorkerPlan {
            enabled: false,
            topology: topo,
            streaming_workers: baseline_streaming,
            shader_workers: baseline_shader,
            shadow_workers: baseline_shadow,
            memcopy_workers: baseline_memcopy,
            active_mask: topo.preferred_mask.max(1),
            widen_single_core_masks: false,
            clamp_excessive_priorities: false,
        };
    }

    // Desktop-aware sizing tailored for 4C/8T (i3-13100F) through 16C+ systems:
    // keep total runnable background threads well below saturating all physical cores
    // alongside the title's main + worker threads and D3D12 submission thread.
    let auto_streaming = if physical <= 4 {
        2
    } else {
        (physical / 2).clamp(2, 4)
    };
    let auto_shader = if physical <= 4 {
        2
    } else {
        physical.saturating_sub(2).clamp(2, 6)
    };
    let auto_shadow = if physical <= 4 {
        2
    } else {
        (physical / 2).clamp(2, 6)
    };
    let auto_memcopy = if physical <= 4 {
        2
    } else {
        (physical / 2).clamp(2, 4)
    };

    let streaming_workers = overrides
        .streaming_workers
        .unwrap_or(auto_streaming)
        .clamp(1, logical);
    let shader_workers = overrides
        .shader_workers
        .unwrap_or(auto_shader)
        .clamp(1, logical);
    let shadow_workers = overrides
        .shadow_workers
        .unwrap_or(auto_shadow)
        .clamp(1, logical);
    let memcopy_workers = overrides
        .memcopy_workers
        .unwrap_or(auto_memcopy)
        .clamp(1, logical);

    let mut active_mask = if topo.preferred_mask != 0 {
        topo.preferred_mask
    } else {
        low_bits_mask(logical)
    };

    if let Some(max_cores) = overrides.max_affinity_cores {
        let cap_bits = low_bits_mask(max_cores.min(logical));
        let masked = active_mask & cap_bits;
        if masked != 0 {
            active_mask = masked;
        } else {
            active_mask = cap_bits;
        }
    }

    WorkerPlan {
        enabled: true,
        topology: topo,
        streaming_workers,
        shader_workers,
        shadow_workers,
        memcopy_workers,
        active_mask: active_mask.max(1),
        widen_single_core_masks: true,
        clamp_excessive_priorities: true,
    }
}

const fn low_bits_mask(bits: u16) -> usize {
    let shift = if bits as u32 >= usize::BITS {
        usize::BITS
    } else {
        bits as u32
    };
    if shift == 0 {
        1
    } else if shift >= usize::BITS {
        usize::MAX
    } else {
        (1usize << shift) - 1
    }
}

/// Global cached REA-002 scheduler plan.
pub fn plan() -> &'static WorkerPlan {
    PLAN.get_or_init(|| {
        let topo = topology();
        let ov = load_overrides_from_env();
        let resolved = plan_for_topology(topo, ov);
        if resolved.enabled {
            tracing::info!(
                physical_cores = topo.physical_cores,
                logical_processors = topo.logical_processors,
                l3_domains = topo.l3_domains,
                has_hybrid = topo.has_hybrid_cores,
                streaming = resolved.streaming_workers,
                shader = resolved.shader_workers,
                shadow = resolved.shadow_workers,
                memcopy = resolved.memcopy_workers,
                active_mask = format_args!("{:#x}", resolved.active_mask),
                "REA-002 host scheduler active"
            );
        }
        resolved
    })
}

/// Whether `VICTORY_REA_HOST_SCHEDULER=1` is active.
pub fn enabled() -> bool {
    plan().enabled
}

/// Resolve DirectStorage / streaming worker count.
pub fn streaming_workers(baseline_default: usize) -> usize {
    let p = plan();
    if p.enabled {
        p.streaming_workers as usize
    } else {
        baseline_default
    }
}

/// Resolve shader compilation worker count.
pub fn shader_workers(baseline_default: usize) -> usize {
    let p = plan();
    if p.enabled {
        p.shader_workers as usize
    } else {
        baseline_default
    }
}

/// Resolve shadow compare / wide merge worker count.
pub fn shadow_workers(baseline_default: usize) -> usize {
    let p = plan();
    if p.enabled {
        p.shadow_workers as usize
    } else {
        baseline_default
    }
}

/// Resolve async CPU memcopy worker count.
pub fn memcopy_workers(baseline_default: usize) -> usize {
    let p = plan();
    if p.enabled {
        p.memcopy_workers as usize
    } else {
        baseline_default
    }
}

/// Transform a title-requested affinity mask for desktop core counts.
///
/// On 4-core desktops, titles compiled for fixed console core assignments often
/// pin heavyweight workers to a single hardware thread or request bits above the
/// host mask. When enabled, widen single-thread pins onto the active P-core mask
/// (preserving multi-thread masks masked down to available bits).
pub fn adapt_title_affinity(requested_mask: usize) -> usize {
    let p = plan();
    if !p.enabled {
        return requested_mask;
    }
    let valid = p.active_mask;
    if valid == 0 {
        return requested_mask;
    }

    let masked = requested_mask & valid;
    if masked == 0 {
        AFFINITY_WIDENED.fetch_add(1, Ordering::Relaxed);
        return valid;
    }

    // If the title asked for a single logical core on a constrained host (<= 4C/8T)
    // widen to the preferred active mask so OS scheduler can migrate across free threads
    // without stalling on a saturated core.
    if p.widen_single_core_masks
        && p.topology.physical_cores <= 4
        && masked.count_ones() <= 1
        && valid.count_ones() > 1
    {
        AFFINITY_WIDENED.fetch_add(1, Ordering::Relaxed);
        return valid;
    }

    if masked != requested_mask {
        AFFINITY_CLAMPED.fetch_add(1, Ordering::Relaxed);
    }
    masked
}

/// Clamp extreme title thread priorities (`TIME_CRITICAL` -> `HIGHEST` / `ABOVE_NORMAL`)
/// so Xbox-assumed priority boosts cannot starve Windows desktop UI, D3D12 submission,
/// or Audio/Input threads on 4-core CPUs.
pub fn clamp_title_priority(requested_priority: i32) -> i32 {
    let p = plan();
    if !p.enabled || !p.clamp_excessive_priorities {
        return requested_priority;
    }
    // Windows priority constants: TIME_CRITICAL = 15, HIGHEST = 2, ABOVE_NORMAL = 1.
    if requested_priority >= 15 {
        PRIORITY_CLAMPED.fetch_add(1, Ordering::Relaxed);
        return 2; // THREAD_PRIORITY_HIGHEST
    }
    if p.topology.physical_cores <= 4 && requested_priority > 1 {
        PRIORITY_CLAMPED.fetch_add(1, Ordering::Relaxed);
        return 1; // THREAD_PRIORITY_ABOVE_NORMAL
    }
    requested_priority
}

/// Apply host scheduling hints (thread priority + ideal processor / affinity on P-cores)
/// to the calling thread when `VICTORY_REA_HOST_SCHEDULER=1` is enabled.
pub fn apply_current_thread_role(role: ThreadRole, worker_index: usize) {
    let p = plan();
    if !p.enabled {
        return;
    }
    HOST_HINTS_APPLIED.fetch_add(1, Ordering::Relaxed);
    apply_os_thread_hints(role, worker_index, p);
}

/// Return current scheduler statistics snapshot.
pub fn stats() -> SchedulerStats {
    let p = plan();
    SchedulerStats {
        enabled: p.enabled,
        physical_cores: p.topology.physical_cores,
        logical_processors: p.topology.logical_processors,
        l3_domains: p.topology.l3_domains,
        numa_nodes: p.topology.numa_nodes,
        has_hybrid_cores: p.topology.has_hybrid_cores,
        streaming_workers: p.streaming_workers,
        shader_workers: p.shader_workers,
        shadow_workers: p.shadow_workers,
        memcopy_workers: p.memcopy_workers,
        affinity_widened: AFFINITY_WIDENED.load(Ordering::Relaxed),
        affinity_clamped: AFFINITY_CLAMPED.load(Ordering::Relaxed),
        priority_clamped: PRIORITY_CLAMPED.load(Ordering::Relaxed),
        host_hints_applied: HOST_HINTS_APPLIED.load(Ordering::Relaxed),
    }
}

#[cfg(windows)]
fn apply_os_thread_hints(role: ThreadRole, worker_index: usize, plan: &WorkerPlan) {
    use windows_sys::Win32::System::Threading::{
        GetCurrentThread, SetThreadIdealProcessor, SetThreadPriority, THREAD_PRIORITY_ABOVE_NORMAL,
        THREAD_PRIORITY_BELOW_NORMAL, THREAD_PRIORITY_LOWEST,
    };

    let prio = match role {
        ThreadRole::RenderSubmit => THREAD_PRIORITY_ABOVE_NORMAL,
        ThreadRole::StreamingIo | ThreadRole::ShaderCompile | ThreadRole::ShadowWorker => {
            THREAD_PRIORITY_BELOW_NORMAL
        }
        ThreadRole::BackgroundTelemetry => THREAD_PRIORITY_LOWEST,
    };

    unsafe {
        let th = GetCurrentThread();
        let _ = SetThreadPriority(th, prio);
        if let Some(ideal) = pick_nth_set_bit(plan.active_mask, worker_index) {
            let _ = SetThreadIdealProcessor(th, ideal as u32);
        }
    }
}

#[cfg(not(windows))]
fn apply_os_thread_hints(_role: ThreadRole, _worker_index: usize, _plan: &WorkerPlan) {}

fn pick_nth_set_bit(mask: usize, n: usize) -> Option<usize> {
    let total = mask.count_ones() as usize;
    if total == 0 {
        return None;
    }
    let target = n % total;
    let mut seen = 0usize;
    for bit in 0..(usize::BITS as usize) {
        if (mask & (1usize << bit)) != 0 {
            if seen == target {
                return Some(bit);
            }
            seen += 1;
        }
    }
    None
}

#[cfg(windows)]
fn probe_topology() -> CpuTopology {
    use windows_sys::Win32::Foundation::{ERROR_INSUFFICIENT_BUFFER, GetLastError};
    use windows_sys::Win32::System::SystemInformation::{
        GetLogicalProcessorInformationEx, GetSystemCpuSetInformation, RelationAll, RelationCache,
        RelationNumaNode, RelationProcessorCore, SYSTEM_CPU_SET_INFORMATION,
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
    };

    let fallback_logical = std::thread::available_parallelism()
        .map(|n| n.get().min(u16::MAX as usize) as u16)
        .unwrap_or(4);

    let mut physical_cores = 0u16;
    let mut logical_processors = 0u16;
    let mut numa_nodes = 0u16;
    let mut l3_domains = 0u16;
    let mut max_efficiency = 0u8;
    let mut min_efficiency = u8::MAX;
    let mut best_mask = 0usize;

    let mut bytes = 0u32;
    let ok =
        unsafe { GetLogicalProcessorInformationEx(RelationAll, std::ptr::null_mut(), &mut bytes) };
    if ok == 0 && unsafe { GetLastError() } == ERROR_INSUFFICIENT_BUFFER && bytes > 0 {
        let mut buf = vec![0u8; bytes as usize];
        let ok2 = unsafe {
            GetLogicalProcessorInformationEx(
                RelationAll,
                buf.as_mut_ptr()
                    .cast::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(),
                &mut bytes,
            )
        };
        if ok2 != 0 {
            let mut offset = 0usize;
            while offset + std::mem::size_of::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>()
                <= bytes as usize
            {
                let info = unsafe {
                    &*buf
                        .as_ptr()
                        .add(offset)
                        .cast::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>()
                };
                if info.Size == 0 {
                    break;
                }
                match info.Relationship {
                    r if r == RelationProcessorCore => {
                        physical_cores = physical_cores.saturating_add(1);
                        let proc = unsafe { &info.Anonymous.Processor };
                        let eff = proc.EfficiencyClass;
                        if eff > max_efficiency {
                            max_efficiency = eff;
                            best_mask = 0;
                        }
                        if eff < min_efficiency {
                            min_efficiency = eff;
                        }
                        let mut core_mask = 0usize;
                        for g in 0..(proc.GroupCount as usize).min(1) {
                            let m = proc.GroupMask[g].Mask;
                            core_mask |= m;
                            logical_processors =
                                logical_processors.saturating_add(m.count_ones() as u16);
                        }
                        if eff >= max_efficiency {
                            best_mask |= core_mask;
                        }
                    }
                    r if r == RelationNumaNode => {
                        numa_nodes = numa_nodes.saturating_add(1);
                    }
                    r if r == RelationCache => {
                        let cache = unsafe { &info.Anonymous.Cache };
                        if cache.Level == 3 {
                            l3_domains = l3_domains.saturating_add(1);
                        }
                    }
                    _ => {}
                }
                offset += info.Size as usize;
            }
        }
    }

    // Also query Windows 10/11 CPU Sets to confirm parked/hybrid scheduling units.
    let mut cpu_set_bytes = 0u32;
    let mut cpu_set_count = 0u16;
    unsafe {
        let _ = GetSystemCpuSetInformation(
            std::ptr::null_mut(),
            0,
            &mut cpu_set_bytes,
            std::ptr::null_mut(),
            0,
        );
    }
    if cpu_set_bytes > 0 {
        let mut cs_buf = vec![0u8; cpu_set_bytes as usize];
        let ok_cs = unsafe {
            GetSystemCpuSetInformation(
                cs_buf.as_mut_ptr().cast::<SYSTEM_CPU_SET_INFORMATION>(),
                cpu_set_bytes,
                &mut cpu_set_bytes,
                std::ptr::null_mut(),
                0,
            )
        };
        if ok_cs != 0 {
            let mut off = 0usize;
            while off + 8 <= cpu_set_bytes as usize {
                let entry = unsafe {
                    &*cs_buf
                        .as_ptr()
                        .add(off)
                        .cast::<SYSTEM_CPU_SET_INFORMATION>()
                };
                if entry.Size == 0 {
                    break;
                }
                cpu_set_count = cpu_set_count.saturating_add(1);
                off += entry.Size as usize;
            }
        }
    }

    let logical = if logical_processors > 0 {
        logical_processors
    } else {
        fallback_logical
    };
    let physical = if physical_cores > 0 {
        physical_cores
    } else {
        (logical / 2).max(1)
    };
    let preferred_mask = if best_mask != 0 {
        best_mask
    } else {
        low_bits_mask(logical)
    };

    CpuTopology {
        physical_cores: physical,
        logical_processors: logical,
        numa_nodes: numa_nodes.max(1),
        l3_domains: l3_domains.max(1),
        has_hybrid_cores: min_efficiency != u8::MAX && max_efficiency > min_efficiency,
        cpu_set_count: cpu_set_count.max(logical),
        preferred_mask,
    }
}

#[cfg(not(windows))]
fn probe_topology() -> CpuTopology {
    let logical = std::thread::available_parallelism()
        .map(|n| n.get().min(u16::MAX as usize) as u16)
        .unwrap_or(4);
    CpuTopology {
        physical_cores: (logical / 2).max(1),
        logical_processors: logical,
        numa_nodes: 1,
        l3_domains: 1,
        has_hybrid_cores: false,
        cpu_set_count: logical,
        preferred_mask: low_bits_mask(logical),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_topology_probe_is_valid() {
        let topo = topology();
        assert!(topo.physical_cores >= 1);
        assert!(topo.logical_processors >= topo.physical_cores);
        assert!(topo.numa_nodes >= 1);
        assert!(topo.preferred_mask != 0);
    }

    #[test]
    fn disabled_scheduler_preserves_baseline_defaults() {
        let topo = CpuTopology {
            physical_cores: 4,
            logical_processors: 8,
            ..Default::default()
        };
        let plan = plan_for_topology(topo, SchedulerOverrides::default());
        assert!(!plan.enabled);
        assert_eq!(plan.streaming_workers, 8);
        assert_eq!(plan.shader_workers, 8);
        assert!(!plan.widen_single_core_masks);
    }

    #[test]
    fn four_core_desktop_plan_prevents_oversubscription() {
        let topo = CpuTopology {
            physical_cores: 4,
            logical_processors: 8,
            numa_nodes: 1,
            l3_domains: 1,
            has_hybrid_cores: false,
            cpu_set_count: 8,
            preferred_mask: 0xFF,
        };
        let plan = plan_for_topology(
            topo,
            SchedulerOverrides {
                enabled: true,
                ..Default::default()
            },
        );
        assert!(plan.enabled);
        assert_eq!(plan.streaming_workers, 2);
        assert_eq!(plan.shader_workers, 2);
        assert_eq!(plan.shadow_workers, 2);
        assert_eq!(plan.memcopy_workers, 2);
        assert_eq!(plan.active_mask, 0xFF);
        assert!(plan.widen_single_core_masks);
    }

    #[test]
    fn pick_nth_set_bit_cycles_valid_cpus() {
        assert_eq!(pick_nth_set_bit(0b1010, 0), Some(1));
        assert_eq!(pick_nth_set_bit(0b1010, 1), Some(3));
        assert_eq!(pick_nth_set_bit(0b1010, 2), Some(1));
        assert_eq!(pick_nth_set_bit(0, 0), None);
    }
}
