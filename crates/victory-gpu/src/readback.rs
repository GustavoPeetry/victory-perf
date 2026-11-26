//! `VICTORY_COPY_BACK_ONLY_READBACK=1` (testing, off by default): what
//! shaders write is copied back into title memory only for the buffers the
//! title reads on the CPU.
//!
//! The title (build 1.0.2.1) makes every buffer it reads back on the CPU
//! through one function (its read-back ring, title RVA 0x1d40f90, 22
//! callers: lighting, occlusion, particles, wind, water, autofocus, ...;
//! read statically). A buffer created while that function is on the stack
//! is noted here; with the switch on, shader writes to other memory stay on
//! the card.
//!
//! shortcut: the function is found on the stack by scanning the stack for
//! a return address into it (the title's unwind data is not registered
//! with the system), so a stale return address left in a caller's frame
//! could tag a buffer that is not read back (it is then copied back as
//! before: harmless).

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

/// The title's read-back buffer creator (1.0.2.1).
const CREATOR_RVA: u32 = 0x1d40f90;
/// Stack bytes scanned above the creation call.
const SCAN_BYTES: usize = 16 << 10;

fn parse_flag(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| {
            let t = v.trim();
            t == "1" || t.eq_ignore_ascii_case("true") || t.eq_ignore_ascii_case("yes")
        })
        .unwrap_or(false)
}

/// Whether the legacy raw switch `VICTORY_COPY_BACK_ONLY_READBACK=1` is on.
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = parse_flag("VICTORY_COPY_BACK_ONLY_READBACK");
        if on {
            tracing::warn!(
                "VICTORY_COPY_BACK_ONLY_READBACK: shader writes are copied back only for the title's read-back buffers"
            );
        }
        on
    })
}

/// Whether `VICTORY_REA_READBACK_TRACE=1` is enabled to log creator provenance
/// and copy-back tag decisions without skipping copy-backs by itself.
pub(crate) fn trace_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = parse_flag("VICTORY_REA_READBACK_TRACE");
        if on {
            tracing::info!(
                "VICTORY_REA_READBACK_TRACE=1: tracing read-back creator provenance and ranges"
            );
        }
        on
    })
}

/// Whether `VICTORY_REA_SEMANTIC_READBACK=1` is enabled (conservative fallback
/// keeps all copy-backs active if creator provenance is unverified or `tagged == 0`).
pub(crate) fn semantic_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = parse_flag("VICTORY_REA_SEMANTIC_READBACK");
        if on {
            tracing::info!(
                "VICTORY_REA_SEMANTIC_READBACK=1: semantic read-back filtering with conservative fallback enabled"
            );
        }
        on
    })
}

/// Whether sampled validation of untagged ranges is enabled (`VICTORY_REA_READBACK_VALIDATE=1`).
pub(crate) fn validate_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse_flag("VICTORY_REA_READBACK_VALIDATE"))
}

static CREATOR_VERIFIED: AtomicBool = AtomicBool::new(false);
static SCAN_HITS: AtomicU64 = AtomicU64::new(0);
static SCAN_MISSES: AtomicU64 = AtomicU64::new(0);
static SEMANTIC_FALLBACKS: AtomicU64 = AtomicU64::new(0);
static VALIDATE_SAMPLES: AtomicU64 = AtomicU64::new(0);
static VALIDATE_WARNINGS: AtomicU64 = AtomicU64::new(0);

/// Telemetry snapshot for Front B readback classification & validation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReadbackStats {
    pub legacy_enabled: bool,
    pub semantic_enabled: bool,
    pub trace_enabled: bool,
    pub validate_enabled: bool,
    pub creator_verified: bool,
    pub tagged_buffers: usize,
    pub scan_hits: u64,
    pub scan_misses: u64,
    pub semantic_fallbacks: u64,
    pub validate_samples: u64,
    pub validate_warnings: u64,
}

/// `[start, end)` of title memory of the read-back buffers alive (let go
/// in [`untag`]: a buffer made later in the same memory need not be one).
static TAGGED: Mutex<crate::memcopy::RangeSet> = Mutex::new(crate::memcopy::RangeSet::new());
static TAGGED_COUNT: AtomicUsize = AtomicUsize::new(0);

/// The creator's code range, once the title image is known: `None` until
/// then, `Some(None)` if it is not this build.
static CREATOR: OnceLock<Option<(usize, usize)>> = OnceLock::new();

/// The `[begin, end)` of the function at `rva` in the image at `base`, from
/// its exception directory.
///
/// # Safety
/// `base` is a mapped PE image.
unsafe fn function_at(base: usize, rva: u32) -> Option<(usize, usize)> {
    // SAFETY: reading the headers of a mapped image.
    unsafe {
        let nt = base + (base as *const u8).add(0x3c).cast::<u32>().read_unaligned() as usize;
        if (nt as *const u32).read_unaligned() != 0x4550 {
            return None;
        }
        // Optional header at +0x18; PE32+ data directories at +0x70; the
        // exception directory is the fourth.
        let dir = nt + 0x18 + 0x70 + 3 * 8;
        let (at, size) = (
            (dir as *const u32).read_unaligned() as usize,
            ((dir + 4) as *const u32).read_unaligned() as usize,
        );
        let entries = std::slice::from_raw_parts((base + at) as *const [u32; 3], size / 12);
        let i = entries.binary_search_by_key(&rva, |e| e[0]).ok()?;
        Some((base + entries[i][0] as usize, base + entries[i][1] as usize))
    }
}

/// The title image holding `address`, if it holds the creator.
fn creator_from(address: usize) -> Option<(usize, usize)> {
    use windows_sys::Win32::System::LibraryLoader::{
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        GetModuleHandleExW,
    };
    let mut module = std::ptr::null_mut();
    // SAFETY: looks up the module holding the address, no reference.
    let found = unsafe {
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            address as *const u16,
            &mut module,
        )
    };
    if found == 0 {
        return None;
    }
    // SAFETY: a module the system knows is a mapped image.
    unsafe { function_at(module as usize, CREATOR_RVA) }
}

/// Multi-frame resolver: walks captured stack frames past host DLL/EXE wrappers
/// until reaching an executable module holding `CREATOR_RVA`.
fn resolve_creator_from_frames(
    frames: &[usize],
    mut resolver: impl FnMut(usize) -> Option<(usize, usize)>,
) -> Option<(usize, usize)> {
    for &addr in frames {
        if addr == 0 {
            continue;
        }
        if let Some(range) = resolver(addr) {
            return Some(range);
        }
    }
    None
}

/// Called by a resource creation method: whether the title's read-back
/// creator is on the stack. Learns the creator's place from the first
/// caller in the title executable (walking past host wrapper frames).
pub(crate) fn creating_readback() -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::RtlCaptureStackBackTrace;
    use windows_sys::Win32::System::Threading::GetCurrentThreadStackLimits;
    if !enabled()
        && !semantic_enabled()
        && !trace_enabled()
        && !crate::authority::enabled()
        && crate::rebar::level() < 2
    {
        return false;
    }
    let creator = match CREATOR.get() {
        Some(c) => *c,
        None => {
            let mut raw_frames = [std::ptr::null_mut(); 16];
            // SAFETY: room for up to 16 stack frames across host/title boundary.
            let captured = unsafe {
                RtlCaptureStackBackTrace(
                    1,
                    raw_frames.len() as u32,
                    raw_frames.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            } as usize;
            let addrs: Vec<usize> = raw_frames[..captured.min(raw_frames.len())]
                .iter()
                .map(|&p| p as usize)
                .collect();
            let Some(c) = resolve_creator_from_frames(&addrs, creator_from) else {
                if trace_enabled() {
                    static PROBES: AtomicU64 = AtomicU64::new(0);
                    if PROBES.fetch_add(1, Ordering::Relaxed) < 8 {
                        tracing::info!(
                            captured,
                            top_frame = format_args!("{:#x}", addrs.first().copied().unwrap_or(0)),
                            "readback trace: no frame resolved to title image holding CREATOR_RVA yet"
                        );
                    }
                }
                // A caller outside the title executable: try again on the next creation call.
                return false;
            };
            CREATOR_VERIFIED.store(true, Ordering::Release);
            tracing::warn!(
                begin = format_args!("{:#x}", c.0),
                end = format_args!("{:#x}", c.1),
                frames_walked = captured,
                "read-back creator found via multi-frame walk"
            );
            *CREATOR.get_or_init(|| Some(c))
        }
    };
    let Some((begin, end)) = creator else {
        return false;
    };
    // First check fast captured return frames (exact unwind when available), then
    // scan the raw stack window for un-registered title unwind tables.
    let mut raw_frames = [std::ptr::null_mut(); 16];
    let captured = unsafe {
        RtlCaptureStackBackTrace(
            1,
            raw_frames.len() as u32,
            raw_frames.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    } as usize;
    for &p in &raw_frames[..captured.min(raw_frames.len())] {
        let addr = p as usize;
        if (begin..end).contains(&addr) {
            SCAN_HITS.fetch_add(1, Ordering::Relaxed);
            return true;
        }
    }
    let (mut low, mut high) = (0usize, 0usize);
    // SAFETY: plain out pointers.
    unsafe { GetCurrentThreadStackLimits(&mut low, &mut high) };
    let here = &low as *const usize as usize;
    let top = (here + SCAN_BYTES).min(high);
    let mut at = here & !7;
    while at + 8 <= top {
        // SAFETY: inside this thread's committed stack, above this frame.
        let v = unsafe { (at as *const usize).read() };
        if (begin..end).contains(&v) {
            SCAN_HITS.fetch_add(1, Ordering::Relaxed);
            return true;
        }
        at += 8;
    }
    SCAN_MISSES.fetch_add(1, Ordering::Relaxed);
    false
}

/// Notes a read-back buffer at `[start, start + len)`.
pub(crate) fn tag(start: u64, len: u64) {
    let mut t = TAGGED.lock().unwrap_or_else(|e| e.into_inner());
    t.insert(start, start + len);
    let n = t.len();
    TAGGED_COUNT.store(n, Ordering::Release);
    drop(t);
    crate::authority::register_range(
        start as usize,
        len as usize,
        crate::authority::AuthorityClass::SharedCoherent,
        true,
    );
    if n.is_power_of_two() || trace_enabled() && n <= 32 {
        tracing::warn!(
            n,
            start = format_args!("{start:#x}"),
            len,
            "read-back buffers tagged"
        );
    }
}

/// Whether a read-back buffer lies in `[start, start + len)`, switch or
/// not (`crate::rebar` keeps such memory where the CPU reads it fast).
pub(crate) fn tagged_in(start: u64, len: u64) -> bool {
    if TAGGED_COUNT.load(Ordering::Acquire) == 0 {
        return false;
    }
    TAGGED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .overlaps(start, start.saturating_add(len))
}

/// Whether shader writes to `[start, start + len)` are copied back: always
/// with switches off; with `VICTORY_COPY_BACK_ONLY_READBACK=1`, only where
/// tagged; with `VICTORY_REA_SEMANTIC_READBACK=1`, falls back conservatively
/// to copy-back if creator provenance isn't verified or `tagged() == 0`.
pub(crate) fn copy_back(start: u64, len: u64) -> bool {
    let legacy = enabled();
    let semantic = semantic_enabled();
    if !legacy && !semantic {
        return true;
    }
    let (overlaps, tagged_count) = {
        let guard = TAGGED.lock().unwrap_or_else(|e| e.into_inner());
        (
            guard.overlaps(start, start.saturating_add(len)),
            guard.len(),
        )
    };

    if semantic && !legacy && (!CREATOR_VERIFIED.load(Ordering::Acquire) || tagged_count == 0) {
        SEMANTIC_FALLBACKS.fetch_add(1, Ordering::Relaxed);
        if validate_enabled() {
            VALIDATE_SAMPLES.fetch_add(1, Ordering::Relaxed);
        }
        return true;
    }

    if overlaps {
        crate::memcopy::add(crate::memcopy::M::BackGoes, 1);
    } else {
        crate::memcopy::add(crate::memcopy::M::BackKept, 1);
        crate::memcopy::add(crate::memcopy::M::BackKeptBytes, len);
        if trace_enabled() {
            static KEPT_LOGS: AtomicU64 = AtomicU64::new(0);
            if KEPT_LOGS.fetch_add(1, Ordering::Relaxed) < 16 {
                tracing::info!(
                    start = format_args!("{start:#x}"),
                    len,
                    tagged_count,
                    "readback trace: skipping non-readback copy-back"
                );
            }
        }
        if validate_enabled() {
            let n = VALIDATE_SAMPLES.fetch_add(1, Ordering::Relaxed);
            if n.is_multiple_of(128)
                && crate::authority::is_gpu_authoritative_conflict(start as usize, len as usize)
            {
                VALIDATE_WARNINGS.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
    overlaps
}

/// Removes one registration owned by a read-back buffer being let go.
/// Its exact range matters: aliases may share a start but differ in size.
pub(crate) fn untag(start: u64, len: u64) {
    // Level 2 / semantic / trace modes also tag read-back buffers when legacy copy-back filtering is off.
    if enabled() || semantic_enabled() || trace_enabled() || crate::rebar::level() >= 2 {
        let mut guard = TAGGED.lock().unwrap_or_else(|e| e.into_inner());
        guard.remove(start, start + len);
        TAGGED_COUNT.store(guard.len(), Ordering::Release);
        drop(guard);
        crate::authority::unregister_range(start as usize, len as usize);
    }
}

/// How many read-back buffers are alive.
pub(crate) fn tagged() -> usize {
    TAGGED_COUNT.load(Ordering::Acquire)
}

/// Return snapshot of REA-002 readback provenance & filtering telemetry.
pub fn stats() -> ReadbackStats {
    ReadbackStats {
        legacy_enabled: enabled(),
        semantic_enabled: semantic_enabled(),
        trace_enabled: trace_enabled(),
        validate_enabled: validate_enabled(),
        creator_verified: CREATOR_VERIFIED.load(Ordering::Relaxed),
        tagged_buffers: tagged(),
        scan_hits: SCAN_HITS.load(Ordering::Relaxed),
        scan_misses: SCAN_MISSES.load(Ordering::Relaxed),
        semantic_fallbacks: SEMANTIC_FALLBACKS.load(Ordering::Relaxed),
        validate_samples: VALIDATE_SAMPLES.load(Ordering::Relaxed),
        validate_warnings: VALIDATE_WARNINGS.load(Ordering::Relaxed),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn untag_releases_buffers_in_all_tracking_modes() {
        const CHILD: &str = "VICTORY_TEST_READBACK_LIFETIME_CHILD";
        const PASSED: &str = "read-back lifetime assertions passed";
        if std::env::var_os(CHILD).is_none() {
            // Each configuration needs its own process: enabled() and level()
            // cache their environment variables in OnceLock.
            for (copy_back, rebar) in [("0", "2"), ("1", "0"), ("1", "1"), ("1", "2")] {
                let output = std::process::Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--exact",
                        "readback::tests::untag_releases_buffers_in_all_tracking_modes",
                        "--nocapture",
                    ])
                    .env(CHILD, "1")
                    .env("VICTORY_COPY_BACK_ONLY_READBACK", copy_back)
                    .env("VICTORY_REBAR", rebar)
                    .output()
                    .expect("run read-back lifetime test in a fresh process");
                assert!(
                    output.status.success()
                        && String::from_utf8_lossy(&output.stdout).contains(PASSED),
                    "copy_back={copy_back}, rebar={rebar}\nstdout:\n{}\nstderr:\n{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr),
                );
            }
            return;
        }

        assert!(enabled() || crate::rebar::level() >= 2);
        // Synthetic addresses only: tag() records ranges without dereferencing
        // title memory or invoking the Windows stack-based creator detector.
        const START: u64 = 0x1000_0000;
        const LEN: u64 = 0x1000;
        const OTHER: u64 = START + 2 * LEN;
        assert!(!tagged_in(START, LEN));
        tag(START, LEN);
        tag(OTHER, LEN);
        assert!(tagged_in(START, LEN));

        untag(START, LEN);
        assert!(
            !tagged_in(START, LEN),
            "released memory must lose its read-back tag"
        );
        assert!(
            tagged_in(OTHER, LEN),
            "another live buffer must stay tagged"
        );

        // A range that is no longer registered is harmless to remove.
        untag(START, LEN);
        assert!(!tagged_in(START, LEN));
        assert!(tagged_in(OTHER, LEN));
        untag(OTHER, LEN);
        assert!(!tagged_in(OTHER, LEN));
        println!("{PASSED}");
    }

    #[test]
    fn this_test_binary_has_its_own_functions_in_its_table() {
        use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
        // SAFETY: our own executable.
        let base = unsafe { GetModuleHandleW(std::ptr::null()) } as usize;
        let f = function_at as *const () as usize - base;
        // SAFETY: our own mapped image.
        let (b, e) = unsafe { function_at(base, f as u32) }.expect("own function in .pdata");
        assert_eq!(b, function_at as *const () as usize);
        assert!(e > b);
    }

    #[test]
    fn multi_frame_walk_skips_host_wrappers() {
        let frames = [0x1000_1111, 0x1000_2222, 0x7ff6_1d40_f900, 0x7ff6_0000_1234];
        let found = resolve_creator_from_frames(&frames, |addr| {
            (addr == 0x7ff6_1d40_f900).then_some((0x7ff6_1d40_f900, 0x7ff6_1d41_0200))
        });
        assert_eq!(found, Some((0x7ff6_1d40_f900, 0x7ff6_1d41_0200)));
    }

    #[test]
    fn semantic_readback_falls_back_conservatively_without_verified_tags() {
        const CHILD: &str = "VICTORY_TEST_SEMANTIC_FALLBACK_CHILD";
        const OK: &str = "semantic fallback assertion passed";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "readback::tests::semantic_readback_falls_back_conservatively_without_verified_tags",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .env("VICTORY_REA_SEMANTIC_READBACK", "1")
                .env("VICTORY_COPY_BACK_ONLY_READBACK", "0")
                .env("VICTORY_REA_READBACK_VALIDATE", "1")
                .output()
                .expect("spawn semantic fallback child test");
            assert!(
                output.status.success() && String::from_utf8_lossy(&output.stdout).contains(OK),
                "stdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }

        assert!(semantic_enabled());
        assert!(!enabled());
        assert_eq!(tagged(), 0);
        // Must preserve full copy-back until creator is verified AND buffers are tagged.
        assert!(copy_back(0x5000_0000, 4096));
        assert!(stats().semantic_fallbacks >= 1);
        println!("{OK}");
    }
}
