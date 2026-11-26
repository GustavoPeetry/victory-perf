//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: write
//! watching, or read-only guards on physical-page views), all of it where
//! neither works. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! shortcut: only the ranges draws are known to read are brought up to
//! date (vertex and index buffer views, constant buffers in tables, root
//! descriptors, buffer views in tables); buffers read through directly
//! indexed heaps keep what their first use copied.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// Per-page last copied write-watch sequence, tracking how many pages have
/// never been copied whole (`uncopied == 0` enables the O(1)/O(|written|) fast
/// path in [`Shadow::copy_stale`]).
struct CopiedPages {
    at: Vec<u64>,
    uncopied: usize,
}

impl CopiedPages {
    fn new(pages: usize) -> Self {
        Self {
            at: vec![0; pages],
            uncopied: pages,
        }
    }

    fn get(&self, index: usize) -> Option<u64> {
        self.at.get(index).copied()
    }

    fn set(&mut self, index: usize, now: u64) {
        if let Some(slot) = self.at.get_mut(index) {
            if *slot == 0 && now != 0 {
                self.uncopied -= 1;
            } else if *slot != 0 && now == 0 {
                self.uncopied += 1;
            }
            *slot = now;
        }
    }

    fn all_copied(&self) -> bool {
        self.uncopied == 0
    }

    #[cfg(any(test, feature = "diagnostics"))]
    fn copied_count(&self) -> usize {
        self.at.len() - self.uncopied
    }
}

/// A title buffer's host copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GpuOwnedRange {
    start: usize,
    end: usize,
    watch_seq: u64,
}

#[cfg(test)]
static COPY_CALLS: AtomicU64 = AtomicU64::new(0);

static ACTIVE_GPU_OWNED: AtomicU64 = AtomicU64::new(0);
static GPU_OWNED_MIN: AtomicU64 = AtomicU64::new(u64::MAX);
static GPU_OWNED_MAX: AtomicU64 = AtomicU64::new(0);
static GPU_OWNED_SHADOWS: RwLock<Vec<(u64, u64, u64, std::sync::Weak<Shadow>)>> =
    RwLock::new(Vec::new());

/// Fast lock-free check for whether any buffer currently holds unmaterialized GPU-owned intervals.
#[inline]
pub(crate) fn has_active_gpu_owned() -> bool {
    ACTIVE_GPU_OWNED.load(Ordering::Relaxed) > 0
}

fn recompute_gpu_owned_bounds(list: &[(u64, u64, u64, std::sync::Weak<Shadow>)]) {
    let (mut min_v, mut max_v) = (u64::MAX, 0u64);
    for &(_, start, end, _) in list {
        min_v = min_v.min(start);
        max_v = max_v.max(end);
    }
    GPU_OWNED_MIN.store(min_v, Ordering::Release);
    GPU_OWNED_MAX.store(max_v, Ordering::Release);
}

fn register_gpu_owned_shadow(shadow: &Arc<Shadow>) {
    let mut list = GPU_OWNED_SHADOWS.write().unwrap_or_else(|e| e.into_inner());
    list.retain(|(_, _, _, w)| w.strong_count() > 0);
    if !list.iter().any(|&(id, _, _, _)| id == shadow.id) {
        list.push((
            shadow.id,
            shadow.start,
            shadow.start.saturating_add(shadow.len),
            Arc::downgrade(shadow),
        ));
    }
    recompute_gpu_owned_bounds(&list);
}

fn unregister_gpu_owned_shadow(shadow_id: u64) {
    let mut list = GPU_OWNED_SHADOWS.write().unwrap_or_else(|e| e.into_inner());
    let before = list.len();
    list.retain(|&(id, _, _, ref w)| id != shadow_id && w.strong_count() > 0);
    if list.len() != before {
        recompute_gpu_owned_bounds(&list);
    }
}

pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// Per page of title memory from `start`'s on: the write-watch asking
    /// after which it was last copied (0: never).
    copied: Mutex<CopiedPages>,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<Vec<(usize, usize)>>,
    /// With [`relay`]: how many submissions not yet copied back declared
    /// each range of `pending` ([`Flights`]). Locked inside `pending`'s
    /// lock.
    flights: Mutex<Flights>,
    /// Title-VA intervals whose authoritative latest bytes live in this shadow
    /// (and NOT yet materialized into title RAM), paired with the watch-asking
    /// watermark active when the GPU write occurred.
    gpu_owned: Mutex<Vec<GpuOwnedRange>>,
    /// Lock-free fast-path gate for `gpu_owned` (`true` iff non-empty).
    has_gpu_owned: std::sync::atomic::AtomicBool,
    /// Asking sequence of the most recent cold (`at == 0`) page copy on this
    /// shadow, cleared once a subsequent full-shadow `GetWriteWatch(RESET)`
    /// (`now > cold_at`) completes with `copied.all_copied()`.
    cold_copied_asking: AtomicU64,
    /// Per-4 KiB subpage copy watermarks (`kpage -> asking`), populated only
    /// on 64 KiB pages that intersect `gpu_owned` to avoid false-sharing clobbers.
    copied_4k: Mutex<std::collections::HashMap<usize, u64>>,
    /// The copy's bytes before the submission ran, for the ranges its
    /// shaders write (`pending`), each at its offset in the copy: what the
    /// GPU wrote differs. Room for the whole copy; only those ranges' bytes
    /// are set (and read).
    before: Mutex<Option<Vec<u8>>>,
    /// With snapshots kept ([`keeps`]): which 64 KiB pieces of the snapshot
    /// hold the copy's bytes as they are now, and whether the snapshot is
    /// counted in [`KEPT_BYTES`].
    kept: Mutex<(Vec<bool>, bool)>,
    /// When the copy was last asked for (`crate::host::moment`): a window
    /// outside buffers nothing asked for in a while, and nothing holds, is
    /// let go ([`let_go_idle`]).
    used: AtomicU64,
    /// The lowest and highest title address asked for in the copy (a
    /// window is 4 MiB whatever was asked: measured for the memory line).
    asked: [AtomicU64; 2],
    /// Asks for the copy that gave no length (root views: they read from
    /// their address to the copy's end, whatever lies there).
    unsized_asks: AtomicU64,
    /// A number no other copy has had. Its address is no name for a copy:
    /// one let go and another made may share it.
    id: u64,
    /// With [`cards`], for a big title buffer: `buffer` is in the card's
    /// own memory, and this is the rest of the copy.
    card: Option<Card>,
    /// A window of title memory outside buffers, not a title buffer's copy.
    window: bool,
    /// With `crate::inplace`: not a copy at all. `buffer` is placed in a
    /// heap opened on the title memory itself, so shaders read and write
    /// the title's own bytes and nothing is copied either way. The heap,
    /// and the bytes from the buffer's first byte to `start` (the heap
    /// begins on a 64 KiB line; a title buffer need not).
    place: Option<(windows::Win32::Graphics::Direct3D12::ID3D12Heap, u64)>,
    /// Bytes reserved in `crate::rebar` (0 if not placed in GPU upload heap).
    rebar: u64,
    /// Live-byte telemetry bucket (`0..=4` in [`LIVE_CLASS`]; `u8::MAX` if uncounted).
    live_class: u8,
}

/// What a copy in the card's memory ([`cards`]) has beside the buffer
/// shaders see.
struct Card {
    /// The CPU's side, in the PC's memory: what `data` maps. Title memory
    /// is copied into it as into any copy, and what changed in it goes to
    /// the card's side before the next submission.
    staging: ID3D12Resource,
    /// The card side's bytes as they were when the submission's shaders
    /// started, in the card's memory too: made when shader writes are
    /// first declared for the copy, and compared with the card's side, on
    /// the card, after each submission that declares some.
    before: Mutex<Option<ID3D12Resource>>,
    /// Offsets of the copy whose bytes changed on the CPU's side and have
    /// not gone to the card's side yet.
    dirty: Mutex<Vec<(usize, usize)>>,
    /// With [`relay`]: the offsets that went from `dirty` to the card's
    /// side, each with the number of the send that took it
    /// ([`card_dirty`]), kept until a fetch sent after it has been merged.
    /// Locked inside `dirty`'s lock.
    sent: Mutex<Vec<(usize, usize, u64)>>,
    /// The buffers' bytes (the title buffer's, rounded up to 64 KiB).
    size: u64,
}

/// Whether the host copies of big title buffers are in the card's own
/// memory (on by default since run 75 showed it holds: the BVH skinning
/// list went from 50.5 to 4.5 ms on the card; `VICTORY_CARD_BUFFERS=0`
/// keeps every copy in the PC's memory, as before. Read when a copy is
/// made, so it takes a fresh start).
///
/// Every host copy was a buffer in the PC's memory that the card reads
/// and writes over the bus, so that our code can fill it and compare it.
/// The title's shaders read almost all their buffer data from a few big
/// title buffers (in the gameplay frame of run 69, 190-330 MB of distinct
/// ranges of the 174 MB heap per list against 0-13 MB elsewhere), and a
/// skinning dispatch of ONE group of 32 threads took 1.8 ms on the card
/// (run 70): the time of some 170 loads one after the other, not of the
/// arithmetic.
///
/// With the switch, a title buffer of [`card_limits`]' size or more has two
/// sides. Shaders see `Shadow::buffer`, in the card's memory. The CPU's
/// side (`Card::staging`, what `Shadow::data` maps) takes title memory as
/// before, and the ranges that changed in it are copied to the card's side
/// before each submission (`Host::card_before`). What shaders wrote is
/// found on the card: after the submission's lists a compute pass of ours
/// compares each declared range, by 64 KiB piece, with the card side's
/// bytes from before (`Card::before`), and only the pieces that differ are
/// fetched and merged into title memory, byte by byte as
/// [`Shadow::copy_back`] does (`Host::card_back`, [`Shadow::card_merge`]).
/// No snapshot is taken and nothing is compared on the CPU for such a copy.
///
/// What it assumes is what [`keeps`] assumes: shaders write a copy only in
/// ranges declared as written. A write outside them is seen by later
/// shaders, as before, and goes back to the title the next time a declared
/// range holds its piece (before: never).
pub(crate) fn cards() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_BUFFERS").is_none_or(|v| v != "0"))
}

/// The least size of a title buffer whose copy goes to the card's memory, and the
/// most bytes of such copies (their "before" hashes not counted), both in bytes:
/// `VICTORY_CARD_BUFFERS_MIN_KB` (default 256 KiB; `VICTORY_CARD_BUFFERS_MIN_MB`
/// overrides in MiB if explicitly set) and `VICTORY_CARD_BUFFERS_MB` (default
/// 2048 MiB).
fn card_limits() -> (u64, u64) {
    static LIMITS: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    *LIMITS.get_or_init(|| {
        let mb = |name: &str, default: u64| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.trim().parse::<u64>().ok())
                .unwrap_or(default)
                << 20
        };
        // From 256 KiB up (checked against actual buffer length, not 64 KiB
        // rounded size) and 2,048 MiB total by default: moves medium-to-large
        // shader-written buffers (640-828 KiB+) onto VRAM + GPU CARD_COMPARE
        // while keeping tiny buffers out of card-compare dispatch churn.
        let min_bytes = if let Ok(v) = std::env::var("VICTORY_CARD_BUFFERS_MIN_MB")
            && let Ok(n) = v.trim().parse::<u64>()
        {
            (n << 20).max(4 << 10)
        } else {
            let kb = std::env::var("VICTORY_CARD_BUFFERS_MIN_KB")
                .ok()
                .and_then(|v| v.trim().parse::<u64>().ok())
                .unwrap_or(256);
            (kb << 10).max(4 << 10)
        };
        (min_bytes, mb("VICTORY_CARD_BUFFERS_MB", 2048))
    })
}

/// Bytes alive of copies' card sides, and of their "before" twins.
static CARD_BYTES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// The copies in the card's memory.
static CARDED: Mutex<Vec<std::sync::Weak<Shadow>>> = Mutex::new(Vec::new());

/// Since the last frame line: bytes copied to card sides and the ranges
/// they were; pieces compared on the card, pieces fetched and their
/// bytes; the waits for the card the fetches added and their microseconds.
pub(crate) static CARD_STATS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// [`CARD_STATS`], taken; then the copies in the card's memory, their
/// bytes and their "before" twins' bytes.
pub(crate) fn card_stats() -> [u64; 10] {
    let mut out = [0u64; 10];
    for (i, stat) in CARD_STATS.iter().enumerate() {
        out[i] = stat.swap(0, Ordering::Relaxed);
    }
    out[7] = CARDED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .filter(|w| w.strong_count() > 0)
        .count() as u64;
    out[8] = CARD_BYTES[0].load(Ordering::Relaxed) + WINDOW_CARD_BYTES.load(Ordering::Relaxed);
    out[9] = CARD_BYTES[1].load(Ordering::Relaxed);
    out
}

/// The bytes of the hashes kept on the card for a copy of `size` bytes: 16
/// for each 64 KiB piece (a copy that does not start on a piece boundary
/// has one piece more).
pub(crate) fn card_hash_bytes(size: u64) -> u64 {
    (size / PIECE as u64 + 2) * 16
}

/// Maximum share of VRAM budget (%) up to which card buffers may be allocated
/// (`VICTORY_CARD_VRAM_PERCENT`, default 92; clamped 50..=95). Kept higher
/// than [`crate::resource::room_percent`] so hot compute/skinning buffers keep
/// VRAM priority over streamed textures.
fn card_vram_percent() -> u64 {
    static PERCENT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *PERCENT.get_or_init(|| {
        std::env::var("VICTORY_CARD_VRAM_PERCENT")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map_or(92, |p| p.clamp(50, 95))
    })
}

/// Rejection reason in [`card_for`]: `[count, bytes]` pairs in [`CARD_REJECTS`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CardReject {
    Threshold = 0,
    DxgiBudget = 1,
    CumulativeCap = 2,
    CreateFailed = 3,
}

/// Cumulative count and bytes rejected by [`card_for`], indexed by `reason as usize * 2`.
static CARD_REJECTS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

fn note_card_reject(reason: CardReject, size: u64) {
    let idx = (reason as usize) * 2;
    CARD_REJECTS[idx].fetch_add(1, Ordering::Relaxed);
    CARD_REJECTS[idx + 1].fetch_add(size, Ordering::Relaxed);
}

/// Pure admission predicate for [`card_for`] before attempting resource creation.
fn check_card_admission(
    len: u64,
    size: u64,
    least: u64,
    most: u64,
    vram_percent: u64,
    video_memory: Option<[u64; 4]>,
    card_bytes_used: u64,
) -> Result<(), CardReject> {
    if len < least || size >= 1 << 30 {
        return Err(CardReject::Threshold);
    }
    if let Some([budget, used, ..]) = video_memory
        && used.saturating_add(size) > budget / 100 * vram_percent
    {
        return Err(CardReject::DxgiBudget);
    }
    if card_bytes_used.saturating_add(size) > most {
        return Err(CardReject::CumulativeCap);
    }
    Ok(())
}

/// A buffer of `size` bytes in the card's memory for a title buffer's
/// copy, if the copy is to be there: [`cards`] is on, the size is within
/// [`card_limits`] and below 1 GiB (the compare counts in 32 bits), and the
/// host has what it takes (`Host::card_ready`).
fn card_for(len: u64, size: u64) -> Option<ID3D12Resource> {
    if !cards() {
        return None;
    }
    let (least, most) = card_limits();
    if let Err(reason) = check_card_admission(len, size, least, most, card_vram_percent(), None, 0)
    {
        note_card_reject(reason, size);
        return None;
    }
    let host = host()?;
    if !host.card_ready() {
        return None;
    }
    // Only while the card has room under `card_vram_percent()` (default 92%,
    // above textures' cap so hot compute buffers like skinning stay in
    // local VRAM instead of spilling onto PCIe).
    if let Err(reason) = check_card_admission(
        len,
        size,
        least,
        most,
        card_vram_percent(),
        crate::accounting::video_memory_budget(host),
        CARD_BYTES[0].load(Ordering::Relaxed),
    ) {
        note_card_reject(reason, size);
        return None;
    }
    if CARD_BYTES[0]
        .fetch_add(size, Ordering::Relaxed)
        .saturating_add(size)
        > most
    {
        CARD_BYTES[0].fetch_sub(size, Ordering::Relaxed);
        note_card_reject(CardReject::CumulativeCap, size);
        return None;
    }
    let buffer = host.uav_buffer(size);
    if buffer.is_none() {
        CARD_BYTES[0].fetch_sub(size, Ordering::Relaxed);
        note_card_reject(CardReject::CreateFailed, size);
    }
    buffer
}

/// The most bytes of windows outside buffers whose copies go to the card's
/// memory (`VICTORY_CARD_WINDOWS_MB`; default 1536 MiB, `=0` disables).
///
/// Windows are two thirds of the PC memory the card shares (run 104: 861
/// windows, 3,552 MB of 5,400), and every draw that takes vertices, indices
/// or constants from the title's own allocator reads them over the bus.
/// Shaders write none of them (run 91: "0.0 MB compared in 0 ranges of
/// windows"), so a window on the card needs only what a card copy has
/// already: the CPU's side takes title memory as before, and what changed
/// in it goes up before each submission. No hashes, no compare, no fetch,
/// unless a shader does declare a write to it one day.
///
/// A window goes to the card while there is room under this number and
/// while VRAM usage stays below 72% of budget, leaving textures (75%) and
/// title buffers (`card_vram_percent`, 92%) their room. Tiny windows under
/// 64 KiB stay in shared memory to avoid cluttering `CARDED`.
fn window_card_most() -> u64 {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_CARD_WINDOWS_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(1536)
            << 20
    })
}

/// Bytes alive of windows' card sides ([`window_card_most`]).
static WINDOW_CARD_BYTES: AtomicU64 = AtomicU64::new(0);

/// A buffer in the card's memory for a window of `size` bytes, if it is to
/// be there ([`window_card_most`]).
fn window_card_for(size: u64) -> Option<ID3D12Resource> {
    let most = window_card_most();
    if most == 0 || !cards() || !((64 << 10)..(1 << 30)).contains(&size) {
        return None;
    }
    // Check cheap atomic ceiling before querying DXGI video memory budget (`fxceless` #1250).
    if WINDOW_CARD_BYTES
        .load(Ordering::Relaxed)
        .saturating_add(size)
        > most
    {
        return None;
    }
    let host = host()?;
    if !host.card_ready() {
        return None;
    }
    if let Some([budget, used, ..]) = crate::accounting::video_memory_budget(host)
        && used + size > budget / 100 * 72
    {
        return None;
    }
    if WINDOW_CARD_BYTES.fetch_add(size, Ordering::Relaxed) + size > most {
        WINDOW_CARD_BYTES.fetch_sub(size, Ordering::Relaxed);
        return None;
    }
    let buffer = host.uav_buffer(size);
    if buffer.is_none() {
        WINDOW_CARD_BYTES.fetch_sub(size, Ordering::Relaxed);
    }
    buffer
}

/// The copies in the card's memory whose CPU side changed since they were
/// last asked: each with the changed offsets, in order and joined where
/// they touch. The ranges are taken: the caller copies them to the card's
/// side (`Host::card_before`).
///
/// With [`relay`], `sent_as` is the number of the send that takes them
/// (`crate::host`'s order of the queue): each range is noted with it, in
/// the same lock that took it, so that a merge of pieces fetched before
/// that send knows the card did not hold these bytes yet
/// ([`Shadow::card_merge_relay`]). Notes of sends that every fetch still to
/// be merged came after are dropped.
pub(crate) fn card_dirty(sent_as: u64) -> Vec<(Arc<Shadow>, Vec<(usize, usize)>)> {
    let mut out = Vec::new();
    let mut carded = CARDED.lock().unwrap_or_else(|e| e.into_inner());
    let fetched = RELAY_FETCHED.load(Ordering::Acquire);
    carded.retain(|weak| {
        let Some(shadow) = weak.upgrade() else {
            return false;
        };
        let ranges = shadow
            .card
            .as_ref()
            .map(|c| {
                let mut dirty = c.dirty.lock().unwrap_or_else(|e| e.into_inner());
                let ranges = std::mem::take(&mut *dirty);
                if relay() {
                    let mut sent = c.sent.lock().unwrap_or_else(|e| e.into_inner());
                    sent.retain(|s| s.2 > fetched);
                    sent.extend(ranges.iter().map(|&(a, b)| (a, b, sent_as)));
                }
                ranges
            })
            .unwrap_or_default();
        if !ranges.is_empty() {
            out.push((shadow, joined(ranges)));
        }
        true
    });
    out
}

/// Ranges in order, those that overlap or touch as one.
fn joined(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    ranges.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for (from, to) in ranges {
        match out.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => out.push((from, to)),
        }
    }
    out
}

/// Runs of pieces to hash again (`Host::card_before`), those that overlap
/// or touch as one, in order: every piece of `runs` is in the result once
/// and no other piece is. Ranges sent up lie close together (514 a frame
/// for 87 MB in run 102), and each had a pass of its own over its pieces,
/// so a piece that held several ranges was hashed once for each.
#[cfg(test)]
pub(crate) fn joined_piece_runs(runs: Vec<CardSpan>) -> Vec<CardSpan> {
    joined_piece_runs_within(runs, 0)
}

/// Joins piece runs that touch or are separated by at most `max_gap` 64 KiB
/// pieces. Re-hashing a few unchanged pieces in parallel inside one compute
/// dispatch costs the same GPU wall-clock as hashing a single piece while
/// cutting hundreds of small `Dispatch` launches per frame.
pub(crate) fn joined_piece_runs_within(mut runs: Vec<CardSpan>, max_gap: u32) -> Vec<CardSpan> {
    runs.sort_unstable_by_key(|r| r.first);
    let mut out: Vec<CardSpan> = Vec::with_capacity(runs.len());
    for run in runs {
        match out.last_mut() {
            Some(last)
                if run.first
                    <= last
                        .first
                        .saturating_add(last.pieces)
                        .saturating_add(max_gap) =>
            {
                let end = (last.first + last.pieces).max(run.first + run.pieces);
                last.pieces = end - last.first;
                last.to = last.to.max(run.to);
            }
            _ => out.push(run),
        }
    }
    out
}

/// A declared range of a copy in the card's memory, as the compare on the
/// card takes it: its offsets in the copy, its first 64 KiB piece (pieces
/// are cut at multiples of 64 KiB of title memory; piece 0 holds offset
/// 0), how many pieces hold it, and where the copy starts in piece 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CardSpan {
    pub(crate) from: u32,
    pub(crate) to: u32,
    pub(crate) first: u32,
    pub(crate) pieces: u32,
    pub(crate) skew: u32,
}

/// [`CardSpan`] for offsets `[from, to)` (`from < to`) of a copy of title
/// memory from `start` on.
fn card_span_of(start: usize, from: usize, to: usize) -> CardSpan {
    let pieces = pieces_of(start, from, to);
    CardSpan {
        from: from as u32,
        to: to as u32,
        first: pieces.start as u32,
        pieces: pieces.len() as u32,
        skew: (start % PIECE) as u32,
    }
}

/// The offsets of piece `i` of `span` that lie in the span, for a copy of
/// `len` bytes of title memory from `start` on: what is fetched and merged
/// of a piece the compare found differing.
fn card_piece_of(start: usize, len: usize, span: &CardSpan, i: u32) -> (usize, usize) {
    let (lo, hi) = piece_at(start, len, (span.first + i) as usize);
    (lo.max(span.from as usize), hi.min(span.to as usize))
}

#[cfg(test)]
mod card_tests {
    use super::{CardSpan, PIECE, card_piece_of, card_span_of, joined, joined_piece_runs};

    /// Joined runs hold every piece of the runs given, once, and no other.
    #[test]
    fn piece_runs_join_without_losing_or_adding_pieces() {
        let run = |first: u32, pieces: u32| CardSpan {
            from: first * 65536,
            to: (first + pieces) * 65536,
            first,
            pieces,
            skew: 7,
        };
        let runs = vec![
            run(40, 1),
            run(3, 2),
            run(4, 1),
            run(5, 3),
            run(40, 1),
            run(9, 1),
            run(41, 2),
            run(100, 1),
        ];
        let mut wanted: Vec<u32> = runs
            .iter()
            .flat_map(|r| r.first..r.first + r.pieces)
            .collect();
        wanted.sort_unstable();
        wanted.dedup();
        let joined = joined_piece_runs(runs);
        let got: Vec<u32> = joined
            .iter()
            .flat_map(|r| r.first..r.first + r.pieces)
            .collect();
        assert_eq!(got, wanted);
        assert_eq!(
            joined
                .iter()
                .map(|r| (r.first, r.pieces))
                .collect::<Vec<_>>(),
            [(3, 5), (9, 1), (40, 3), (100, 1)]
        );
        assert!(joined.iter().all(|r| r.skew == 7));
        assert!(joined_piece_runs(Vec::new()).is_empty());
    }

    /// The offsets the compare's shader takes for a piece (`CARD_COMPARE`
    /// in host.rs), before it widens them to 16 bytes.
    fn shader_piece(span: &CardSpan, i: u32) -> (u32, u32) {
        let piece = span.first + i;
        let lo = piece * 65536;
        let lo = if lo > span.skew { lo - span.skew } else { 0 };
        let hi = (piece + 1) * 65536 - span.skew;
        (lo.max(span.from), hi.min(span.to))
    }

    /// The pieces of a span cover it without gaps, and the shader's
    /// arithmetic names the same bytes as ours.
    #[test]
    fn the_shader_and_the_merge_cut_the_same_pieces() {
        for start in [
            0usize,
            4096,
            PIECE - 16,
            PIECE,
            3 * PIECE + 0x1f00,
            0xd_b790_df00,
        ] {
            let len = 5 * PIECE + 4096;
            for (from, to) in [
                (0, len),
                (0, 1),
                (16, PIECE),
                (PIECE - 4, PIECE + 4),
                (2 * PIECE, len),
                (12345, 4 * PIECE + 7),
            ] {
                let span = card_span_of(start, from, to);
                let mut at = from;
                for i in 0..span.pieces {
                    let (lo, hi) = card_piece_of(start, len, &span, i);
                    assert_eq!(lo, at, "{start:#x} {from} {to} piece {i}");
                    assert!(hi > lo && hi - lo <= PIECE);
                    assert_eq!(
                        shader_piece(&span, i),
                        (lo as u32, hi as u32),
                        "{start:#x} {from} {to} piece {i}"
                    );
                    at = hi;
                }
                assert_eq!(at, to, "{start:#x} {from} {to}");
            }
        }
    }

    /// Changed ranges are joined where they overlap or touch, and only there.
    #[test]
    fn ranges_join_where_they_touch() {
        assert_eq!(
            joined(vec![(8, 12), (0, 4), (4, 6), (10, 20), (30, 31)]),
            [(0, 6), (8, 20), (30, 31)]
        );
        assert!(joined(Vec::new()).is_empty());
    }
}

/// Whether the copy-back of what shaders wrote runs on a thread of its own,
/// beside the submitting thread and outside the submission lock
/// (`VICTORY_RELAY=1` or `2`; off by default; not with kept snapshots,
/// [`keeps`]). `crate::host::relay` says what it does; here it changes
/// what a copy keeps:
///
/// - the ranges shaders may be writing are counted per submission
///   ([`Flights`]): a range two flights declared stays closed to refreshes
///   until both have been copied back;
/// - the merge of what the card wrote never stores into memory the card
///   may be writing ([`merge_piece`]): what it learns goes into title
///   memory and into the copy's reference (the snapshot, or the CPU's side
///   of a copy in the card's memory), under that reference's lock;
/// - what goes up to a copy in the card's memory is numbered
///   ([`card_dirty`]), so that a merge can tell the title's bytes the card
///   had not received from bytes the card wrote.
pub(crate) fn relay() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default (stage 2, `crate::host::relay_exact`); `VICTORY_RELAY=0`
    // turns it off, `=1` asks for the first stage.
    // The crate's own tests run without it unless they ask: several of
    // them look at title memory right after a submission, which the
    // relay's landing thread may still be writing.
    *ON.get_or_init(|| {
        std::env::var("VICTORY_RELAY").map_or(!cfg!(test), |v| v == "1" || v == "2") && !keeps()
    })
}

/// The submission being prepared (its number among those started): what
/// [`Shadow::writing`] counts a declared range for ([`Flights`]).
static PREPARED: AtomicU64 = AtomicU64::new(0);

/// A submission's preparing begins (`crate::host`, under the submission
/// lock).
pub(crate) fn preparing(number: u64) {
    PREPARED.store(number, Ordering::Relaxed);
}

/// With [`relay`]: the number of the last send of changed ranges to the
/// card (`crate::host`'s order of the queue) that every fetch still to be
/// merged came after. Notes of sends up to it are of no use any more.
static RELAY_FETCHED: AtomicU64 = AtomicU64::new(0);

/// A fetch sent after send `number` is the oldest one that may still be
/// merged ([`RELAY_FETCHED`]).
pub(crate) fn relay_fetched(number: u64) {
    RELAY_FETCHED.fetch_max(number, Ordering::AcqRel);
}

/// With [`relay`]: how many submissions that are not copied back yet
/// declared each range shaders may be writing (range, count, the last
/// submission that declared it).
///
/// Without it a range declared by two flights was one entry, and the end
/// of the first flight's copy-back removed it while the second was still
/// writing: the next refresh would have put title bytes over what the
/// second flight's shaders were writing.
#[derive(Default)]
struct Flights(Vec<((usize, usize), u32, u64)>);

impl Flights {
    /// Submission `number` declares `range`: counted once for it, however
    /// often it declares it.
    fn declare(&mut self, range: (usize, usize), number: u64) {
        match self.0.iter_mut().find(|f| f.0 == range) {
            Some(f) => {
                if f.2 != number {
                    f.1 += 1;
                    f.2 = number;
                }
            }
            None => self.0.push((range, 1, number)),
        }
    }

    /// One submission that declared `ranges` (each once) is copied back:
    /// the ranges no other submission still holds.
    fn done(&mut self, ranges: &[(usize, usize)]) -> Vec<(usize, usize)> {
        let mut free = Vec::new();
        for range in ranges {
            match self.0.iter().position(|f| f.0 == *range) {
                Some(at) => {
                    self.0[at].1 = self.0[at].1.saturating_sub(1);
                    if self.0[at].1 == 0 {
                        self.0.swap_remove(at);
                        free.push(*range);
                    }
                }
                // Not counted (declared before anything counted): free.
                None => free.push(*range),
            }
        }
        free
    }
}

/// The relay's merge of a stretch of a copy ([`relay`]). `now` is what the
/// card's memory holds, read once; `held` is the copy's reference for the
/// stretch (its snapshot, or the CPU's side of a copy in the card's
/// memory): what the card's memory held when it was last looked at;
/// `title` is title memory as just read. Every byte the card changed
/// (`now` differs from `held`) goes into `title` and into `held`; every
/// other byte of both stays as it is, so a byte the title wrote and the
/// card did not keeps the title's value. Whether `title` changed.
///
/// Nothing is stored into `now`'s source: the card may be writing it for a
/// later flight at this moment, and a store there, even of the value just
/// read, could undo that write. (The merge of a waited-for submission
/// stores every byte of a differing stretch back into the copy.)
fn merge_piece(now: &[u8], held: &mut [u8], title: &mut [u8]) -> bool {
    let mut differs = 0u8;
    for ((n, h), t) in now.iter().zip(held.iter_mut()).zip(title.iter_mut()) {
        let wrote = if *n != *h { 0xffu8 } else { 0 };
        let merged = (*t & !wrote) | (*n & wrote);
        differs |= merged ^ *t;
        *t = merged;
        *h = (*h & !wrote) | (*n & wrote);
    }
    differs != 0
}

/// `[from, to)` without the ranges of `skip`, in order.
fn outside(from: usize, to: usize, mut skip: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    skip.sort_unstable();
    let mut out = Vec::new();
    let mut at = from;
    for (a, b) in skip {
        if b <= at {
            continue;
        }
        if a >= to {
            break;
        }
        if a > at {
            out.push((at, a));
        }
        at = at.max(b);
    }
    if at < to {
        out.push((at, to));
    }
    out
}

#[cfg(test)]
mod relay_tests {
    use super::{Flights, changed_runs, merge_piece, outside};

    /// A range two flights declared is free only when both are done, in
    /// either order; a submission that declares it many times counts once.
    #[test]
    fn a_range_stays_closed_until_every_flight_that_declared_it_is_done() {
        let (a, b) = ((0x1000, 0x2000), (0x1800, 0x3000));
        let mut flights = Flights::default();
        for _ in 0..3 {
            flights.declare(a, 7);
        }
        flights.declare(b, 7);
        flights.declare(a, 8);
        // Flight 7 lands: `a` is still flight 8's, `b` was only flight 7's.
        assert_eq!(flights.done(&[a, b]), vec![b]);
        // Declared again meanwhile by a third.
        flights.declare(a, 9);
        assert_eq!(flights.done(&[a]), vec![]);
        assert_eq!(flights.done(&[a]), vec![a]);
        assert!(flights.0.is_empty());
        // The later flight first: the same.
        flights.declare(a, 1);
        flights.declare(a, 2);
        assert_eq!(flights.done(&[a]), vec![]);
        assert_eq!(flights.done(&[a]), vec![a]);
        // A range never counted is free at once.
        assert_eq!(flights.done(&[b]), vec![b]);
    }

    /// The merge moves exactly the bytes the card changed, into title
    /// memory and into the reference, and stores nothing else.
    #[test]
    fn the_merge_stores_only_what_the_card_changed() {
        let held0: Vec<u8> = (0..200u32).map(|i| (i * 5 + 1) as u8).collect();
        let mut now = held0.clone();
        for at in [0usize, 3, 4, 5, 64, 199] {
            now[at] ^= 0x33;
        }
        // The title wrote other bytes meanwhile, and one the card wrote too.
        let mut title = held0.clone();
        for at in [1usize, 5, 100] {
            title[at] = 0xee;
        }
        let title0 = title.clone();
        let mut held = held0.clone();
        assert!(merge_piece(&now, &mut held, &mut title));
        for at in 0..200 {
            if now[at] != held0[at] {
                assert_eq!(
                    (title[at], held[at]),
                    (now[at], now[at]),
                    "the card's byte at {at}"
                );
            } else {
                assert_eq!(
                    (title[at], held[at]),
                    (title0[at], held0[at]),
                    "untouched at {at}"
                );
            }
        }
        // Again with nothing new: nothing changes and nothing is reported.
        let (title1, held1) = (title.clone(), held.clone());
        assert!(!merge_piece(&now, &mut held, &mut title));
        assert_eq!((title, held), (title1, held1));
    }

    #[test]
    fn ranges_are_cut_out() {
        assert_eq!(outside(10, 50, vec![]), vec![(10, 50)]);
        assert_eq!(outside(10, 50, vec![(0, 10), (50, 60)]), vec![(10, 50)]);
        assert_eq!(
            outside(10, 50, vec![(30, 35), (0, 12), (34, 40), (48, 99)]),
            vec![(12, 30), (40, 48)]
        );
        assert_eq!(outside(10, 50, vec![(0, 99)]), vec![]);
    }

    /// A model of a copy in the PC's memory under the relay: the card
    /// writes `copy` directly; `snap` is the snapshot; `title` is title
    /// memory. Flight A and flight B declare overlapping ranges and both
    /// are in the air; the title's CPU writes in between and its bytes are
    /// put in beside the flights' (`merge_under_writes`: into the copy and
    /// the snapshot); A lands while B is still writing; then B lands.
    #[test]
    fn two_flights_overlapping_ranges_and_a_cpu_write_in_between() {
        const N: usize = 64;
        let base: Vec<u8> = (0..N as u32).map(|i| i as u8).collect();
        let (mut copy, mut title) = (base.clone(), base.clone());
        // A declares [0, 40), B declares [24, 64): each byte's snapshot is
        // taken when the first range that holds it is declared.
        let (a, b) = ((0usize, 40usize), (24usize, 64usize));
        let mut snap = base.clone();
        // The card runs A: writes 4..12 and 30..34.
        for at in (4..12).chain(30..34) {
            copy[at] = 0xa0;
        }
        // The title's CPU writes 8..10 (over A's bytes), 20 and 50, after
        // A's refresh: the next submission's preparing finds them against
        // the snapshot and puts them in, into the copy and the snapshot.
        for at in [8usize, 9, 20, 50] {
            title[at] = 0x77;
        }
        for (at, n) in changed_runs(&title, &snap) {
            copy[at..at + n].copy_from_slice(&title[at..at + n]);
            snap[at..at + n].copy_from_slice(&title[at..at + n]);
        }
        // The card starts B: it has written 32..36 so far, over A's 32..34.
        for at in 32..36 {
            copy[at] = 0xb0;
        }
        // A lands (B still on the card): its range is merged. The copy is
        // read once and not written.
        let seen = copy.clone();
        {
            let mut t = title[a.0..a.1].to_vec();
            assert!(merge_piece(&seen[a.0..a.1], &mut snap[a.0..a.1], &mut t));
            title[a.0..a.1].copy_from_slice(&t);
        }
        assert_eq!(copy, seen, "the merge stores nothing into the copy");
        // A's bytes are in title memory, except where the title wrote
        // after A's shaders did (8, 9: the title's); B's first bytes came
        // along; nothing of B's range beyond A's was touched.
        for at in (4..8).chain(10..12).chain(30..32) {
            assert_eq!(title[at], 0xa0, "A's byte at {at}");
        }
        assert_eq!(
            (title[8], title[9], title[20], title[50]),
            (0x77, 0x77, 0x77, 0x77)
        );
        for at in 32..36 {
            assert_eq!(title[at], 0xb0, "B's byte so far at {at}");
        }
        // The card finishes B: 36..44.
        for at in 36..44 {
            copy[at] = 0xb1;
        }
        // The title's CPU writes 40 after B's shader did: found against
        // the snapshot at the next preparing, which is before B lands.
        title[40] = 0x55;
        for (at, n) in changed_runs(&title[b.0..b.1], &snap[b.0..b.1]) {
            let at = at + b.0;
            copy[at..at + n].copy_from_slice(&title[at..at + n]);
            snap[at..at + n].copy_from_slice(&title[at..at + n]);
        }
        // B lands.
        let seen = copy.clone();
        {
            let mut t = title[b.0..b.1].to_vec();
            assert!(merge_piece(&seen[b.0..b.1], &mut snap[b.0..b.1], &mut t));
            title[b.0..b.1].copy_from_slice(&t);
        }
        // B's results are there, never replaced by A's older ones.
        for at in 32..36 {
            assert_eq!(title[at], 0xb0);
        }
        for at in (36..40).chain(41..44) {
            assert_eq!(title[at], 0xb1, "B's byte at {at}");
        }
        assert_eq!(title[40], 0x55, "the title wrote it last");
        // In the declared ranges the copy, the snapshot and title memory
        // now agree byte for byte: nothing is left to hand back, and
        // nothing was lost.
        assert_eq!(copy[a.0..b.1], title[a.0..b.1]);
        assert_eq!(snap[a.0..b.1], title[a.0..b.1]);
        // A third merge finds nothing.
        let mut t = title.clone();
        assert!(!merge_piece(&copy, &mut snap, &mut t));
    }

    /// A model of a copy in the card's memory under the relay: `card` is
    /// the card's side, `cpu` the CPU's side, `title` title memory. The
    /// title's changed bytes go into the CPU's side at once and up to the
    /// card with a later send; pieces fetched before that send must not
    /// take the card's older bytes there for something the card wrote.
    #[test]
    fn title_bytes_not_yet_on_the_card_are_not_taken_for_shader_writes() {
        const N: usize = 48;
        let base: Vec<u8> = (0..N as u32).map(|i| (i + 100) as u8).collect();
        let (mut card, mut cpu, mut title) = (base.clone(), base.clone(), base.clone());
        // Flight A's shaders write 10..14 on the card.
        for at in 10..14 {
            card[at] = 0xa0;
        }
        // The title's CPU writes 12 (over A's) and 30; the preparing of
        // the next submission puts both into the CPU's side and notes them
        // to go up (send 5). They are not on the card yet.
        title[12] = 0x77;
        title[30] = 0x78;
        let mut noted = Vec::new();
        for (at, n) in changed_runs(&title, &cpu) {
            cpu[at..at + n].copy_from_slice(&title[at..at + n]);
            noted.push((at, at + n, 5u64));
        }
        // A's pieces are fetched after send 4, before send 5: the fetch
        // holds A's bytes and the card's old bytes at 12 and 30.
        let fetched = card.clone();
        let fetched_after = 4u64;
        // Send 5 goes up behind it.
        for &(a, b, _) in &noted {
            card[a..b].copy_from_slice(&cpu[a..b]);
        }
        // The merge leaves out what went up after the fetch.
        let skip: Vec<(usize, usize)> = noted
            .iter()
            .filter(|s| s.2 > fetched_after)
            .map(|s| (s.0, s.1))
            .collect();
        let mut changed = false;
        for (a, b) in outside(0, N, skip) {
            let mut t = title[a..b].to_vec();
            changed |= merge_piece(&fetched[a..b], &mut cpu[a..b], &mut t);
            title[a..b].copy_from_slice(&t);
        }
        assert!(changed);
        // A's bytes arrived; the title's two writes stand, in title memory
        // and on both sides of the copy.
        for at in [10usize, 11, 13] {
            assert_eq!((title[at], cpu[at]), (0xa0, 0xa0));
        }
        assert_eq!((title[12], cpu[12], card[12]), (0x77, 0x77, 0x77));
        assert_eq!((title[30], cpu[30], card[30]), (0x78, 0x78, 0x78));
        assert_eq!(card, cpu);
        assert_eq!(card, title);
        // Without the leaving out, the card's old byte at 30 would have
        // been taken for a shader write and put over the title's.
        let mut wrong_title = title.clone();
        let mut wrong_cpu = cpu.clone();
        merge_piece(&fetched, &mut wrong_cpu, &mut wrong_title);
        assert_eq!(wrong_title[30], base[30]);
    }
}

/// Snapshot memory kept for reuse (already touched, so no page faults).
static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// An empty buffer with room for `n` bytes, from the pool if it has one
/// of about that size: a small copy's snapshot took the 174 MB buffer kept
/// for the big heap's, which then got another, until eight of them were
/// reserved (1.4 GB in run 24).
fn pooled(n: usize) -> Vec<u8> {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let most = n.saturating_mul(4).max(1 << 20);
    let fit = pool
        .iter()
        .enumerate()
        .filter(|(_, v)| v.capacity() >= n && v.capacity() <= most)
        .min_by_key(|(_, v)| v.capacity())
        .map(|(i, _)| i);
    let mut v = match fit {
        Some(i) => pool.swap_remove(i),
        None => Vec::with_capacity(n),
    };
    v.clear();
    v
}

/// Returns a snapshot buffer to the pool: at most 8 kept, the largest, and
/// of those the largest one and the next ones up to `KEPT` bytes in all.
fn give_back(v: Vec<u8>) {
    const KEPT: usize = 512 << 20;
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.push(v);
    pool.sort_by_key(|v| std::cmp::Reverse(v.capacity()));
    pool.truncate(8);
    let (mut seen, mut total) = (0, 0usize);
    pool.retain(|v| {
        seen += 1;
        total = total.saturating_add(v.capacity());
        seen == 1 || total <= KEPT
    });
}

const NOT_WATCHED: u8 = 2;
/// Physical-page view memory tracked by `watch::writes_guarded` (skips the
/// failing `GetWriteWatch` kernel syscall on every refresh).
const VIEW_GUARDED: u8 = 3;

/// Whether `VICTORY_REA_DIRTY_RANGES=1` enables AVX2 diffing, subsumption
/// pruning, and coalesced readback DMA ranges (OFF by default).
pub(crate) fn dirty_ranges_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("VICTORY_REA_DIRTY_RANGES")
            .ok()
            .is_some_and(|v| {
                let t = v.trim();
                t == "1" || t.eq_ignore_ascii_case("true") || t.eq_ignore_ascii_case("yes")
            });
        if on {
            tracing::info!(
                avx2 = avx2_supported(),
                "VICTORY_REA_DIRTY_RANGES=1: AVX2 dirty-range diffing and coalesced readbacks active"
            );
        }
        on
    })
}

#[inline]
fn avx2_supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static AVX2: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *AVX2.get_or_init(|| std::arch::is_x86_feature_detected!("avx2"))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

static DIRTY_DIFF_CALLS: AtomicU64 = AtomicU64::new(0);
static DIRTY_DIFF_BYTES: AtomicU64 = AtomicU64::new(0);
static DIRTY_AVX2_CALLS: AtomicU64 = AtomicU64::new(0);
static DIRTY_SUBSUMED_SKIPS: AtomicU64 = AtomicU64::new(0);
static COPY_REGIONS_RAW: AtomicU64 = AtomicU64::new(0);
static COPY_REGIONS_COALESCED: AtomicU64 = AtomicU64::new(0);

/// Telemetry snapshot for dirty range diffing and coalescing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DirtyRangeStats {
    pub enabled: bool,
    pub avx2: bool,
    pub diff_calls: u64,
    pub diff_bytes: u64,
    pub avx2_calls: u64,
    pub subsumed_skips: u64,
    pub copy_regions_raw: u64,
    pub copy_regions_coalesced: u64,
}

pub(crate) fn note_copy_region_coalescing(raw: usize, coalesced: usize) {
    COPY_REGIONS_RAW.fetch_add(raw as u64, Ordering::Relaxed);
    COPY_REGIONS_COALESCED.fetch_add(coalesced as u64, Ordering::Relaxed);
}

pub fn dirty_stats() -> DirtyRangeStats {
    DirtyRangeStats {
        enabled: dirty_ranges_enabled(),
        avx2: avx2_supported(),
        diff_calls: DIRTY_DIFF_CALLS.load(Ordering::Relaxed),
        diff_bytes: DIRTY_DIFF_BYTES.load(Ordering::Relaxed),
        avx2_calls: DIRTY_AVX2_CALLS.load(Ordering::Relaxed),
        subsumed_skips: DIRTY_SUBSUMED_SKIPS.load(Ordering::Relaxed),
        copy_regions_raw: COPY_REGIONS_RAW.load(Ordering::Relaxed),
        copy_regions_coalesced: COPY_REGIONS_COALESCED.load(Ordering::Relaxed),
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn slices_differ_avx2(a: &[u8], b: &[u8]) -> bool {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    let mut i = 0usize;
    while i + 128 <= n {
        // SAFETY: `i + 128 <= n`, unaligned loads allowed on AVX2.
        unsafe {
            let a0 = _mm256_loadu_si256(pa.add(i).cast());
            let b0 = _mm256_loadu_si256(pb.add(i).cast());
            let a1 = _mm256_loadu_si256(pa.add(i + 32).cast());
            let b1 = _mm256_loadu_si256(pb.add(i + 32).cast());
            let a2 = _mm256_loadu_si256(pa.add(i + 64).cast());
            let b2 = _mm256_loadu_si256(pb.add(i + 64).cast());
            let a3 = _mm256_loadu_si256(pa.add(i + 96).cast());
            let b3 = _mm256_loadu_si256(pb.add(i + 96).cast());
            let eq0 = _mm256_cmpeq_epi8(a0, b0);
            let eq1 = _mm256_cmpeq_epi8(a1, b1);
            let eq2 = _mm256_cmpeq_epi8(a2, b2);
            let eq3 = _mm256_cmpeq_epi8(a3, b3);
            let all = _mm256_and_si256(_mm256_and_si256(eq0, eq1), _mm256_and_si256(eq2, eq3));
            if _mm256_movemask_epi8(all) != -1 {
                return true;
            }
        }
        i += 128;
    }
    while i + 32 <= n {
        // SAFETY: `i + 32 <= n`.
        unsafe {
            let va = _mm256_loadu_si256(pa.add(i).cast());
            let vb = _mm256_loadu_si256(pb.add(i).cast());
            if _mm256_movemask_epi8(_mm256_cmpeq_epi8(va, vb)) != -1 {
                return true;
            }
        }
        i += 32;
    }
    a[i..n] != b[i..n]
}

#[inline]
fn slices_differ(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return true;
    }
    #[cfg(target_arch = "x86_64")]
    if dirty_ranges_enabled() && avx2_supported() && a.len() >= 32 {
        DIRTY_AVX2_CALLS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: `avx2` confirmed via runtime `is_x86_feature_detected!("avx2")`.
        return unsafe { slices_differ_avx2(a, b) };
    }
    a != b
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn first_difference_avx2(a: &[u8], b: &[u8]) -> Option<usize> {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    let mut i = 0usize;
    while i + 32 <= n {
        // SAFETY: `i + 32 <= n`.
        let eq_mask = unsafe {
            let va = _mm256_loadu_si256(pa.add(i).cast());
            let vb = _mm256_loadu_si256(pb.add(i).cast());
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(va, vb))
        };
        if eq_mask != -1 {
            let diff_mask = (!eq_mask) as u32;
            return Some(i + diff_mask.trailing_zeros() as usize);
        }
        i += 32;
    }
    a[i..n]
        .iter()
        .zip(&b[i..n])
        .position(|(x, y)| x != y)
        .map(|k| i + k)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn last_difference_avx2(a: &[u8], b: &[u8]) -> Option<usize> {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let tail = n & !31;
    if let Some(k) = a[tail..n]
        .iter()
        .zip(&b[tail..n])
        .rposition(|(x, y)| x != y)
    {
        return Some(tail + k);
    }
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    let mut i = tail;
    while i >= 32 {
        i -= 32;
        // SAFETY: `i + 32 <= n`.
        let eq_mask = unsafe {
            let va = _mm256_loadu_si256(pa.add(i).cast());
            let vb = _mm256_loadu_si256(pb.add(i).cast());
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(va, vb))
        };
        if eq_mask != -1 {
            let diff_mask = (!eq_mask) as u32;
            return Some(i + 31 - diff_mask.leading_zeros() as usize);
        }
    }
    None
}

fn first_difference_scalar(a: &[u8], b: &[u8]) -> Option<usize> {
    let whole = a.len().min(b.len()) / 8 * 8;
    let from = a[..whole]
        .as_chunks::<8>()
        .0
        .iter()
        .zip(b[..whole].chunks_exact(8))
        .position(|(x, y)| x != y)
        .map_or(whole, |word| word * 8);
    a[from..]
        .iter()
        .zip(&b[from..])
        .position(|(x, y)| x != y)
        .map(|i| from + i)
}

fn last_difference_scalar(a: &[u8], b: &[u8]) -> Option<usize> {
    let whole = a.len().min(b.len()) / 8 * 8;
    if let Some(i) = a[whole..]
        .iter()
        .zip(&b[whole..])
        .rposition(|(x, y)| x != y)
    {
        return Some(whole + i);
    }
    let word = a[..whole]
        .as_chunks::<8>()
        .0
        .iter()
        .zip(b[..whole].chunks_exact(8))
        .rposition(|(x, y)| x != y)?
        * 8;
    a[word..word + 8]
        .iter()
        .zip(&b[word..word + 8])
        .rposition(|(x, y)| x != y)
        .map(|i| word + i)
}

/// Where `a` and `b` (of one length) first differ, using AVX2 when
/// `VICTORY_REA_DIRTY_RANGES=1` and supported.
fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    if dirty_ranges_enabled() && avx2_supported() && a.len().min(b.len()) >= 32 {
        DIRTY_AVX2_CALLS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: AVX2 support verified at runtime.
        return unsafe { first_difference_avx2(a, b) };
    }
    first_difference_scalar(a, b)
}

/// Where `a` and `b` (of one length) last differ, using AVX2 when
/// `VICTORY_REA_DIRTY_RANGES=1` and supported.
fn last_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    if dirty_ranges_enabled() && avx2_supported() && a.len().min(b.len()) >= 32 {
        DIRTY_AVX2_CALLS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: AVX2 support verified at runtime.
        return unsafe { last_difference_avx2(a, b) };
    }
    last_difference_scalar(a, b)
}

#[cfg(test)]
mod difference_tests {
    use super::{first_difference, last_difference};

    /// The word-wise searches agree with the byte-wise ones they replace.
    #[test]
    fn differences_are_found_as_byte_by_byte() {
        let base: Vec<u8> = (0..61u32).map(|i| (i * 7 + 3) as u8).collect();
        let same = base.clone();
        assert_eq!(first_difference(&base, &same), None);
        assert_eq!(last_difference(&base, &same), None);
        for len in [0usize, 1, 7, 8, 9, 16, 23, 61] {
            for at in 0..len {
                for other in at..len {
                    let (a, mut b) = (&base[..len], base[..len].to_vec());
                    b[at] ^= 0x80;
                    b[other] ^= 0x01;
                    let first = a.iter().zip(&b).position(|(x, y)| x != y);
                    let last = a.iter().zip(&b).rposition(|(x, y)| x != y);
                    assert_eq!(first_difference(a, &b), first, "{len} {at} {other}");
                    assert_eq!(last_difference(a, &b), last, "{len} {at} {other}");
                }
            }
        }
    }
}

/// The unit shader writes are looked for and put back in.
const PIECE: usize = 64 << 10;

/// Copies `len` bytes, on several threads if they are many: a snapshot of
/// the world's big heap is 100-170 MB per submission, 2-4 GB a frame, and
/// one thread copies some 8 GB/s.
///
/// # Safety
/// Both ranges are valid for `len` bytes and do not overlap.
unsafe fn copy_wide(src: *const u8, dst: *mut u8, len: usize) {
    const PER_THREAD: usize = 16 << 20;
    let threads = (len / PER_THREAD).clamp(1, 6);
    if threads == 1 {
        // SAFETY: the caller's.
        unsafe { std::ptr::copy_nonoverlapping(src, dst, len) };
        return;
    }
    let (src, dst, each) = (src as usize, dst as usize, len.div_ceil(threads));
    std::thread::scope(|s| {
        for i in 0..threads {
            let at = (i * each).min(len);
            let n = each.min(len - at);
            s.spawn(move || {
                // SAFETY: a part of the caller's ranges.
                unsafe {
                    std::ptr::copy_nonoverlapping((src + at) as *const u8, (dst + at) as *mut u8, n)
                };
            });
        }
    });
}

/// Whether the snapshots of a submission's shader-written ranges, and the
/// compare that finds what the shaders wrote, run on several threads
/// across ranges (`VICTORY_WIDE_BACK=1`; off by default).
///
/// Both ran range by range on the submitting thread, a range on several
/// threads only if it was 16 MiB or more. The ranges left on the CPU's side
/// since the big buffers went to the card are small and many: in run 86,
/// 7,024 declared ranges a frame, 375 MB of them compared after the card
/// ran in 563 pieces of work, none of 16 MiB: 51 ms a frame of snapshots
/// and 35 ms of comparing, one thread, bound by one core's memory speed.
///
/// With the switch a submission's snapshots are noted as it is prepared
/// and taken together when its refreshes are done, just before it goes to
/// the card ([`take_snapshots`]), and the compares of everything declared
/// run together after the card ([`find_all`]), both cut into pieces of
/// work of about 1 MiB for a few threads that are kept ([`wide`]). The
/// bytes taken and the bytes compared are the same ones: a range's bytes
/// cannot change between the moment it is declared and the end of the
/// preparing (a declared range is left alone by every refresh, the title's
/// own copies write title memory and not the copy, and the card runs the
/// lists only afterwards).
pub(crate) fn wide_back() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default; `VICTORY_WIDE_BACK=0` turns it off.
    *ON.get_or_init(|| std::env::var_os("VICTORY_WIDE_BACK").is_none_or(|v| v != "0"))
}

type Job = Box<dyn FnOnce() + Send>;

/// Threads kept for [`wide`]: what they are to do, and the word that there
/// is something.
struct Workers {
    jobs: Mutex<std::collections::VecDeque<Job>>,
    more: std::sync::Condvar,
}

fn workers() -> &'static Workers {
    static WORKERS: std::sync::OnceLock<Workers> = std::sync::OnceLock::new();
    static STARTED: std::sync::Once = std::sync::Once::new();
    let workers = WORKERS.get_or_init(|| Workers {
        jobs: Mutex::new(std::collections::VecDeque::new()),
        more: std::sync::Condvar::new(),
    });
    STARTED.call_once(|| {
        let baseline = std::thread::available_parallelism()
            .map_or(4, |n| n.get() / 4)
            .clamp(2, 6);
        let n = victory_core::scheduler::shadow_workers(baseline);
        for i in 0..n {
            let _ = std::thread::Builder::new()
                .name(format!("victory-wide-{i}"))
                .spawn(move || {
                    victory_core::scheduler::apply_current_thread_role(
                        victory_core::scheduler::ThreadRole::ShadowWorker,
                        i,
                    );
                    let workers = WORKERS.get().expect("made above");
                    loop {
                        let job = {
                            let mut jobs = workers.jobs.lock().unwrap_or_else(|e| e.into_inner());
                            loop {
                                if let Some(job) = jobs.pop_front() {
                                    break job;
                                }
                                jobs = workers.more.wait(jobs).unwrap_or_else(|e| e.into_inner());
                            }
                        };
                        job();
                    }
                });
        }
    });
    workers
}

/// Runs `jobs` on the kept threads and on this one, and returns when all
/// of them are done. The jobs copy and compare memory; they do not panic.
fn wide(jobs: Vec<Job>) {
    if jobs.len() < 2 {
        for job in jobs {
            job();
        }
        return;
    }
    let workers = workers();
    let left = Arc::new((Mutex::new(jobs.len()), std::sync::Condvar::new()));
    {
        let mut queue = workers.jobs.lock().unwrap_or_else(|e| e.into_inner());
        for job in jobs {
            let left = left.clone();
            queue.push_back(Box::new(move || {
                job();
                let mut n = left.0.lock().unwrap_or_else(|e| e.into_inner());
                *n -= 1;
                if *n == 0 {
                    left.1.notify_all();
                }
            }));
        }
    }
    workers.more.notify_all();
    loop {
        let job = workers
            .jobs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop_front();
        match job {
            Some(job) => job(),
            None => break,
        }
    }
    let mut n = left.0.lock().unwrap_or_else(|e| e.into_inner());
    while *n > 0 {
        n = left.1.wait(n).unwrap_or_else(|e| e.into_inner());
    }
}

/// [`wide`], for the host's own work on host copies (a submission's
/// refreshes, the merge of what shaders wrote to card copies).
pub(crate) fn wide_run(jobs: Vec<Box<dyn FnOnce() + Send>>) {
    wide(jobs);
}

/// About this many bytes to a piece of work for [`wide`].
const WIDE_JOB: usize = 1 << 20;

/// With [`wide_back`]: the snapshots noted while a submission is prepared
/// (from, to, bytes), taken by [`take_snapshots`].
static SNAPS: Mutex<Vec<(usize, usize, usize)>> = Mutex::new(Vec::new());

/// With [`wide_back`], when a submission's refreshes are done and before
/// its lists go to the card: takes the snapshots [`Shadow::writing`] noted.
/// Nothing to do without the switch.
pub(crate) fn take_snapshots() {
    let snaps = std::mem::take(&mut *SNAPS.lock().unwrap_or_else(|e| e.into_inner()));
    if snaps.is_empty() {
        return;
    }
    let began = std::time::Instant::now();
    wide(copy_jobs(snaps));
    BACK_TIMING_US[0].fetch_add(began.elapsed().as_micros() as u64, Ordering::Relaxed);
}

/// `copies` (from, to, bytes) as pieces of work of about [`WIDE_JOB`]
/// bytes each: long ones cut, short ones together.
fn copy_jobs(copies: Vec<(usize, usize, usize)>) -> Vec<Job> {
    let mut jobs: Vec<Job> = Vec::new();
    let mut group: Vec<(usize, usize, usize)> = Vec::new();
    let mut bytes = 0;
    let flush = |group: &mut Vec<(usize, usize, usize)>, jobs: &mut Vec<Job>| {
        let group = std::mem::take(group);
        if !group.is_empty() {
            jobs.push(Box::new(move || {
                for (src, dst, len) in group {
                    // SAFETY: the caller's ranges: a host copy's mapping and
                    // its snapshot, which do not overlap.
                    unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, len) };
                }
            }));
        }
    };
    for (src, dst, len) in copies {
        let mut at = 0;
        while at < len {
            let n = (WIDE_JOB - bytes).min(len - at);
            group.push((src + at, dst + at, n));
            at += n;
            bytes += n;
            if bytes >= WIDE_JOB {
                flush(&mut group, &mut jobs);
                bytes = 0;
            }
        }
    }
    flush(&mut group, &mut jobs);
    jobs
}

/// With [`wide_back`]: for each of `work` (a copy not in the card's memory
/// and a range `[address, end)` its shaders may have written, declared and
/// not yet done), the 64 KiB pieces that differ from the snapshot, as
/// [`Shadow::copy_back_with`] takes them; found on several threads.
pub(crate) fn find_all(work: &[(Arc<Shadow>, usize, usize)]) -> Vec<Vec<(usize, usize)>> {
    let began = std::time::Instant::now();
    let scans = work
        .iter()
        .map(|(shadow, address, end)| {
            let (from, to) = shadow.inside(*address, end.saturating_sub(*address));
            // The snapshot's memory stays until the range is done with
            // (`copy_back_done`), which comes after this.
            match shadow
                .before
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_ref()
            {
                Some(before) => (
                    shadow.data as usize,
                    before.as_ptr() as usize,
                    shadow.start as usize,
                    from,
                    to,
                ),
                None => (0, 0, 0, 0, 0),
            }
        })
        .collect();
    let found = find_pieces(scans);
    BACK_TIMING_US[1].fetch_add(began.elapsed().as_micros() as u64, Ordering::Relaxed);
    found
}

/// For each of `scans` (where a copy's bytes are, where its snapshot's
/// are, the title address the copy starts at, offsets from and to): the
/// 64 KiB pieces (cut at multiples of 64 KiB of title memory, as
/// [`Shadow::differing`] cuts them) of `[from, to)` whose bytes differ, as
/// (offset, bytes) in order. On several threads ([`wide`]), in pieces of
/// work of about [`WIDE_JOB`] bytes that hold whole pieces.
fn find_pieces(scans: Vec<(usize, usize, usize, usize, usize)>) -> Vec<Vec<(usize, usize)>> {
    // What a job finds: (which scan, its place among that scan's jobs, the
    // pieces).
    type Found = Vec<(usize, usize, Vec<(usize, usize)>)>;
    // A job's part of a scan: the scan, then which and place.
    type Part = (usize, usize, usize, usize, usize, usize, usize);
    let found: Arc<Mutex<Found>> = Arc::default();
    let mut jobs: Vec<Job> = Vec::new();
    let mut group: Vec<Part> = Vec::new();
    let mut bytes = 0;
    let flush = |group: &mut Vec<Part>, jobs: &mut Vec<Job>| {
        let group = std::mem::take(group);
        if group.is_empty() {
            return;
        }
        let found = found.clone();
        jobs.push(Box::new(move || {
            let mut mine: Found = Vec::with_capacity(group.len());
            let mut local_calls = 0u64;
            let mut local_bytes = 0u64;
            for (data, before, start, from, to, which, place) in group {
                let mut pieces = Vec::new();
                let mut at = from;
                while at < to {
                    let n = (PIECE - (start + at) % PIECE).min(to - at);
                    // SAFETY: inside the copy's mapping and its snapshot
                    // (which holds the copy's bytes at the copy's offsets);
                    // nothing writes either while a submission's results
                    // are compared.
                    let (now, old) = unsafe {
                        (
                            std::slice::from_raw_parts((data + at) as *const u8, n),
                            std::slice::from_raw_parts((before + at) as *const u8, n),
                        )
                    };
                    local_calls += 1;
                    local_bytes += n as u64;
                    if slices_differ(now, old) {
                        pieces.push((at, n));
                    }
                    at += n;
                }
                mine.push((which, place, pieces));
            }
            if local_calls > 0 && dirty_ranges_enabled() {
                DIRTY_DIFF_CALLS.fetch_add(local_calls, Ordering::Relaxed);
                DIRTY_DIFF_BYTES.fetch_add(local_bytes, Ordering::Relaxed);
            }
            found.lock().unwrap_or_else(|e| e.into_inner()).extend(mine);
        }));
    };
    let count = scans.len();
    for (which, (data, before, start, from, to)) in scans.into_iter().enumerate() {
        let (mut at, mut place) = (from, 0);
        while at < to {
            // Cut where pieces end, so that every job sees whole pieces.
            let room = (WIDE_JOB - bytes).max(PIECE);
            let mut next = (at + room).min(to);
            if next < to {
                next -= (start + next) % PIECE;
                if next <= at {
                    next = (at + PIECE - (start + at) % PIECE).min(to);
                }
            }
            group.push((data, before, start, at, next, which, place));
            bytes += next - at;
            place += 1;
            at = next;
            if bytes >= WIDE_JOB {
                flush(&mut group, &mut jobs);
                bytes = 0;
            }
        }
    }
    flush(&mut group, &mut jobs);
    wide(jobs);
    let mut found = std::mem::take(&mut *found.lock().unwrap_or_else(|e| e.into_inner()));
    found.sort_unstable_by_key(|f| (f.0, f.1));
    let mut out = vec![Vec::new(); count];
    for (which, _, pieces) in found {
        out[which].extend(pieces);
    }
    out
}

/// Since the last frame-rate line, of the shader-written ranges looked for
/// on the CPU (copies not in the card's memory): bytes snapshotted in
/// windows outside buffers and in title buffers' copies; bytes compared in
/// windows, in buffers' copies, and of those in copies of 1 MiB or more;
/// ranges compared in windows and in buffers' copies.
static BACK_CLASS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// Live shadow bytes by class: `0` card buffers, `1` CPU buffers `>= least`,
/// `2` CPU buffers `< least`, `3` card windows, `4` CPU windows.
static LIVE_CLASS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// Effective CPU microseconds per interval in snapshot (`0`), compare (`1`),
/// and merge (`2`).
static BACK_TIMING_US: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Snapshot of [`crate::resource::KINDS`] from the previous telemetry line.
static LAST_TEX_KINDS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

fn card_telemetry_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_TELEMETRY").is_none_or(|v| v != "0"))
}

/// P0 VRAM, `card_for` rejection, shadow-class, texture-churn, and
/// copy-back phase telemetry (`VICTORY_CARD_TELEMETRY=0` disables).
pub(crate) fn card_telemetry_line(frames: f64) {
    if !card_telemetry_enabled() || frames <= 0.0 {
        return;
    }
    let mb = |b: u64| b >> 20;
    let r = [0, 1, 2, 3, 4, 5, 6, 7].map(|i| CARD_REJECTS[i].load(Ordering::Relaxed));
    let cls = [0, 1, 2, 3, 4].map(|i| LIVE_CLASS[i].load(Ordering::Relaxed));
    let [snap_ms, cmp_ms, merge_ms] =
        [0, 1, 2].map(|i| BACK_TIMING_US[i].swap(0, Ordering::Relaxed) as f64 / 1e3 / frames);
    let vm = host().and_then(crate::accounting::video_memory_budget);
    let [lb_mb, lu_mb, nlb_mb, nlu_mb] = vm.map_or([0; 4], |v| v.map(mb));
    let [
        tex_alive,
        tex_alive_bytes,
        tex_let_go,
        tex_let_go_bytes,
        tex_remade,
        tex_remade_bytes,
        tex_made,
    ] = crate::resource::KINDS
        .each_ref()
        .map(|k| k.load(Ordering::Relaxed));
    let d_let_go = tex_let_go.saturating_sub(LAST_TEX_KINDS[0].swap(tex_let_go, Ordering::Relaxed));
    let d_let_go_b = tex_let_go_bytes
        .saturating_sub(LAST_TEX_KINDS[1].swap(tex_let_go_bytes, Ordering::Relaxed));
    let d_remade = tex_remade.saturating_sub(LAST_TEX_KINDS[2].swap(tex_remade, Ordering::Relaxed));
    let d_remade_b = tex_remade_bytes
        .saturating_sub(LAST_TEX_KINDS[3].swap(tex_remade_bytes, Ordering::Relaxed));
    let d_made = tex_made.saturating_sub(LAST_TEX_KINDS[4].swap(tex_made, Ordering::Relaxed));
    tracing::info!(
        "card telemetry (P0): card_for rejected: threshold={} ({} MB), dxgi_budget={} ({} MB), cap={} ({} MB), create={} ({} MB); \
         live shadows: card_buf={} MB, cpu_buf_ge_min={} MB, cpu_buf_lt_min={} MB, card_win={} MB, cpu_win={} MB; \
         vram local={}/{} MB, non_local={}/{} MB; tex churn: alive={} ({} MB), +made={}, +let_go={} ({} MB), +remade={} ({} MB); \
         cpu_back per frame: snapshot={snap_ms:.2} ms, compare={cmp_ms:.2} ms, merge={merge_ms:.2} ms",
        r[0],
        mb(r[1]),
        r[2],
        mb(r[3]),
        r[4],
        mb(r[5]),
        r[6],
        mb(r[7]),
        mb(cls[0]),
        mb(cls[1]),
        mb(cls[2]),
        mb(cls[3]),
        mb(cls[4]),
        lu_mb,
        lb_mb,
        nlu_mb,
        nlb_mb,
        tex_alive,
        mb(tex_alive_bytes),
        d_made,
        d_let_go,
        mb(d_let_go_b),
        d_remade,
        mb(d_remade_b),
    );
}

/// For the frame-rate line: where the CPU's search for shader writes goes
/// (per frame). A copy of a title buffer can go to the card's memory,
/// where the card does the search (`VICTORY_CARD_BUFFERS_MIN_MB`); a
/// window outside buffers cannot as the code stands.
pub(crate) fn back_line(frames: f64) {
    let c: Vec<f64> = BACK_CLASS
        .iter()
        .map(|c| c.swap(0, Ordering::Relaxed) as f64 / frames)
        .collect();
    if c.iter().all(|&v| v == 0.0) {
        return;
    }
    let mb = |v: f64| v / (1 << 20) as f64;
    tracing::info!(
        "shader writes looked for on the CPU per frame: {:.1} MB compared in {:.0} ranges of windows outside buffers ({:.1} MB \
         snapshotted), {:.1} MB in {:.0} ranges of title buffers' copies ({:.1} MB snapshotted), of them {:.1} MB in copies \
         of 1 MiB or more (VICTORY_CARD_BUFFERS_MIN_MB=1 would have the card look there)",
        mb(c[2]),
        c[5],
        mb(c[0]),
        mb(c[3]),
        c[6],
        mb(c[1]),
        mb(c[4]),
    );
}

#[cfg(test)]
mod wide_tests {
    use super::{Job, PIECE, WIDE_JOB, copy_jobs, find_pieces, wide};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Every job runs once, however many there are, and `wide` returns
    /// only when all have.
    #[test]
    fn all_jobs_run_before_wide_returns() {
        for n in [0usize, 1, 2, 7, 300] {
            let ran = Arc::new(AtomicUsize::new(0));
            let jobs: Vec<Job> = (0..n)
                .map(|i| {
                    let ran = ran.clone();
                    Box::new(move || {
                        ran.fetch_add(i + 1, Ordering::Relaxed);
                    }) as Job
                })
                .collect();
            wide(jobs);
            assert_eq!(ran.load(Ordering::Relaxed), n * (n + 1) / 2, "{n} jobs");
        }
    }

    /// Copies cut into jobs copy every byte once: long ones, short ones
    /// and empty ones.
    #[test]
    fn copies_cut_into_jobs_copy_the_same_bytes() {
        let len = 3 * WIDE_JOB + 12345;
        let src: Vec<u8> = (0..len).map(|i| (i * 31 + i / 977) as u8).collect();
        let mut dst = vec![0u8; len];
        let (s, d) = (src.as_ptr() as usize, dst.as_mut_ptr() as usize);
        // Ranges of the buffer, each to the same offsets of the other.
        let cuts = [
            0,
            1,
            1,
            4096,
            70_000,
            WIDE_JOB + 3,
            2 * WIDE_JOB + 3,
            2 * WIDE_JOB + 9,
            len,
        ];
        let copies: Vec<(usize, usize, usize)> = cuts
            .windows(2)
            .map(|w| (s + w[0], d + w[0], w[1] - w[0]))
            .collect();
        let jobs = copy_jobs(copies);
        assert!(jobs.len() >= 3);
        wide(jobs);
        assert!(src == dst);
    }

    /// The pieces found on several threads are the ones a walk piece by
    /// piece finds, for copies that start anywhere in a 64 KiB piece of
    /// title memory and ranges that start and end anywhere.
    #[test]
    fn pieces_found_wide_are_those_found_one_by_one() {
        let len = 5 * WIDE_JOB / 2 + 777;
        let old: Vec<u8> = (0..len).map(|i| (i * 7 + i / 513) as u8).collect();
        let mut now = old.clone();
        // Changes: single bytes at piece edges, a run across two pieces,
        // the last byte, and nothing in between.
        for at in [
            0usize,
            1,
            PIECE - 1,
            PIECE,
            3 * PIECE + 5,
            WIDE_JOB - 1,
            WIDE_JOB,
            WIDE_JOB + PIECE / 2,
            2 * WIDE_JOB + 9,
            len - 1,
        ] {
            now[at] ^= 0x5a;
        }
        for byte in &mut now[9 * PIECE - 40..9 * PIECE + 40] {
            *byte ^= 0x01;
        }
        let (data, before) = (now.as_ptr() as usize, old.as_ptr() as usize);
        let mut scans = Vec::new();
        let mut wanted = Vec::new();
        for start in [0usize, 4096, PIECE - 16, 0xd_b790_df00] {
            for (from, to) in [
                (0, len),
                (0, 1),
                (1, 2),
                (PIECE - 1, PIECE + 1),
                (3 * PIECE, 3 * PIECE),
                (12345, len - 1),
                (WIDE_JOB - 3, 2 * WIDE_JOB + 10),
            ] {
                scans.push((data, before, start, from, to));
                let mut pieces = Vec::new();
                let mut at = from;
                while at < to {
                    let n = (PIECE - (start + at) % PIECE).min(to - at);
                    if now[at..at + n] != old[at..at + n] {
                        pieces.push((at, n));
                    }
                    at += n;
                }
                wanted.push(pieces);
            }
        }
        // A scan of nothing (a range whose snapshot is gone).
        scans.push((0, 0, 0, 0, 0));
        wanted.push(Vec::new());
        assert_eq!(find_pieces(scans), wanted);
    }
}

/// Since the last frame-rate line, of [`Shadow::merge_under_writes`]: pages
/// compared, changed runs put in, and their bytes.
static MERGED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// [`MERGED`], taken.
pub(crate) fn merged_stats() -> [u64; 3] {
    [0, 1, 2].map(|i| MERGED[i].swap(0, Ordering::Relaxed))
}

fn changed_runs_scalar(now: &[u8], was: &[u8]) -> Vec<(usize, usize)> {
    let n = now.len().min(was.len());
    let mut runs = Vec::new();
    let mut i = 0;
    while i < n {
        // Equal bytes, eight at a time while they last.
        while i + 8 <= n && now[i..i + 8] == was[i..i + 8] {
            i += 8;
        }
        while i < n && now[i] == was[i] {
            i += 1;
        }
        if i >= n {
            break;
        }
        let first = i;
        while i < n && now[i] != was[i] {
            i += 1;
        }
        runs.push((first, i - first));
    }
    runs
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn changed_runs_avx2(now: &[u8], was: &[u8]) -> Vec<(usize, usize)> {
    use std::arch::x86_64::*;
    let n = now.len().min(was.len());
    let pn = now.as_ptr();
    let pw = was.as_ptr();
    let mut runs = Vec::new();
    let mut i = 0usize;
    while i < n {
        while i + 32 <= n {
            // SAFETY: `i + 32 <= n`.
            let eq_mask = unsafe {
                let a = _mm256_loadu_si256(pn.add(i).cast());
                let b = _mm256_loadu_si256(pw.add(i).cast());
                _mm256_movemask_epi8(_mm256_cmpeq_epi8(a, b))
            };
            if eq_mask != -1 {
                i += (!eq_mask as u32).trailing_zeros() as usize;
                break;
            }
            i += 32;
        }
        while i < n && now[i] == was[i] {
            i += 1;
        }
        if i >= n {
            break;
        }
        let first = i;
        while i < n && now[i] != was[i] {
            i += 1;
        }
        runs.push((first, i - first));
    }
    runs
}

/// The runs of bytes where `now` differs from `was` (of one length), as
/// (offset, bytes) in order: every differing byte is in a run and no equal
/// byte is.
fn changed_runs(now: &[u8], was: &[u8]) -> Vec<(usize, usize)> {
    #[cfg(target_arch = "x86_64")]
    if dirty_ranges_enabled() && avx2_supported() && now.len().min(was.len()) >= 32 {
        DIRTY_AVX2_CALLS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: AVX2 support verified at runtime.
        return unsafe { changed_runs_avx2(now, was) };
    }
    changed_runs_scalar(now, was)
}

#[cfg(test)]
mod merge_tests {
    use super::changed_runs;

    /// The runs hold exactly the bytes that differ: putting them into the
    /// old bytes gives the new ones, and no run holds an equal byte.
    #[test]
    fn changed_runs_are_exactly_the_bytes_that_differ() {
        let was: Vec<u8> = (0..300u32).map(|i| (i * 13 + 5) as u8).collect();
        for changes in [
            vec![],
            vec![0],
            vec![299],
            vec![7, 8, 9],
            vec![7, 9],
            vec![
                15, 16, 17, 23, 24, 64, 65, 66, 67, 68, 69, 70, 71, 72, 298, 299,
            ],
        ] {
            let mut now = was.clone();
            for &at in &changes {
                now[at] ^= 0x41;
            }
            let runs = changed_runs(&now, &was);
            let mut put = was.clone();
            let mut held = Vec::new();
            for &(at, n) in &runs {
                assert!(n > 0);
                put[at..at + n].copy_from_slice(&now[at..at + n]);
                held.extend(at..at + n);
            }
            assert_eq!(put, now, "{changes:?}");
            assert_eq!(held, changes, "{changes:?}");
            // Runs do not touch: two that did would be one.
            assert_eq!(runs, runs_scalar_check(&now, &was));
        }
        assert!(changed_runs(&[], &[]).is_empty());
    }

    fn runs_scalar_check(now: &[u8], was: &[u8]) -> Vec<(usize, usize)> {
        super::changed_runs_scalar(now, was)
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_diff_and_changed_runs_match_scalar_exactly() {
        if !super::avx2_supported() {
            return;
        }
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for len in (0..=132usize).chain([255, 256, 257, 1024, 1025, 4096, 65536]) {
            for align_off in [0usize, 1, 7] {
                let mut base = vec![0u8; len + 16];
                for b in &mut base {
                    *b = next() as u8;
                }
                let mut modv = base.clone();
                // Flip isolated bytes, multi-byte runs, and boundaries.
                if len > 0 {
                    for _ in 0..8 {
                        let idx = (next() as usize % len) + align_off;
                        modv[idx] ^= ((next() & 0x7F) as u8) | 1;
                    }
                    if len >= 40 {
                        let mid = align_off + len / 2;
                        for b in &mut modv[mid - 3..mid + 19] {
                            *b ^= 0x5A;
                        }
                    }
                }
                let a = &base[align_off..align_off + len];
                let b = &modv[align_off..align_off + len];
                // SAFETY: `avx2_supported()` checked above.
                unsafe {
                    assert_eq!(
                        super::slices_differ_avx2(a, b),
                        a != b,
                        "slices_differ len={len} off={align_off}"
                    );
                    assert!(
                        !super::slices_differ_avx2(a, a),
                        "slices_differ eq len={len}"
                    );
                    assert_eq!(
                        super::first_difference_avx2(a, b),
                        super::first_difference_scalar(a, b),
                        "first_difference len={len} off={align_off}"
                    );
                    assert_eq!(
                        super::last_difference_avx2(a, b),
                        super::last_difference_scalar(a, b),
                        "last_difference len={len} off={align_off}"
                    );
                    assert_eq!(
                        super::changed_runs_avx2(b, a),
                        super::changed_runs_scalar(b, a),
                        "changed_runs len={len} off={align_off}"
                    );
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore = "rea002 microbenchmark"]
    fn rea002_bench_avx2_vs_scalar_diff() {
        const SIZE: usize = 16 << 20;
        const ITERS: usize = 16;
        let a = vec![0x3Cu8; SIZE];
        let mut b = a.clone();
        // Inject 1% dirty spans near tail/middle/start to exercise changed_runs.
        for chunk in (0..SIZE).step_by(4096) {
            if (chunk / 4096).is_multiple_of(10) {
                b[chunk..chunk + 64].fill(0xA5);
            }
        }
        // 1) Equal-buffer scan (worst case for early-exit slices_differ).
        let t0 = std::time::Instant::now();
        for _ in 0..ITERS {
            assert!(!(std::hint::black_box(&a) != std::hint::black_box(&a)));
        }
        let scalar_eq = t0.elapsed();

        let avx2 = super::avx2_supported();
        let t1 = std::time::Instant::now();
        if avx2 {
            for _ in 0..ITERS {
                assert!(unsafe {
                    !super::slices_differ_avx2(std::hint::black_box(&a), std::hint::black_box(&a))
                });
            }
        }
        let avx2_eq = t1.elapsed();

        // 2) changed_runs over 16 MiB with sparse dirty ranges.
        let t2 = std::time::Instant::now();
        let mut total_runs = 0usize;
        for _ in 0..4 {
            total_runs += std::hint::black_box(super::changed_runs_scalar(&b, &a)).len();
        }
        let scalar_runs = t2.elapsed();

        let t3 = std::time::Instant::now();
        let mut total_runs_avx = 0usize;
        if avx2 {
            for _ in 0..4 {
                total_runs_avx +=
                    std::hint::black_box(unsafe { super::changed_runs_avx2(&b, &a) }).len();
            }
        }
        let avx2_runs = t3.elapsed();
        assert_eq!(total_runs, total_runs_avx);

        let total_gb_eq = (SIZE * ITERS) as f64 / 1e9;
        let total_gb_runs = (SIZE * 4) as f64 / 1e9;
        println!(
            "REA002_BENCH_AVX2 avx2={avx2} eq_scalar_gbps={:.2} eq_avx2_gbps={:.2} eq_speedup={:.2}x \
             runs_scalar_ms={:.2} runs_avx2_ms={:.2} runs_scalar_gbps={:.2} runs_avx2_gbps={:.2} runs_speedup={:.2}x runs={total_runs}",
            total_gb_eq / scalar_eq.as_secs_f64(),
            total_gb_eq / avx2_eq.as_secs_f64(),
            scalar_eq.as_secs_f64() / avx2_eq.as_secs_f64(),
            scalar_runs.as_secs_f64() * 1e3 / 4.0,
            avx2_runs.as_secs_f64() * 1e3 / 4.0,
            total_gb_runs / scalar_runs.as_secs_f64(),
            total_gb_runs / avx2_runs.as_secs_f64(),
            scalar_runs.as_secs_f64() / avx2_runs.as_secs_f64(),
        );
    }

    #[test]
    fn word_merge_matches_scalar_byte_for_byte() {
        fn next(state: &mut u64) -> u64 {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            *state
        }
        let mut rng = 0x9e37_79b9_7f4a_7c15u64;
        for len in (0..=40).chain([63, 64, 65, 4095, 4096, 65536]) {
            for mode in 0..4 {
                let old: Vec<u8> = (0..len).map(|_| next(&mut rng) as u8).collect();
                let mut title_s = old.clone();
                let mut host_s = old.clone();
                for j in 0..len {
                    let r = next(&mut rng);
                    if mode == 1 || (mode != 0 && r.is_multiple_of(17)) {
                        host_s[j] ^= ((r >> 8) as u8) | 1;
                    }
                    if mode == 2 || (mode != 0 && (r >> 3).is_multiple_of(13)) {
                        title_s[j] ^= (r >> 16) as u8;
                    }
                }
                let mut title_w = title_s.clone();
                let mut host_w = host_s.clone();
                let changed_s = super::merge_gpu_writes_scalar(&mut title_s, &mut host_s, &old);
                let changed_w = super::merge_gpu_writes_words(&mut title_w, &mut host_w, &old);
                assert_eq!(changed_w, changed_s, "len={len} mode={mode}");
                assert_eq!(title_w, title_s, "title mismatch len={len} mode={mode}");
                assert_eq!(host_w, host_s, "host mismatch len={len} mode={mode}");
            }
        }
    }

    #[test]
    fn copy_back_word_merge_preserves_concurrent_title_writes_and_aliases() {
        let old = vec![0x11u8; 64];
        let mut title = old.clone();
        let mut host = old.clone();
        // Simulate concurrent CPU writes across word boundaries `[5..10)` and `[31..35)`.
        title[5..10].copy_from_slice(&[0xaa, 0xbb, 0xcc, 0xdd, 0xee]);
        title[31..35].copy_from_slice(&[0x22, 0x33, 0x44, 0x55]);
        // Simulate GPU UAV write at `[16..20)` plus a single-byte overlap at index `9`.
        host[16..20].copy_from_slice(&[0xde, 0xad, 0xbe, 0xef]);
        host[9] = 0x77;

        let mut title_ref = title.clone();
        let mut host_ref = host.clone();
        assert!(super::merge_gpu_writes_words(&mut title, &mut host, &old));
        assert!(super::merge_gpu_writes_scalar(
            &mut title_ref,
            &mut host_ref,
            &old
        ));
        assert_eq!(title, title_ref);
        assert_eq!(host, host_ref);
        // Untouched title write bytes survive; GPU-written bytes win and sync host.
        assert_eq!(&title[5..9], &[0xaa, 0xbb, 0xcc, 0xdd]);
        assert_eq!(title[9], 0x77);
        assert_eq!(&title[16..20], &[0xde, 0xad, 0xbe, 0xef]);
        assert_eq!(&title[31..35], &[0x22, 0x33, 0x44, 0x55]);
        assert_eq!(host, title);
    }
}

fn copy_back_word_merge() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_COPY_BACK_WORD_MERGE").is_none_or(|v| v != "0"))
}

/// Scalar byte-by-byte merge of GPU writes (`host != old`) over `title`.
/// Returns whether `title` changed anywhere.
fn merge_gpu_writes_scalar(title: &mut [u8], host: &mut [u8], old: &[u8]) -> bool {
    let mut differs = 0u8;
    for ((t, h), o) in title.iter_mut().zip(host.iter_mut()).zip(old) {
        let gpu = if *h != *o { 0xffu8 } else { 0 };
        let merged = (*t & !gpu) | (*h & gpu);
        differs |= merged ^ *t;
        *t = merged;
        *h = merged;
    }
    differs != 0
}

/// Word-accelerated merge matching [`merge_gpu_writes_scalar`] byte for byte:
/// skips per-byte masking across 8-byte words where `host == old`.
fn merge_gpu_writes_words(title: &mut [u8], host: &mut [u8], old: &[u8]) -> bool {
    let n = title.len().min(host.len()).min(old.len());
    let mut i = 0usize;
    let mut differs = false;
    while i + 8 <= n {
        let h = u64::from_ne_bytes(host[i..i + 8].try_into().unwrap());
        let o = u64::from_ne_bytes(old[i..i + 8].try_into().unwrap());
        if h == o {
            let t = u64::from_ne_bytes(title[i..i + 8].try_into().unwrap());
            if t != h {
                host[i..i + 8].copy_from_slice(&title[i..i + 8]);
            }
        } else {
            differs |=
                merge_gpu_writes_scalar(&mut title[i..i + 8], &mut host[i..i + 8], &old[i..i + 8]);
        }
        i += 8;
    }
    if i < n {
        differs |= merge_gpu_writes_scalar(&mut title[i..n], &mut host[i..n], &old[i..n]);
    }
    differs
}

/// Host copies alive (also those only a list, a descriptor pass or a
/// snapshot still holds) and the bytes of title memory they copy.
static LIVE: [std::sync::atomic::AtomicU64; 2] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 2];

impl Drop for Shadow {
    fn drop(&mut self) {
        if self.place.is_some() {
            // Never counted with the copies (`Shadow::placed`).
            crate::inplace::gone(self.window, self.len);
            return;
        }
        let owned_len = self
            .gpu_owned
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .len() as u64;
        if owned_len > 0 {
            let _ = ACTIVE_GPU_OWNED.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                Some(v.saturating_sub(owned_len))
            });
            unregister_gpu_owned_shadow(self.id);
        }
        LIVE[0].fetch_sub(1, Ordering::Relaxed);
        LIVE[1].fetch_sub(self.len, Ordering::Relaxed);
        if let Some(slot) = LIVE_CLASS.get(self.live_class as usize) {
            slot.fetch_sub(self.len, Ordering::Relaxed);
            let ram_delta = if self.rebar != 0 { 0 } else { self.len as i64 };
            let vram_delta = self
                .card
                .as_ref()
                .map_or(0, |c| c.size as i64)
                .saturating_add(self.rebar as i64);
            crate::accounting::on_shadow_delta(-ram_delta, -vram_delta);
            crate::authority::unregister_range(self.start as usize, self.len as usize);
        }
        if self.rebar != 0 {
            crate::rebar::released(self.rebar);
        }
        if self.kept.get_mut().unwrap_or_else(|e| e.into_inner()).1 {
            KEPT_BYTES.fetch_sub(self.len, Ordering::Relaxed);
        }
        let window = self.window;
        if let Some(card) = &mut self.card {
            if window {
                WINDOW_CARD_BYTES.fetch_sub(card.size, Ordering::Relaxed);
            } else {
                CARD_BYTES[0].fetch_sub(card.size, Ordering::Relaxed);
            }
            if card
                .before
                .get_mut()
                .unwrap_or_else(|e| e.into_inner())
                .is_some()
            {
                CARD_BYTES[1].fetch_sub(card_hash_bytes(card.size), Ordering::Relaxed);
            }
        }
    }
}

/// Whether a copy's snapshot is kept from one submission to the next and
/// only the pieces that changed in between are taken again
/// (`VICTORY_KEEP_SNAPSHOTS=1`; off by default).
///
/// A snapshot is the copy's bytes before a submission, to find what its
/// shaders wrote. Taken anew for every submission it is 1,050 MB of copying
/// a frame (72-86 ms in gameplay, runs 61 and 62) for buffers of which
/// 100 MB changed. Kept, a piece of it is taken again only after something
/// changed the copy there: a refresh from title memory ([`Shadow::copy`]),
/// shader writes found after a submission, or the bytes put back then
/// ([`Shadow::copy_back`]).
///
/// What it assumes, and why it is not on by default: that nothing else
/// writes a copy. A shader that writes a buffer through a view we did not
/// declare as written (we declare the unordered-access views of shaders
/// that can store) changes the copy unseen; with a fresh snapshot that
/// write is simply never put back, with a kept one it would be taken for a
/// write of a later submission and put back then, over whatever the title
/// wrote there since.
fn keeps() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_KEEP_SNAPSHOTS").is_some_and(|v| v == "1"))
}

/// Bytes of the snapshots kept, and the most that may be.
static KEPT_BYTES: AtomicU64 = AtomicU64::new(0);
/// `VICTORY_KEEP_SNAPSHOTS_MB`, 256 unless set (it was 768: run 80, with
/// kept snapshots beside copies in the card's memory, ran the PC out of
/// memory in 27 s; kept snapshots are the PC's memory, all of it resident).
fn kept_most() -> u64 {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_KEEP_SNAPSHOTS_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(256)
            << 20
    })
}

/// The 64 KiB pieces, cut at multiples of 64 KiB of title memory, that
/// hold offsets `[from, to)` of a copy of memory from `start` on
/// (`from < to`); piece 0 holds offset 0.
fn pieces_of(start: usize, from: usize, to: usize) -> std::ops::Range<usize> {
    let first = start / PIECE;
    (start + from) / PIECE - first..(start + to - 1) / PIECE - first + 1
}

/// The offsets that piece `piece` holds of a copy of `len` bytes of memory
/// from `start` on.
fn piece_at(start: usize, len: usize, piece: usize) -> (usize, usize) {
    let first = start / PIECE;
    let lo = ((first + piece) * PIECE).saturating_sub(start);
    let hi = ((first + piece + 1) * PIECE - start).min(len);
    (lo, hi)
}

#[cfg(test)]
mod piece_tests {
    use super::{PIECE, piece_at, pieces_of};

    /// The pieces cover a copy without gaps, each cut where the compare
    /// cuts (at multiples of 64 KiB of title memory).
    #[test]
    fn pieces_cover_the_copy_as_the_compare_cuts_it() {
        for start in [
            0usize,
            1,
            4096,
            PIECE - 1,
            PIECE,
            PIECE + 77,
            5 * PIECE + 12345,
        ] {
            for len in [1usize, 100, PIECE - 1, PIECE, PIECE + 1, 3 * PIECE + 500] {
                let count = pieces_of(start, 0, len).end;
                let mut at = 0;
                for piece in 0..count {
                    let (lo, hi) = piece_at(start, len, piece);
                    assert_eq!(lo, at, "{start} {len} {piece}");
                    // As `Shadow::differing` cuts: to the next multiple.
                    let n = (PIECE - (start + at) % PIECE).min(len - at);
                    assert_eq!(hi, at + n, "{start} {len} {piece}");
                    assert_eq!(pieces_of(start, lo, hi), piece..piece + 1);
                    assert_eq!(pieces_of(start, hi - 1, hi), piece..piece + 1);
                    at = hi;
                }
                assert_eq!(at, len, "{start} {len}");
                assert_eq!(pieces_of(start, 0, len), 0..count);
            }
        }
    }
}

/// Copies of windows outside buffers let go since the start, and their
/// bytes.
static LET_GO: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// For the memory line: the windows let go since the start and their
/// bytes, and of the bytes of the windows alive, how many lie between the
/// lowest and the highest address ever asked for in each.
#[cfg(feature = "diagnostics")]
pub(crate) fn windows() -> [u64; 3] {
    let asked: u64 = {
        let loose = LOOSE.read().unwrap_or_else(|e| e.into_inner());
        loose
            .values()
            .map(|s| {
                let low = s.asked[0].load(Ordering::Relaxed);
                let high = s.asked[1].load(Ordering::Relaxed);
                high.saturating_sub(low)
            })
            .sum()
    };
    [
        LET_GO[0].load(Ordering::Relaxed),
        LET_GO[1].load(Ordering::Relaxed),
        asked,
    ]
}

/// For the memory line, of the windows alive: the bytes of their pages
/// ever copied from title memory (the rest of each window is memory the
/// host zeroed and nothing read: what a window made no bigger than what is
/// read would not hold); the windows a descriptor slot shows and their
/// bytes (they stay until the title writes the slot again); the windows a
/// list or a pass over the heap holds and their bytes; the windows asked
/// for without a length (root views, which keep the 4 MiB grid) and their
/// bytes; and the window grid in bytes ([`window`]).
#[cfg(feature = "diagnostics")]
pub(crate) fn window_use() -> [u64; 8] {
    let mut out = [0u64; 8];
    let loose = LOOSE.read().unwrap_or_else(|e| e.into_inner());
    for s in loose.values() {
        let copied = s.copied.lock().unwrap_or_else(|e| e.into_inner());
        let pages = copied.copied_count() as u64;
        out[0] += (pages * PAGE as u64).min(s.len);
        if crate::host::references(&s.buffer) > 1 {
            out[1] += 1;
            out[2] += s.len;
        }
        if Arc::strong_count(s) > 1 {
            out[3] += 1;
            out[4] += s.len;
        }
        if s.unsized_asks.load(Ordering::Relaxed) > 0 {
            out[5] += 1;
            out[6] += s.len;
        }
    }
    out[7] = window();
    out
}

/// Lets go of the copies of windows outside buffers that nothing names any
/// more. They were never freed, and the world's streaming keeps asking for
/// new ones: 648 windows (2.8 GB) when the cutscene began, 974 (4.3 GB)
/// eight minutes later (run 24). A window goes when
/// - only the table of windows holds its copy: no list not yet reset reads
///   or writes it, no pass over the heap keeps its range, no submission is
///   comparing it;
/// - only the copy holds its host buffer: no descriptor slot views it (a
///   slot keeps the buffer it views, `HostHeap::kept`), so no draw can read
///   it through a slot made earlier;
/// - nothing asked for it for `IDLE_MS` and `IDLE_FRAMES` frames.
///
/// The next read of that memory makes a new copy and copies the window
/// again. Called before a submission starts, by the thread that submits.
pub(crate) fn let_go_idle() {
    const IDLE_FRAMES: u64 = 4;
    let gone: Vec<Arc<Shadow>> = {
        // Held from the look to the removal: `loose` hands a copy out
        // under the same lock, so none is handed out in between.
        let mut map = LOOSE.write().unwrap_or_else(|e| e.into_inner());
        if map.is_empty() {
            return;
        }
        let idle_ms = crate::host::ram_idle_ms(crate::host::available_physical_ram());
        let idle: Vec<(u64, u64)> = map
            .iter()
            .filter(|(_, s)| {
                Arc::strong_count(s) == 1
                    && crate::host::long_ago(s.used.load(Ordering::Relaxed), idle_ms, IDLE_FRAMES)
                    && crate::host::references(&s.buffer) == 1
            })
            .map(|(key, _)| *key)
            .collect();
        let taken: Vec<Arc<Shadow>> = idle
            .into_iter()
            .filter_map(|key| map.remove(&key))
            .collect();
        if !taken.is_empty() {
            bump_shadow_epoch();
        }
        taken
    };
    if !gone.is_empty() {
        let (went, bytes) = (gone.len(), gone.iter().map(|s| s.len).sum::<u64>());
        crate::host::trail(|| {
            format!(
                "sweep: {went} host copies of windows of title memory let go ({} KiB), held by no list and no slot",
                bytes >> 10
            )
        });
    }
    for shadow in gone {
        LET_GO[0].fetch_add(1, Ordering::Relaxed);
        LET_GO[1].fetch_add(shadow.len, Ordering::Relaxed);
        // Nothing runs now and nothing names it: freed at once unless a
        // submission is running after all.
        crate::host::retire(shadow);
    }
}

/// For the memory line: host copies alive and their bytes, of them the
/// windows of memory outside buffers and their bytes, and the snapshot
/// buffers kept for reuse and the bytes they reserve.
#[cfg(feature = "diagnostics")]
pub(crate) fn memory() -> [u64; 6] {
    let (windows, window_bytes) = {
        let loose = LOOSE.read().unwrap_or_else(|e| e.into_inner());
        // Each copy once: a window may stand under several keys.
        let mut seen = std::collections::HashSet::new();
        let bytes: u64 = loose
            .values()
            .filter(|s| seen.insert(Arc::as_ptr(s) as usize))
            .map(|s| s.len)
            .sum();
        (seen.len() as u64, bytes)
    };
    let (kept, kept_bytes) = {
        let pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        (
            pool.len() as u64,
            pool.iter().map(|v| v.capacity() as u64).sum::<u64>(),
        )
    };
    [
        LIVE[0].load(Ordering::Relaxed),
        LIVE[1].load(Ordering::Relaxed),
        windows,
        window_bytes,
        kept,
        kept_bytes,
    ]
}

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// The title memory the copy is of: its start and length.
    pub(crate) fn span(&self) -> (u64, u64) {
        (self.start, self.len)
    }

    /// The offset of a title address in the copy, if it lies in it.
    pub(crate) fn offset_in(&self, address: u64) -> Option<u64> {
        address.checked_sub(self.start).filter(|&at| at < self.len)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// Whether the memory behind the copy turned out not to be write-watched:
    /// every refresh then copies its whole range.
    pub(crate) fn unwatched(&self) -> bool {
        self.watched.load(Ordering::Relaxed) == NOT_WATCHED
    }

    /// Whether the memory behind the copy is physical-page view memory tracked
    /// by guards rather than `GetWriteWatch` (`ask_ahead` can skip it).
    pub(crate) fn is_view_guarded(&self) -> bool {
        self.watched.load(Ordering::Relaxed) == VIEW_GUARDED
    }

    /// The copy's own number (see the field).
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start + self.place.as_ref().map_or(0, |place| place.1)
    }

    /// Whether the copy is placed over title memory in place (other
    /// buffers may alias it).
    pub(crate) fn in_place(&self) -> bool {
        self.place.is_some()
    }

    /// Title memory `[start, start + len)` in place: `placed` is a buffer
    /// over it in a heap opened on it. Shaped as a copy so that everything
    /// that binds, views or keeps a copy works on it unchanged; what
    /// copies does nothing ([`Shadow::update`], [`Shadow::gpu_writes`]).
    fn placed(placed: crate::inplace::Placed, start: u64, len: u64, window: bool) -> Shadow {
        crate::host::name(&placed.buffer, || {
            format!("title memory {start:#x}+{len:#x} in place")
        });
        let mut shadow = Shadow::new(placed.buffer, start as *mut u8, placed.gpu, start, len);
        // Not a copy: not counted with them, and nothing asks what the
        // title wrote there.
        LIVE[0].fetch_sub(1, Ordering::Relaxed);
        LIVE[1].fetch_sub(len, Ordering::Relaxed);
        *shadow.watched.get_mut() = NOT_WATCHED;
        shadow.window = window;
        shadow.place = Some((placed.heap, placed.bias));
        shadow
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Shadow {
        let pages = (start + len).div_ceil(PAGE as u64) - start / PAGE as u64;
        LIVE[0].fetch_add(1, Ordering::Relaxed);
        LIVE[1].fetch_add(len, Ordering::Relaxed);
        Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            copied: Mutex::new(CopiedPages::new(pages as usize)),
            watched: AtomicU8::new(0),
            pending: Mutex::default(),
            flights: Mutex::default(),
            gpu_owned: Mutex::default(),
            has_gpu_owned: std::sync::atomic::AtomicBool::new(false),
            cold_copied_asking: AtomicU64::new(0),
            copied_4k: Mutex::default(),
            before: Mutex::default(),
            kept: Mutex::default(),
            used: AtomicU64::new(crate::host::moment()),
            asked: [AtomicU64::new(u64::MAX), AtomicU64::new(0)],
            unsized_asks: AtomicU64::new(0),
            id: {
                static NEXT: AtomicU64 = AtomicU64::new(1);
                NEXT.fetch_add(1, Ordering::Relaxed)
            },
            card: None,
            window: false,
            place: None,
            rebar: 0,
            live_class: u8::MAX,
        }
    }

    fn set_live_class(&mut self, class: u8) {
        if (class as usize) < LIVE_CLASS.len() && self.live_class == u8::MAX {
            self.live_class = class;
            LIVE_CLASS[class as usize].fetch_add(self.len, Ordering::Relaxed);
            let ram_delta = if self.rebar != 0 { 0 } else { self.len as i64 };
            let vram_delta = self
                .card
                .as_ref()
                .map_or(0, |c| c.size as i64)
                .saturating_add(self.rebar as i64);
            crate::accounting::on_shadow_delta(ram_delta, vram_delta);
            crate::authority::register_range(
                self.start as usize,
                self.len as usize,
                crate::authority::AuthorityClass::SharedCoherent,
                crate::readback::stats().creator_verified,
            );
        }
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        // In place the shaders write title memory itself: no snapshot, no
        // compare, nothing to copy back, and no wait on its account.
        // shortcut: another host copy of the same bytes (a title buffer
        // inside this one with a copy of its own, a texture in it) is not
        // told: the card's writes are not seen by write watching.
        if self.place.is_some() {
            return;
        }
        let len = len.min(self.left(address));
        if len == 0 {
            return;
        }
        let tagged = crate::readback::tagged_in(address, len);
        let mut needs_copy_back = crate::readback::copy_back(address, len);
        if crate::authority::enabled() {
            let rb = crate::readback::stats();
            let readback_ready =
                rb.creator_verified && rb.tagged_buffers > 0 && crate::queue::presented() > 0;
            let can_skip = crate::authority::observe_gpu_write_and_check(
                address as usize,
                len as usize,
                tagged,
                readback_ready,
            );
            if readback_ready && can_skip {
                crate::authority::note_skipped_card_compare();
                needs_copy_back = false;
                crate::readback::validate_authority_skip(address, len, tagged);
            }
        }
        // Always record GPU write awareness even when GPU->CPU copy-back is elided
        // so refreshes, hazard waits, and future CPU reads preserve GPU-authored data.
        list.defer_gpu_write(self, address as usize, len as usize, needs_copy_back);
    }

    /// Records a GPU write `[address, address + len)` whose GPU->CPU copy-back
    /// was elided, storing the watch-asking sequence so subsequent CPU refreshes
    /// on adjacent 4 KiB pages never clobber GPU-authored bytes.
    pub(crate) fn note_gpu_write(self: &Arc<Self>, address: usize, len: usize) {
        if self.place.is_some() || len == 0 {
            return;
        }
        let (off_from, off_to) = self.inside(address, len);
        if off_from >= off_to {
            return;
        }
        let base = self.start as usize;
        let (from, to) = (base + off_from, base + off_to);
        if self.has_gpu_owned.load(Ordering::Acquire) {
            let mut owned = self.gpu_owned.lock().unwrap_or_else(|e| e.into_inner());
            let pos = owned.partition_point(|r| r.end < from);
            if let Some(existing) = owned.get_mut(pos)
                && existing.start <= from
                && to <= existing.end
            {
                existing.watch_seq = existing
                    .watch_seq
                    .max(crate::watch::advance_gpu_owned_seq(from, to, false));
                return;
            }
        }
        let cold_reset = self.watched.load(Ordering::Relaxed) == 0
            || self.cold_copied_asking.load(Ordering::Acquire) != 0;
        let seq = crate::watch::advance_gpu_owned_seq(from, to, cold_reset);
        self.add_gpu_owned_interval_with_owner(from, to, seq, Some(self));
    }

    fn add_gpu_owned_interval_with_owner(
        &self,
        from: usize,
        to: usize,
        watch_seq: u64,
        owner: Option<&Arc<Shadow>>,
    ) {
        if from >= to {
            return;
        }
        let mut owned = self.gpu_owned.lock().unwrap_or_else(|e| e.into_inner());
        let prev = owned.len() as u64;
        // Fast path: existing sorted interval already covers `[from, to)`.
        let pos = owned.partition_point(|r| r.end < from);
        if let Some(existing) = owned.get_mut(pos)
            && existing.start <= from
            && to <= existing.end
        {
            existing.watch_seq = existing.watch_seq.max(watch_seq);
            return;
        }
        // Merge in-place while maintaining sorted disjoint invariant without full sort.
        let mut merge_from = from;
        let mut merge_to = to;
        let mut merge_seq = watch_seq;
        let mut end_pos = pos;
        while end_pos < owned.len() && owned[end_pos].start <= merge_to {
            let r = owned[end_pos];
            merge_from = merge_from.min(r.start);
            merge_to = merge_to.max(r.end);
            merge_seq = merge_seq.max(r.watch_seq);
            end_pos += 1;
        }
        let new_entry = GpuOwnedRange {
            start: merge_from,
            end: merge_to,
            watch_seq: merge_seq,
        };
        if end_pos > pos {
            owned[pos] = new_entry;
            if end_pos > pos + 1 {
                owned.drain((pos + 1)..end_pos);
            }
        } else {
            owned.insert(pos, new_entry);
        }
        let next = owned.len() as u64;
        drop(owned);
        self.has_gpu_owned.store(next > 0, Ordering::Release);
        match next.cmp(&prev) {
            std::cmp::Ordering::Greater => {
                ACTIVE_GPU_OWNED.fetch_add(next - prev, Ordering::Relaxed);
            }
            std::cmp::Ordering::Less => {
                let _ = ACTIVE_GPU_OWNED.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                    Some(v.saturating_sub(prev - next))
                });
            }
            std::cmp::Ordering::Equal => {}
        }
        if prev == 0
            && next > 0
            && let Some(arc) = owner
        {
            register_gpu_owned_shadow(arc);
        }
    }

    fn remove_gpu_owned_interval(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        if from >= to || !self.has_gpu_owned.load(Ordering::Acquire) {
            return Vec::new();
        }
        let mut owned = self.gpu_owned.lock().unwrap_or_else(|e| e.into_inner());
        let first = owned.partition_point(|r| r.end <= from);
        if first >= owned.len() || owned[first].start >= to {
            return Vec::new();
        }
        let prev = owned.len() as u64;
        let mut removed = Vec::new();
        let mut replacement: [Option<GpuOwnedRange>; 2] = [None, None];
        let mut end = first;
        while end < owned.len() && owned[end].start < to {
            let r = owned[end];
            let ov_a = r.start.max(from);
            let ov_b = r.end.min(to);
            if ov_a < ov_b {
                removed.push((ov_a, ov_b));
            }
            if r.start < from {
                replacement[0] = Some(GpuOwnedRange {
                    start: r.start,
                    end: from,
                    watch_seq: r.watch_seq,
                });
            }
            if to < r.end {
                let slot = if replacement[0].is_none() { 0 } else { 1 };
                replacement[slot] = Some(GpuOwnedRange {
                    start: to,
                    end: r.end,
                    watch_seq: r.watch_seq,
                });
            }
            end += 1;
        }
        let mut rep_vec = [replacement[0], replacement[1]].into_iter().flatten();
        owned.splice(first..end, &mut rep_vec);
        let next = owned.len() as u64;
        drop(owned);
        self.has_gpu_owned.store(next > 0, Ordering::Release);
        if next < prev {
            let _ = ACTIVE_GPU_OWNED.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                Some(v.saturating_sub(prev - next))
            });
        }
        if prev > 0 && next == 0 {
            if let Ok(mut kmap) = self.copied_4k.try_lock() {
                kmap.clear();
            }
            unregister_gpu_owned_shadow(self.id);
        }
        removed
    }

    /// Materializes any GPU-owned bytes overlapping `[address, address + len)` back
    /// into title memory before a legitimate CPU read, demoting GPU authority.
    pub(crate) fn materialize_gpu_owned(&self, address: usize, len: usize) -> usize {
        if self.place.is_some() || len == 0 || !self.has_gpu_owned.load(Ordering::Acquire) {
            return 0;
        }
        let (off_from, off_to) = self.inside(address, len);
        if off_from >= off_to {
            return 0;
        }
        let t0 = std::time::Instant::now();
        let base = self.start as usize;
        let (q_from, q_to) = (base + off_from, base + off_to);
        let removed = self.remove_gpu_owned_interval(q_from, q_to);
        if removed.is_empty() {
            return 0;
        }
        crate::authority::observe_cpu_read_or_copy(q_from, q_to - q_from);
        if let Some(card) = &self.card {
            let _ = crate::host::host().map(|h| {
                h.fetch_buffer_slice(
                    &card.staging,
                    &self.buffer,
                    off_from as u64,
                    (off_to - off_from) as u64,
                )
            });
        }
        let mut materialized = 0usize;
        for (a, b) in removed {
            let n = b - a;
            // SAFETY: `[a, b)` lies inside `[self.start, self.start + self.len)`,
            // so `self.data.add(a - base)` is readable for `n` bytes.
            unsafe {
                let src = std::slice::from_raw_parts(self.data.add(a - base).cast_const(), n);
                crate::watch::writing(a, n);
                crate::host::write_mapped(a as *mut u8, src.as_ptr(), n);
            }
            materialized += n;
        }
        crate::rea_telemetry::record_prep_phase_us(
            crate::rea_telemetry::PrepPhase::Materialize,
            t0.elapsed().as_micros() as u64,
        );
        materialized
    }

    /// Clears GPU ownership when the CPU explicitly overwrites `[address, address + len)`.
    pub(crate) fn on_cpu_overwrite(&self, address: usize, len: usize) {
        if self.place.is_some() || len == 0 || !self.has_gpu_owned.load(Ordering::Acquire) {
            return;
        }
        let (off_from, off_to) = self.inside(address, len);
        if off_from >= off_to {
            return;
        }
        let base = self.start as usize;
        let (q_from, q_to) = (base + off_from, base + off_to);
        if !self.remove_gpu_owned_interval(q_from, q_to).is_empty() {
            crate::authority::observe_cpu_write(q_from, q_to - q_from);
        }
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// one snapshot per copy and submission, each byte taken once, when the
    /// first range holding it is declared (the world declares overlapping
    /// views of its 174 MB heap dozens of times, from different offsets: a
    /// snapshot grown to cover each was copied whole every time), in reused
    /// memory (fresh snapshots cost ~2.3 million page faults per world
    /// frame).
    pub(crate) fn writing(&self, address: usize, len: usize) {
        if self.place.is_some() {
            return;
        }
        let range = (address, address + len);
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        // With the copy-back on its own thread: counted for this submission
        // (once), whether or not an earlier one still holds the range.
        if relay() {
            self.flights
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .declare(range, PREPARED.load(Ordering::Relaxed));
        }
        // Dispatches declare the same or subsumed ranges thousands of times.
        // Never skip a subrange while `relay()` tracks exact per-flight ranges in
        // `Flights`, or `copy_back_done` will unprotect inner flights early.
        if pending.contains(&range) {
            return;
        }
        if !relay()
            && dirty_ranges_enabled()
            && self.card.is_none()
            && pending.iter().any(|&(a, e)| a <= range.0 && range.1 <= e)
        {
            DIRTY_SUBSUMED_SKIPS.fetch_add(1, Ordering::Relaxed);
            return;
        }
        // In the card's memory: no snapshot. What shaders wrote is found on
        // the card (`Host::card_before` reads the declared ranges from the
        // lists).
        if self.card.is_some() {
            pending.push(range);
            return;
        }
        // The parts no declared range holds yet: their bytes are taken now,
        // just after the range's refresh (a part declared earlier has not
        // been refreshed since, and keeps the bytes taken then).
        let mut parts = vec![self.inside(address, len)];
        for &(a, e) in pending.iter() {
            let (a, e) = self.inside(a, e - a);
            let mut rest = Vec::with_capacity(parts.len() + 1);
            for (from, to) in parts {
                if e <= from || a >= to {
                    rest.push((from, to));
                    continue;
                }
                if from < a {
                    rest.push((from, a));
                }
                if e < to {
                    rest.push((e, to));
                }
            }
            parts = rest;
        }
        pending.push(range);
        let taken: usize = parts
            .iter()
            .map(|&(from, to)| to.saturating_sub(from))
            .sum();
        BACK_CLASS[usize::from(!self.window)].fetch_add(taken as u64, Ordering::Relaxed);
        let total = self.len as usize;
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let snapshot = before.get_or_insert_with(|| pooled(total));
        if keeps() {
            let began = std::time::Instant::now();
            // Piece by piece, and only the pieces not known current. A
            // piece the part holds whole is current from now on; of a
            // piece it holds partly only the part's bytes are taken, as
            // without the switch, and it is taken again next time (the
            // rest of it may belong to a range declared earlier, whose
            // bytes stay as they were taken).
            let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            if kept.0.is_empty() {
                kept.0 = vec![false; self.pieces(0, total.max(1)).end];
            }
            for (from, to) in parts {
                if from >= to {
                    continue;
                }
                for piece in self.pieces(from, to) {
                    if kept.0[piece] {
                        continue;
                    }
                    let (lo, hi) = self.piece(piece);
                    let whole = from <= lo && hi <= to;
                    let (lo, hi) = (lo.max(from), hi.min(to));
                    // SAFETY: as below; `hi <= total`.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            self.data.add(lo),
                            snapshot.as_mut_ptr().add(lo),
                            hi - lo,
                        );
                    }
                    kept.0[piece] = whole;
                }
            }
            BACK_TIMING_US[0].fetch_add(began.elapsed().as_micros() as u64, Ordering::Relaxed);
            return;
        }
        if wide_back() {
            // Noted, and taken with the submission's other snapshots when
            // its refreshes are done (`take_snapshots`): the range is
            // declared from here on, so nothing changes its bytes before
            // then. The snapshot's memory stays where it is until the
            // range is done with (`copy_back_done`).
            let mut snaps = SNAPS.lock().unwrap_or_else(|e| e.into_inner());
            let (data, snapshot) = (self.data as usize, snapshot.as_mut_ptr() as usize);
            snaps.extend(
                parts
                    .into_iter()
                    .filter(|&(from, to)| from < to)
                    .map(|(from, to)| (data + from, snapshot + from, to - from)),
            );
            return;
        }
        let began = std::time::Instant::now();
        for (from, to) in parts {
            if from < to {
                // SAFETY: the copy's mapping and the snapshot's allocation
                // (`pooled`) both hold `total` bytes, and `to <= total`.
                unsafe {
                    copy_wide(
                        self.data.add(from),
                        snapshot.as_mut_ptr().add(from),
                        to - from,
                    );
                }
            }
        }
        BACK_TIMING_US[0].fetch_add(began.elapsed().as_micros() as u64, Ordering::Relaxed);
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written. The range must lie in ranges declared with
    /// [`Shadow::writing`] and not yet done: `Host::execute` passes the
    /// union of a submission's ranges, each byte once, then calls
    /// [`Shadow::copy_back_done`].
    pub(crate) fn copy_back(&self, address: usize, len: usize) -> usize {
        self.copy_back_with(address, len, None)
    }

    /// [`Shadow::copy_back`]; `found`, if given, is the pieces of the range
    /// that differ from the snapshot, found beforehand ([`find_all`]).
    pub(crate) fn copy_back_with(
        &self,
        address: usize,
        len: usize,
        found: Option<Vec<(usize, usize)>>,
    ) -> usize {
        if self.place.is_some() {
            return 0;
        }
        let snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        self.compared(from, to);
        if let Some(before) = &*snapshot
            && from < to
        {
            // SAFETY: `writing` set the snapshot's bytes of every declared
            // range, at the copy's offsets.
            let before =
                unsafe { std::slice::from_raw_parts(before.as_ptr().add(from), to - from) };
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission (on several threads: gigabytes a frame, bound by
            // memory speed), title memory read only where the GPU wrote.
            let scanning = std::time::Instant::now();
            let differing = match found {
                Some(found) => found,
                None => {
                    let res = self.differing(from, to, before);
                    BACK_TIMING_US[1]
                        .fetch_add(scanning.elapsed().as_micros() as u64, Ordering::Relaxed);
                    res
                }
            };
            let syncing = std::time::Instant::now();
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                add(C::BackScanUs, scanning.elapsed().as_micros() as u64);
                add(C::BackDiffer, differing.len() as u64);
            }
            // Where the copy changes (the GPU wrote, and below the title's
            // bytes come in), a kept snapshot is no longer current.
            for &(at, n) in &differing {
                self.changed(at, at + n);
            }
            let mut title = vec![0u8; PIECE];
            for (at, n) in differing {
                // SAFETY: inside the mapping; only the submitting thread
                // writes it.
                let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                let old = &before[at - from..at - from + n];
                // From the first byte that differs to the last: title
                // memory is read and written there only (a shader that sets
                // a few words of a piece cost a read, a walk and a write of
                // all 64 KiB of it).
                let Some(first) = first_difference(host, old) else {
                    continue;
                };
                let last = last_difference(host, old).unwrap_or(first);
                let (at, n) = (at + first, last + 1 - first);
                let (host, old) = (&mut host[first..=last], &old[first..=last]);
                let title = &mut title[..n];
                if !crate::host::read_into((start + at) as *const u8, title) {
                    continue;
                }
                // Title memory with the GPU's bytes put in; the copy takes
                // the title's bytes in turn.
                // Without a branch, so the loop runs many bytes at a time:
                // the GPU's byte where the copy's byte is not what it was
                // before the submission, else the title's. (With the two
                // searches above this was 44 ms a frame for 970 pieces.)
                let differs = if copy_back_word_merge() {
                    merge_gpu_writes_words(title, host, old)
                } else {
                    merge_gpu_writes_scalar(title, host, old)
                };
                if differs {
                    crate::watch::writing(start + at, n);
                    crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
                    written += n;
                }
            }
            // Our write shows as a title write at the copy's next refresh,
            // which copies those pages once more (some dozens of MB a
            // frame). Asking about the whole range here, to note them
            // current, cost more: 19-31 askings a frame over 110-174 MB of
            // view memory, each a walk of its guards.
            let sync_us = syncing.elapsed().as_micros() as u64;
            BACK_TIMING_US[2].fetch_add(sync_us, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::BackSyncUs, sync_us);
        }
        if self.has_gpu_owned.load(Ordering::Acquire) && from < to {
            self.remove_gpu_owned_interval(start + from, start + to);
        }
        written
    }

    /// Offsets `[from, to)` of the copy are compared with the snapshot: for
    /// the frame line's counts.
    fn compared(&self, from: usize, to: usize) {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::BackCompared, (to - from) as u64);
        let (bytes, ranges) = if self.window { (2, 5) } else { (3, 6) };
        BACK_CLASS[bytes].fetch_add((to - from) as u64, Ordering::Relaxed);
        BACK_CLASS[ranges].fetch_add(1, Ordering::Relaxed);
        if !self.window && self.len >= 1 << 20 {
            BACK_CLASS[4].fetch_add((to - from) as u64, Ordering::Relaxed);
        }
    }

    /// [`relay`]: the declared range `[address, address + len)` of a copy
    /// in the PC's memory is about to be looked through
    /// (`crate::shadow::find_all`) by the thread that settles: counted as
    /// [`Shadow::copy_back_with`] counts it.
    pub(crate) fn relay_compared(&self, address: usize, len: usize) {
        let (from, to) = self.inside(address, len);
        self.compared(from, to);
    }

    /// [`relay`], for a copy in the PC's memory: offsets `[at, at + n)`
    /// are a 64 KiB piece of a declared range that differed from the
    /// snapshot when it was looked through. What the card wrote there
    /// since the snapshot was last brought up goes into title memory, from
    /// the first to the last such byte, and into the snapshot, so that the
    /// next flight's merge finds only what the card writes from now on.
    /// The copy is read once and never written: a later flight's shaders
    /// may be writing it at this moment. Under the snapshot's lock, which
    /// [`Shadow::merge_under_writes`] holds while it reads title memory
    /// and puts the title's bytes into the copy and the snapshot: the two
    /// never see each other half done. `now` and `title` are scratch
    /// memory. Returns the bytes written to title memory.
    pub(crate) fn relay_merge(
        &self,
        at: usize,
        n: usize,
        now: &mut Vec<u8>,
        title: &mut Vec<u8>,
    ) -> usize {
        let start = self.start as usize;
        if self.card.is_some() || n == 0 || at + n > self.len as usize {
            return 0;
        }
        let began = std::time::Instant::now();
        let mut snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let Some(before) = snapshot.as_mut() else {
            return 0;
        };
        now.clear();
        now.reserve(n);
        // SAFETY: inside the copy's mapping (checked above), read into
        // room just reserved; the card may write the mapping meanwhile, and
        // each byte read is one it held at some moment.
        unsafe {
            std::ptr::copy_nonoverlapping(self.data.add(at), now.as_mut_ptr(), n);
            now.set_len(n);
        }
        // SAFETY: the snapshot's allocation holds the copy's length
        // (`writing`), and these offsets are in a range declared and not
        // done with, whose bytes it holds; nothing else touches them
        // without the lock held here.
        let held = unsafe { std::slice::from_raw_parts_mut(before.as_mut_ptr().add(at), n) };
        let Some(first) = first_difference(now, held) else {
            BACK_TIMING_US[2].fetch_add(began.elapsed().as_micros() as u64, Ordering::Relaxed);
            return 0;
        };
        let last = last_difference(now, held).unwrap_or(first);
        let (at, n) = (at + first, last + 1 - first);
        let (now, held) = (&now[first..=last], &mut held[first..=last]);
        title.clear();
        title.resize(n, 0);
        // Title memory not there: looked at again when the next flight
        // that declared the range lands (the snapshot stays as it is).
        if !crate::host::read_into((start + at) as *const u8, title) {
            BACK_TIMING_US[2].fetch_add(began.elapsed().as_micros() as u64, Ordering::Relaxed);
            return 0;
        }
        if !merge_piece(now, held, title) {
            BACK_TIMING_US[2].fetch_add(began.elapsed().as_micros() as u64, Ordering::Relaxed);
            return 0;
        }
        crate::watch::writing(start + at, n);
        crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
        BACK_TIMING_US[2].fetch_add(began.elapsed().as_micros() as u64, Ordering::Relaxed);
        n
    }

    /// The 64 KiB pieces (cut at multiples of 64 KiB of title memory) of
    /// `[from, to)` of the copy that differ from `before` (the bytes from
    /// `from` on), as (offset, bytes) in order; compared on several threads
    /// when the range is big.
    fn differing(&self, from: usize, to: usize, before: &[u8]) -> Vec<(usize, usize)> {
        let (start, data) = (self.start as usize, self.data as usize);
        let scan = |a: usize, b: usize| -> Vec<(usize, usize)> {
            let mut found = Vec::new();
            let mut at = a;
            while at < b {
                let n = (PIECE - (start + at) % PIECE).min(b - at);
                // SAFETY: inside the mapping, which nothing writes while a
                // submission's results are compared.
                let host = unsafe { std::slice::from_raw_parts((data + at) as *const u8, n) };
                DIRTY_DIFF_CALLS.fetch_add(1, Ordering::Relaxed);
                DIRTY_DIFF_BYTES.fetch_add(n as u64, Ordering::Relaxed);
                if slices_differ(host, &before[at - from..at - from + n]) {
                    found.push((at, n));
                }
                at += n;
            }
            found
        };
        const PER_THREAD: usize = 16 << 20;
        let threads = ((to - from) / PER_THREAD).clamp(1, 8);
        if threads == 1 {
            return scan(from, to);
        }
        // Cut on piece boundaries, so that every thread sees whole pieces.
        let cut = |i: usize| {
            if i == threads {
                return to;
            }
            let at = from + (to - from) / threads * i;
            (at - (start + at) % PIECE).max(from)
        };
        let scan = &scan;
        std::thread::scope(|s| {
            let parts: Vec<_> = (0..threads)
                .map(|i| {
                    let (a, b) = (cut(i), cut(i + 1));
                    s.spawn(move || scan(a, b))
                })
                .collect();
            parts
                .into_iter()
                .flat_map(|part| match part.join() {
                    Ok(found) => found,
                    Err(panic) => std::panic::resume_unwind(panic),
                })
                .collect()
        })
    }

    /// After [`Shadow::copy_back`] of a submission's ranges: every submitted
    /// step that declared one of `ranges` (`(address, end)`, as declared) is
    /// done with it (one submission runs at a time), and the snapshot with
    /// the last range.
    ///
    /// With [`relay`] several submissions may hold a range: `ranges` are
    /// those of one of them (each once), and a range is done with when the
    /// last submission that declared it is ([`Flights`]).
    pub(crate) fn copy_back_done(&self, ranges: &[(usize, usize)]) {
        if self.place.is_some() {
            return;
        }
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        if relay() {
            let free = self
                .flights
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .done(ranges);
            pending.retain(|r| !free.contains(r));
        } else {
            pending.retain(|r| !ranges.contains(r));
        }
        if !pending.is_empty() || self.card.is_some() {
            return;
        }
        if keeps() {
            // Kept for the next submission, if it is worth it (a megabyte
            // or more) and there is room.
            let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            if !kept.1 && self.len >= 1 << 20 {
                if KEPT_BYTES.fetch_add(self.len, Ordering::Relaxed) + self.len <= kept_most() {
                    kept.1 = true;
                } else {
                    KEPT_BYTES.fetch_sub(self.len, Ordering::Relaxed);
                }
            }
            if kept.1 {
                return;
            }
            // Given back: nothing of it is current next time. (The lock
            // on the pieces ends here: `copy_back` takes it while it holds
            // the snapshot's.)
            kept.0.clear();
        }
        if let Some(bytes) = self.before.lock().unwrap_or_else(|e| e.into_inner()).take() {
            give_back(bytes);
        }
    }

    /// The 64 KiB pieces (cut at multiples of 64 KiB of title memory, as
    /// [`Shadow::differing`] cuts them) that hold offsets `[from, to)` of
    /// the copy; `from < to`.
    fn pieces(&self, from: usize, to: usize) -> std::ops::Range<usize> {
        pieces_of(self.start as usize, from, to)
    }

    /// The offsets in the copy that piece `piece` holds.
    fn piece(&self, piece: usize) -> (usize, usize) {
        piece_at(self.start as usize, self.len as usize, piece)
    }

    /// The copy's bytes at offsets `[from, to)` change: a kept snapshot is
    /// not current there any more.
    fn changed(&self, from: usize, to: usize) {
        if !keeps() || from >= to {
            return;
        }
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        if kept.0.is_empty() {
            return;
        }
        for piece in self.pieces(from, to) {
            kept.0[piece] = false;
        }
    }

    /// Whether shaders of a submission not yet copied back may be writing
    /// the copy.
    pub(crate) fn writes_pending(&self) -> bool {
        !self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_empty()
    }

    /// Whether [`Shadow::update`] of `[address, address + len)` would leave
    /// title bytes behind: a range shaders may be writing holds a page the
    /// title wrote since the copy last took it. The answer comes from the
    /// write-watch journal alone, as the submission's one asking left it
    /// (`crate::watch::asked`); where the journal cannot say (memory not
    /// write-watched, a range not asked about) it is yes. A page the copy
    /// never took whole counts as written if the title ever wrote it.
    ///
    /// For submissions sent to the card without a wait
    /// (`crate::host::late`): the ranges are those of flights in the air,
    /// which are then settled before this refresh.
    pub(crate) fn stale_under_writes(&self, address: usize, len: usize) -> bool {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return false;
        }
        let busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|&&(a, e)| a < to && e > from)
            .map(|&(a, e)| (a.max(from), e.min(to)))
            .collect();
        if busy.is_empty() {
            return false;
        }
        if self.watched.load(Ordering::Relaxed) == NOT_WATCHED {
            return true;
        }
        let base = start / PAGE;
        let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        busy.into_iter()
            .any(|(a, e)| match crate::watch::asked(a, e) {
                Some((_, written)) => written.into_iter().any(|(page, write)| {
                    page >= base
                        && copied
                            .get(page - base)
                            .is_none_or(|at| at == 0 || write > at)
                }),
                None => true,
            })
    }

    /// Where [`Shadow::stale_under_writes`] says yes: puts the bytes the
    /// title changed into the copy, in the ranges shaders may be writing,
    /// and only those bytes, so that what the shaders write beside them
    /// stays (`crate::host::late_exact`). For each page of such a range
    /// that the title wrote since the copy last took it, title memory as
    /// it is now is compared with what the copy held when the range was
    /// given to the shaders: the copy's CPU side for a copy in the card's
    /// memory (the changed runs go up before the next submission, each a
    /// copy of its own bytes, `runs_left` of them at most), the snapshot
    /// for the others (the changed bytes go into the copy and into the
    /// snapshot, so the compare after the card ran takes them for the
    /// title's and not the shaders').
    ///
    /// Whether the caller must still settle the flights: the journal
    /// cannot say what was written, a snapshot is missing, or `runs_left`
    /// ran out. Pages done before that stay done.
    pub(crate) fn merge_under_writes(
        &self,
        address: usize,
        len: usize,
        runs_left: &mut usize,
    ) -> bool {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return false;
        }
        let busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|&&(a, e)| a < to && e > from)
            .map(|&(a, e)| (a.max(from), e.min(to)))
            .collect();
        if busy.is_empty() {
            return false;
        }
        if self.watched.load(Ordering::Relaxed) == NOT_WATCHED {
            return true;
        }
        let base = start / PAGE;
        let mut title = vec![0u8; PAGE];
        for (a, e) in busy {
            // As the refresh itself asks (`update_free`): the journal if
            // the submission's asking covered the range, else the kernel,
            // else the guards. The journal alone said nothing for copies
            // of view memory, which is guarded and not write-watched, and
            // those were all of run 99's cases: the flights were settled
            // 4-5 times a frame for "0 pages looked at".
            let Some((now, written)) = crate::watch::writes(a, e, true) else {
                return true;
            };
            let stale: Vec<usize> = {
                let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                written
                    .into_iter()
                    .filter(|&(page, write)| {
                        page >= base
                            && copied
                                .get(page - base)
                                .is_none_or(|at| at == 0 || write > at)
                    })
                    .map(|(page, _)| page)
                    .collect()
            };
            for page in stale {
                let (lo, hi) = ((page * PAGE).max(a), ((page + 1) * PAGE).min(e));
                if lo >= hi {
                    continue;
                }
                let n = hi - lo;
                let title = &mut title[..n];
                // The lock of the copy's reference (the snapshot, or the
                // CPU's side of a copy in the card's memory), from before
                // title memory is read until the title's bytes are in: with
                // the copy-back on its own thread ([`relay`]) that thread
                // puts what the card wrote into title memory and into the
                // reference under the same lock, and a page read before
                // such a merge and compared after it would take the card's
                // bytes for the title's.
                let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
                // A page that is not there is left as it is, and looked at
                // again next time.
                if !crate::host::read_into(lo as *const u8, title) {
                    continue;
                }
                MERGED[0].fetch_add(1, Ordering::Relaxed);
                let at = lo - start;
                if self.card.is_some() {
                    // SAFETY: inside the mapping of the copy's CPU side,
                    // which only the submitting thread writes.
                    let held = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                    let runs = changed_runs(title, held);
                    if runs.len() > *runs_left {
                        return true;
                    }
                    *runs_left -= runs.len();
                    for (r, m) in runs {
                        held[r..r + m].copy_from_slice(&title[r..r + m]);
                        self.card_changed(at + r, at + r + m);
                        MERGED[1].fetch_add(1, Ordering::Relaxed);
                        MERGED[2].fetch_add(m as u64, Ordering::Relaxed);
                    }
                } else {
                    let Some(snapshot) = before.as_mut() else {
                        return true;
                    };
                    // SAFETY: the snapshot's allocation holds the copy's
                    // bytes at the copy's offsets (`writing`), and these
                    // are in a declared range.
                    let held =
                        unsafe { std::slice::from_raw_parts_mut(snapshot.as_mut_ptr().add(at), n) };
                    for (r, m) in changed_runs(title, held) {
                        // SAFETY: inside the copy's mapping. The card may be
                        // writing the copy beside these bytes; they are
                        // written here and nothing around them.
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                title.as_ptr().add(r),
                                self.data.add(at + r),
                                m,
                            )
                        };
                        held[r..r + m].copy_from_slice(&title[r..r + m]);
                        MERGED[1].fetch_add(1, Ordering::Relaxed);
                        MERGED[2].fetch_add(m as u64, Ordering::Relaxed);
                    }
                }
                drop(before);
                // Taken, if the range holds all of the page that is in the
                // copy; else the rest of it is the refresh's, which notes
                // only whole pages, and the page is looked at again.
                if (page * PAGE).max(start) >= a && ((page + 1) * PAGE).min(end) <= e {
                    let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                    copied.set(page - base, now);
                }
            }
        }
        false
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        // In place: the card reads title memory itself.
        if self.place.is_some() {
            return 0;
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        if self.watched.load(Ordering::Relaxed) == NOT_WATCHED
            && self.has_gpu_owned.load(Ordering::Acquire)
        {
            let owned = self.gpu_owned.lock().unwrap_or_else(|e| e.into_inner());
            for g in owned.iter() {
                if g.start < to && g.end > from {
                    busy.push((g.start, g.end));
                }
            }
        }
        if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame).
            busy.sort_unstable();
            let (mut at, mut bytes) = (from, 0);
            for (a, e) in busy {
                if a > at {
                    bytes += self.update_free(at, a);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_free(at, to);
            }
            return bytes;
        }
        self.update_free(from, to)
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn update_free(&self, from: usize, to: usize) -> usize {
        let mode = self.watched.load(Ordering::Relaxed);
        if mode != NOT_WATCHED {
            let result = if mode == VIEW_GUARDED {
                crate::watch::writes_guarded(from, to).map(|(now, w)| (now, w, true))
            } else {
                crate::watch::writes_kind(from, to, true)
            };
            if let Some((now, written, is_view)) = result {
                let next = if is_view { VIEW_GUARDED } else { 1 };
                if mode != next {
                    self.watched.store(next, Ordering::Relaxed);
                }
                let copied = self.copy_stale(from, to, now, &written);
                self.refresh_check(from, to);
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
                }
                return copied;
            }
            self.watched.store(NOT_WATCHED, Ordering::Relaxed);
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::WholeBytes, (to - from) as u64);
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        self.copy(from, to);
        #[cfg(feature = "diagnostics")]
        crate::copystat::whole(from, to - from, started.elapsed().as_micros() as u64);
        to - from
    }

    /// Diagnostic (`VICTORY_REFRESH_CHECK=1`; off by default): after a
    /// refresh of `[from, to)` that trusted the write watch or the guards,
    /// the first 256 bytes of every 64 KiB block of it are read from title
    /// memory and compared with the copy. A block that differs on three
    /// refreshes in a row was changed with nothing to tell of it (a write
    /// through another view of its pages, as DirectStorage's page
    /// destinations make) and the card reads old bytes there: said with
    /// the copy and the address. One that differs once is a write of the
    /// title's that arrived between the refresh and the look.
    fn refresh_check(&self, from: usize, to: usize) {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| {
            std::env::var_os("VICTORY_REFRESH_CHECK").is_some_and(|v| v == "1")
                || crate::host::walls_trace()
        }) {
            return;
        }
        const SAMPLE: usize = 256;
        // Blocks found differing: how many refreshes in a row.
        static DIFFERING: Mutex<Option<std::collections::HashMap<usize, u32>>> = Mutex::new(None);
        static SAID: AtomicU64 = AtomicU64::new(0);
        let start = self.start as usize;
        let mut title = [0u8; SAMPLE];
        let mut at = from;
        while at < to {
            let n = SAMPLE.min(to - at);
            if crate::host::read_into(at as *const u8, &mut title[..n]) {
                // SAFETY: `[from, to)` lies inside the buffer (`update`),
                // so these bytes are inside the copy's mapping.
                let copy = unsafe {
                    std::slice::from_raw_parts(self.data.add(at - start).cast_const(), n)
                };
                let mut map = DIFFERING.lock().unwrap_or_else(|e| e.into_inner());
                let map = map.get_or_insert_with(Default::default);
                if copy == &title[..n] {
                    map.remove(&at);
                } else {
                    let times = map.entry(at).or_insert(0);
                    *times += 1;
                    if *times == 3 {
                        let said = SAID.fetch_add(1, Ordering::Relaxed) + 1;
                        if said <= 64 || said.is_power_of_two() {
                            tracing::warn!(
                                said,
                                copy = self.id,
                                copy_start = format_args!("{start:#x}"),
                                copy_bytes = self.len,
                                address = format_args!("{at:#x}"),
                                refreshed = format_args!("{from:#x}..{to:#x}"),
                                on_the_card = self.card.is_some(),
                                window = self.window,
                                "a host copy of title memory is stale after its refresh, three times in a row:                                  the memory changed with nothing to tell of it (VICTORY_REFRESH_CHECK=1)"
                            );
                        }
                    }
                }
            }
            // The first bytes of the next 64 KiB block.
            at = (at / PAGE + 1) * PAGE;
        }
    }

    /// Copies the pages of `[from, to)` written after their last copy
    /// (`written`: last writes by page, in page order; `now`: this
    /// asking), whole pages within the buffer.
    /// Refresh `[from, to)` from `written` pages while preserving any 4 KiB
    /// or sub-4 KiB `gpu_owned` intervals inside the same 64 KiB page.
    fn copy_stale(&self, from: usize, to: usize, now: u64, written: &[(usize, u64)]) -> usize {
        let base = self.start as usize / PAGE;
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let first_page = from / PAGE;
        let last = (to - 1) / PAGE;
        let gpu_ranges: Vec<GpuOwnedRange> = if self.has_gpu_owned.load(Ordering::Acquire) {
            let owned = self.gpu_owned.lock().unwrap_or_else(|e| e.into_inner());
            let first_g = owned.partition_point(|g| g.end <= from);
            let mut slice = Vec::new();
            for g in &owned[first_g..] {
                if g.start >= to {
                    break;
                }
                slice.push(*g);
            }
            slice
        } else {
            Vec::new()
        };
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let mut gpu_cursor = 0usize;
        let page_intersects_gpu = |page: usize, cursor: &mut usize| -> bool {
            if gpu_ranges.is_empty() {
                return false;
            }
            let p_lo = (page * PAGE).max(from);
            let p_hi = ((page + 1) * PAGE).min(to);
            while *cursor < gpu_ranges.len() && gpu_ranges[*cursor].end <= p_lo {
                *cursor += 1;
            }
            *cursor < gpu_ranges.len() && gpu_ranges[*cursor].start < p_hi
        };

        // Fast path: once every page of the copy has been initialized at least
        // once, skip untouched pages completely — O(1) when `written` is empty
        // and O(|written|) otherwise (instead of O((to - from) / PAGE)).
        if copied.all_copied() {
            if from <= start && to >= end {
                let cold_at = self.cold_copied_asking.load(Ordering::Acquire);
                if cold_at != 0 && now > cold_at {
                    self.cold_copied_asking.store(0, Ordering::Release);
                }
            }
            if written.is_empty() {
                return 0;
            }
            let mut bytes = 0;
            let mut run: Option<(usize, usize)> = None;
            for &(page, write) in written {
                if page < first_page || page > last {
                    continue;
                }
                let at = copied.at[page - base];
                if write <= at {
                    continue;
                }
                if page_intersects_gpu(page, &mut gpu_cursor) {
                    if let Some((run_start, run_end)) = run.take() {
                        bytes += self.copy((run_start * PAGE).max(from), (run_end * PAGE).min(to));
                    }
                    if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                        copied.set(page - base, now);
                    }
                    bytes += self.copy_page_preserving_gpu_owned(
                        page,
                        from,
                        to,
                        now,
                        at,
                        &gpu_ranges[gpu_cursor..],
                    );
                    continue;
                }
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    copied.set(page - base, now);
                }
                match run {
                    Some((_, run_end)) if page < run_end => {}
                    Some((run_start, run_end)) if page == run_end => {
                        run = Some((run_start, page + 1));
                    }
                    Some((run_start, run_end)) => {
                        bytes += self.copy((run_start * PAGE).max(from), (run_end * PAGE).min(to));
                        run = Some((page, page + 1));
                    }
                    None => run = Some((page, page + 1)),
                }
            }
            if let Some((run_start, run_end)) = run {
                bytes += self.copy((run_start * PAGE).max(from), (run_end * PAGE).min(to));
            }
            return bytes;
        }

        let mut writes = written.iter().peekable();
        let (mut bytes, mut run): (usize, Option<usize>) = (0, None);
        let mut had_cold_copy = false;
        for page in first_page..=last {
            while writes.peek().is_some_and(|&&(p, _)| p < page) {
                writes.next();
            }
            let write = match writes.peek() {
                Some(&&(p, w)) if p == page => {
                    writes.next();
                    w
                }
                _ => 0,
            };
            let at = copied.at[page - base];
            if at == 0 || write > at {
                if at == 0 {
                    had_cold_copy = true;
                }
                if page_intersects_gpu(page, &mut gpu_cursor) {
                    if let Some(first) = run.take() {
                        bytes += self.copy((first * PAGE).max(from), (page * PAGE).min(to));
                    }
                    if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                        copied.set(page - base, now);
                    }
                    bytes += self.copy_page_preserving_gpu_owned(
                        page,
                        from,
                        to,
                        now,
                        at,
                        &gpu_ranges[gpu_cursor..],
                    );
                    continue;
                }
                // A page only partly in the range is copied in part: it
                // stays stale for the rest.
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    copied.set(page - base, now);
                }
                run.get_or_insert(page);
            } else if let Some(first) = run.take() {
                bytes += self.copy((first * PAGE).max(from), (page * PAGE).min(to));
            }
        }
        // Cut to the range: next to it shaders may be writing the copy.
        if let Some(first) = run {
            bytes += self.copy((first * PAGE).max(from), ((last + 1) * PAGE).min(to));
        }
        if had_cold_copy {
            self.cold_copied_asking.fetch_max(now, Ordering::Release);
        }
        bytes
    }

    /// Copies CPU-dirty 4 KiB sub-intervals of `page` while skipping any bytes
    /// actively owned by the GPU (unless the CPU freshly wrote that 4 KiB page
    /// strictly after `watch_seq`, which transfers ownership back to the CPU).
    fn copy_page_preserving_gpu_owned(
        &self,
        page: usize,
        from: usize,
        to: usize,
        now: u64,
        copied_at: u64,
        gpu_ranges: &[GpuOwnedRange],
    ) -> usize {
        let t0 = std::time::Instant::now();
        crate::authority::record_partial_page_split();
        let page_lo = (page * PAGE).max(from);
        let page_hi = ((page + 1) * PAGE).min(to);
        if page_lo >= page_hi {
            return 0;
        }
        // Emits contiguous non-GPU-owned sub-spans of sorted `gpu_ranges` directly with zero Vec allocs.
        let mut copy_run: Option<(usize, usize)> = None;
        let mut bytes = 0usize;
        let emit_copy =
            |a: usize, b: usize, copy_run: &mut Option<(usize, usize)>, bytes: &mut usize| {
                if a >= b {
                    return;
                }
                match *copy_run {
                    Some((rl, rh)) if rh == a => *copy_run = Some((rl, b)),
                    Some((rl, rh)) => {
                        *bytes += self.copy(rl, rh);
                        *copy_run = Some((a, b));
                    }
                    None => *copy_run = Some((a, b)),
                }
            };
        let emit_outside_gpu =
            |lo: usize, hi: usize, copy_run: &mut Option<(usize, usize)>, bytes: &mut usize| {
                let mut cur = lo;
                for g in gpu_ranges {
                    if g.start >= hi {
                        break;
                    }
                    if g.end <= cur {
                        continue;
                    }
                    if cur < g.start {
                        emit_copy(cur, g.start, copy_run, bytes);
                    }
                    cur = cur.max(g.end);
                    if cur >= hi {
                        return;
                    }
                }
                if cur < hi {
                    emit_copy(cur, hi, copy_run, bytes);
                }
            };

        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            let page_4k = crate::watch::journal_page_4k(page);
            let kfirst = page_lo / crate::watch::SUBPAGE;
            let klast = page_hi.div_ceil(crate::watch::SUBPAGE);
            let mut kmap = self.copied_4k.lock().unwrap_or_else(|e| e.into_inner());
            let mut demotes: [(usize, usize); 16] = [(0, 0); 16];
            let mut demote_len = 0usize;
            for kpage in kfirst..klast {
                let s_lo = (kpage * crate::watch::SUBPAGE).max(page_lo);
                let s_hi = ((kpage + 1) * crate::watch::SUBPAGE).min(page_hi);
                if s_lo >= s_hi {
                    continue;
                }
                let w_4k = page_4k[kpage % 16];
                let prev_at = kmap.get(&kpage).copied().unwrap_or(copied_at);
                let mut max_gpu_seq: Option<u64> = None;
                for g in gpu_ranges {
                    if g.start >= s_hi {
                        break;
                    }
                    if s_lo < g.end {
                        max_gpu_seq = Some(max_gpu_seq.map_or(g.watch_seq, |m| m.max(g.watch_seq)));
                    }
                }
                match max_gpu_seq {
                    None => {
                        if prev_at == 0 || w_4k > prev_at {
                            emit_copy(s_lo, s_hi, &mut copy_run, &mut bytes);
                            kmap.insert(kpage, now);
                        }
                    }
                    Some(gpu_seq) => {
                        if w_4k > gpu_seq.max(prev_at) {
                            // CPU wrote inside this 4 KiB page strictly after GPU write.
                            if demote_len > 0 && demotes[demote_len - 1].1 == s_lo {
                                demotes[demote_len - 1].1 = s_hi;
                            } else if demote_len < demotes.len() {
                                demotes[demote_len] = (s_lo, s_hi);
                                demote_len += 1;
                            }
                            emit_copy(s_lo, s_hi, &mut copy_run, &mut bytes);
                            kmap.insert(kpage, now);
                        } else if prev_at == 0 || w_4k > prev_at {
                            // Copy only adjacent bytes on the same 4 KiB page outside `[ga, ge)`.
                            emit_outside_gpu(s_lo, s_hi, &mut copy_run, &mut bytes);
                            kmap.insert(kpage, now);
                        }
                    }
                }
            }
            drop(kmap);
            if let Some((rl, rh)) = copy_run.take() {
                bytes += self.copy(rl, rh);
            }
            for &(d_lo, d_hi) in &demotes[..demote_len] {
                crate::authority::observe_cpu_write(d_lo, d_hi - d_lo);
                self.remove_gpu_owned_interval(d_lo, d_hi);
            }
            crate::rea_telemetry::record_prep_phase_us(
                crate::rea_telemetry::PrepPhase::SubpageSplit,
                t0.elapsed().as_micros() as u64,
            );
            return bytes;
        }
        emit_outside_gpu(page_lo, page_hi, &mut copy_run, &mut bytes);
        if let Some((rl, rh)) = copy_run {
            bytes += self.copy(rl, rh);
        }
        crate::rea_telemetry::record_prep_phase_us(
            crate::rea_telemetry::PrepPhase::SubpageSplit,
            t0.elapsed().as_micros() as u64,
        );
        bytes
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(&self, from: usize, to: usize) -> usize {
        #[cfg(test)]
        COPY_CALLS.fetch_add(1, Ordering::Relaxed);
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        crate::host::copy_mapped(at, from as *const u8, to - from, false);
        // After the bytes are in: a snapshot taken meanwhile is not current.
        self.changed(from - start, to - start);
        self.card_changed(from - start, to - start);
        to - from
    }

    /// Whether shaders read and write this copy in the card's own memory
    /// ([`cards`]).
    pub(crate) fn carded(&self) -> bool {
        self.card.is_some()
    }

    /// The CPU side's bytes at offsets `[from, to)` changed: they go to the
    /// card's side before the next submission.
    fn card_changed(&self, from: usize, to: usize) {
        let Some(card) = &self.card else {
            return;
        };
        if from >= to {
            return;
        }
        let mut dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
        match dirty.last_mut() {
            Some(last) if from <= last.1 && to >= last.0 => {
                *last = (last.0.min(from), last.1.max(to))
            }
            _ => dirty.push((from, to)),
        }
    }

    /// Of a copy in the card's memory: the CPU's side (PC memory), the
    /// card side's bytes from before if made yet, and their size.
    pub(crate) fn card_sides(&self) -> Option<(&ID3D12Resource, Option<ID3D12Resource>, u64)> {
        let card = self.card.as_ref()?;
        let before = card
            .before
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        Some((&card.staging, before, card.size))
    }

    /// Shader writes are declared for a copy in the card's memory: makes
    /// its "before" twin with `make` (bytes to buffer) if it has none.
    /// Whether one was made now: the caller fills it, whole, from the
    /// card's side before the submission's lists run. Without one (no
    /// memory) every piece of a declared range is fetched.
    pub(crate) fn card_make_before(
        &self,
        make: impl FnOnce(u64) -> Option<ID3D12Resource>,
    ) -> bool {
        let Some(card) = &self.card else {
            return false;
        };
        let mut before = card.before.lock().unwrap_or_else(|e| e.into_inner());
        if before.is_some() {
            return false;
        }
        *before = make(card.size);
        if before.is_some() {
            CARD_BYTES[1].fetch_add(card_hash_bytes(card.size), Ordering::Relaxed);
        }
        before.is_some()
    }

    /// The declared range `[address, end)` as the compare on the card
    /// takes it; `None` if none of it lies in the copy.
    pub(crate) fn card_span(&self, address: usize, end: usize) -> Option<CardSpan> {
        let (from, to) = self.inside(address, end.saturating_sub(address));
        (from < to).then(|| card_span_of(self.start as usize, from, to))
    }

    /// The pieces that hold offsets `[from, to)` of the copy (cut to it),
    /// as the card's hashing takes them.
    pub(crate) fn card_run(&self, from: usize, to: usize) -> CardSpan {
        let to = to.min(self.len as usize);
        card_span_of(self.start as usize, from.min(to.saturating_sub(1)), to)
    }

    /// The offsets of piece `i` of `span` that lie in the span.
    pub(crate) fn card_piece(&self, span: &CardSpan, i: u32) -> (usize, usize) {
        card_piece_of(self.start as usize, self.len as usize, span, i)
    }

    /// The walls trace (`crate::host::walls_trace`): a piece of `fetched`
    /// bytes of this copy came back from the card and `written` of them
    /// went into title memory. Summed per buffer and logged once a frame:
    /// which shader-written buffers the title's CPU could see fresh in
    /// that frame, and how much of each changed.
    pub(crate) fn walls_merged(&self, fetched: usize, written: usize) {
        if !crate::host::walls_trace() {
            return;
        }
        static SEEN: Mutex<(u64, BTreeMap<u64, (u64, u32, u64, u64)>)> =
            Mutex::new((0, BTreeMap::new()));
        let frame = crate::frame::current();
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        if seen.0 != frame {
            let buffers: Vec<String> = seen
                .1
                .iter()
                .map(|(at, v)| {
                    format!(
                        "{at:#x}+{:#x}: {} pieces, {} fetched, {} written",
                        v.0, v.1, v.2, v.3
                    )
                })
                .take(32)
                .collect();
            if !buffers.is_empty() {
                tracing::info!(
                    frame = seen.0,
                    count = seen.1.len(),
                    buffers = ?buffers,
                    "walls trace: shader-written card buffers merged into title memory in this frame"
                );
            }
            seen.1.clear();
            seen.0 = frame;
        }
        let entry = seen.1.entry(self.start).or_insert((self.len, 0, 0, 0));
        entry.1 += 1;
        entry.2 += fetched as u64;
        entry.3 += written as u64;
    }

    /// After a submission, for a copy in the card's memory: `gpu` is what
    /// the card's side holds at offset `at` on, in a piece the compare
    /// found differing. The bytes shaders wrote (those that differ from the
    /// CPU's side, which is what the card's side held when they started)
    /// go into title memory, from the first to the last of them, as
    /// [`Shadow::copy_back`] puts them; the CPU's side takes the result and
    /// sends it up before the next submission. `title` is scratch memory.
    /// Returns the bytes written to title memory.
    pub(crate) fn card_merge(&self, at: usize, gpu: &[u8], title: &mut Vec<u8>) -> usize {
        let start = self.start as usize;
        if self.card.is_none() || at + gpu.len() > self.len as usize {
            return 0;
        }
        let began = std::time::Instant::now();
        // SAFETY: inside the mapping (checked above); only the submitting
        // thread writes it.
        let old = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), gpu.len()) };
        let Some(first) = first_difference(gpu, old) else {
            BACK_TIMING_US[2].fetch_add(began.elapsed().as_micros() as u64, Ordering::Relaxed);
            return 0;
        };
        let last = last_difference(gpu, old).unwrap_or(first);
        let (at, n) = (at + first, last + 1 - first);
        let (gpu, old) = (&gpu[first..=last], &mut old[first..=last]);
        title.clear();
        title.resize(n, 0);
        let mut written = 0;
        if crate::host::read_into((start + at) as *const u8, title) {
            // As in `copy_back`: the card's byte where it is not what it
            // was before the submission, else the title's.
            let mut differs = 0u8;
            for ((t, g), o) in title.iter_mut().zip(gpu).zip(old.iter_mut()) {
                let wrote = if *g != *o { 0xffu8 } else { 0 };
                let merged = (*t & !wrote) | (*g & wrote);
                differs |= merged ^ *t;
                *t = merged;
                *o = merged;
            }
            if differs != 0 {
                crate::watch::writing(start + at, n);
                crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
                written = n;
            }
        } else {
            // Title memory is not mapped there: the copy keeps what the
            // card wrote, as before.
            old.copy_from_slice(gpu);
        }
        // The card's side and its "before" take the result.
        self.card_changed(at, at + n);
        BACK_TIMING_US[2].fetch_add(began.elapsed().as_micros() as u64, Ordering::Relaxed);
        written
    }

    /// [`relay`], for a copy in the card's memory: `gpu` is what the
    /// card's side held at offset `at` on when a fetch sent after send
    /// `fetched_after` (`crate::host`'s order of the queue) ran, in a piece
    /// the compare found differing. The bytes that differ from the CPU's
    /// side are what shaders wrote; they go into title memory, from the
    /// first to the last of them, and into the CPU's side. Nothing is sent
    /// up afterwards (the card holds these bytes or newer ones: sending
    /// them up could undo a later flight's write), and the title's other
    /// bytes are left to the refresh.
    ///
    /// Left out: bytes of the CPU's side that hold title bytes the card
    /// had not received when the fetch ran, those still to go up and those
    /// that went up with a later send. There the card's side is the older
    /// one, and taking it for a shader's write would put it over what the
    /// title wrote. Under the CPU side's lock (the snapshot's, which such
    /// a copy has no other use for), as [`Shadow::merge_under_writes`].
    /// `title` is scratch memory. Returns the bytes written to title
    /// memory.
    pub(crate) fn card_merge_relay(
        &self,
        at: usize,
        gpu: &[u8],
        fetched_after: u64,
        title: &mut Vec<u8>,
    ) -> usize {
        let start = self.start as usize;
        let Some(card) = &self.card else {
            return 0;
        };
        if at + gpu.len() > self.len as usize {
            return 0;
        }
        let began = std::time::Instant::now();
        let _side = self.before.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: inside the mapping (checked above). The offsets are in a
        // range declared and not done with, which no refresh writes; what
        // does write it (the title's bytes put in under shader writes)
        // holds the lock held here.
        let held = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), gpu.len()) };
        let Some(first) = first_difference(gpu, held) else {
            BACK_TIMING_US[2].fetch_add(began.elapsed().as_micros() as u64, Ordering::Relaxed);
            return 0;
        };
        let last = last_difference(gpu, held).unwrap_or(first);
        let (at, n) = (at + first, last + 1 - first);
        let (gpu, held) = (&gpu[first..=last], &mut held[first..=last]);
        let skip: Vec<(usize, usize)> = {
            let dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
            let sent = card.sent.lock().unwrap_or_else(|e| e.into_inner());
            dirty
                .iter()
                .copied()
                .chain(
                    sent.iter()
                        .filter(|s| s.2 > fetched_after)
                        .map(|s| (s.0, s.1)),
                )
                .filter(|&(a, b)| a < at + n && b > at)
                .collect()
        };
        title.clear();
        title.resize(n, 0);
        let mapped = crate::host::read_into((start + at) as *const u8, title);
        let mut changed = false;
        for (a, b) in outside(at, at + n, skip) {
            let (a, b) = (a - at, b - at);
            // Title memory that is not mapped: the CPU's side keeps what
            // the card wrote, as before; `title` is scratch then.
            changed |= merge_piece(&gpu[a..b], &mut held[a..b], &mut title[a..b]);
        }
        if !mapped || !changed {
            BACK_TIMING_US[2].fetch_add(began.elapsed().as_micros() as u64, Ordering::Relaxed);
            return 0;
        }
        crate::watch::writing(start + at, n);
        crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
        BACK_TIMING_US[2].fetch_add(began.elapsed().as_micros() as u64, Ordering::Relaxed);
        n
    }
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: RwLock<BTreeMap<(u64, usize), Entry>> = RwLock::new(BTreeMap::new());

fn buffers_read() -> std::sync::RwLockReadGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.read().unwrap_or_else(|e| e.into_inner())
}

fn buffers_write() -> std::sync::RwLockWriteGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.write().unwrap_or_else(|e| e.into_inner())
}

#[repr(align(64))]
struct AlignedAtomicU64(AtomicU64);

static SHADOW_EPOCH: AlignedAtomicU64 = AlignedAtomicU64(AtomicU64::new(1));
static SHADOW_TLS_HITS: AlignedAtomicU64 = AlignedAtomicU64(AtomicU64::new(0));
static SHADOW_TLS_MISSES: AlignedAtomicU64 = AlignedAtomicU64(AtomicU64::new(0));

#[inline]
fn bump_shadow_epoch() {
    SHADOW_EPOCH.0.fetch_add(1, Ordering::Release);
}

/// Returns `(hits, misses)` for the per-thread shadow lookup cache
/// (`VICTORY_REA_GPU_FASTPATH=1`).
pub fn tls_cache_stats() -> (u64, u64) {
    TLS_SHADOW_CACHE.with(|cell| {
        if let Ok(mut cache) = cell.try_borrow_mut() {
            cache.flush_stats();
        }
    });
    (
        SHADOW_TLS_HITS.0.load(Ordering::Relaxed),
        SHADOW_TLS_MISSES.0.load(Ordering::Relaxed),
    )
}

#[derive(Clone, Default)]
struct TlsShadowEntry {
    epoch: u64,
    safe_start: u64,
    safe_end: u64,
    exact_addr: u64,
    exact_end: u64,
    any: bool,
    is_window: bool,
    range_safe: bool,
    shadow: std::sync::Weak<Shadow>,
}

struct TlsShadowCache {
    entries: [TlsShadowEntry; 16],
    next: usize,
    pending_hits: u32,
    pending_misses: u32,
}

impl Default for TlsShadowCache {
    fn default() -> Self {
        Self {
            entries: std::array::from_fn(|_| TlsShadowEntry::default()),
            next: 0,
            pending_hits: 0,
            pending_misses: 0,
        }
    }
}

impl TlsShadowCache {
    #[inline]
    fn flush_stats(&mut self) {
        if self.pending_hits > 0 {
            SHADOW_TLS_HITS
                .0
                .fetch_add(u64::from(self.pending_hits), Ordering::Relaxed);
            self.pending_hits = 0;
        }
        if self.pending_misses > 0 {
            SHADOW_TLS_MISSES
                .0
                .fetch_add(u64::from(self.pending_misses), Ordering::Relaxed);
            self.pending_misses = 0;
        }
    }
}

impl Drop for TlsShadowCache {
    fn drop(&mut self) {
        self.flush_stats();
    }
}

thread_local! {
    static TLS_SHADOW_CACHE: std::cell::RefCell<TlsShadowCache> =
        std::cell::RefCell::new(TlsShadowCache::default());
}

fn tls_cache_lookup(address: u64, end: u64, unsized_ask: bool, any: bool) -> Option<Arc<Shadow>> {
    let cur_epoch = SHADOW_EPOCH.0.load(Ordering::Acquire);
    let found = TLS_SHADOW_CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();
        let mut hit = None;
        for idx in 0..cache.entries.len() {
            let e = &cache.entries[idx];
            if e.epoch != cur_epoch {
                continue;
            }
            let matched = if e.range_safe && !e.is_window {
                address >= e.safe_start && end <= e.safe_end
            } else {
                address == e.exact_addr && end == e.exact_end && (!e.is_window || e.any == any)
            };
            if matched {
                if let Some(arc) = e.shadow.upgrade() {
                    if idx > 0 {
                        cache.entries[..=idx].rotate_right(1);
                    }
                    hit = Some(arc);
                } else {
                    cache.entries[idx].epoch = 0;
                }
                break;
            }
        }
        if hit.is_some() {
            cache.pending_hits += 1;
        } else {
            cache.pending_misses += 1;
        }
        if cache.pending_hits + cache.pending_misses >= 256 {
            cache.flush_stats();
        }
        hit
    });
    if let Some(arc) = found {
        if arc.window {
            arc.used.store(crate::host::moment(), Ordering::Relaxed);
            arc.asked[0].fetch_min(address, Ordering::Relaxed);
            arc.asked[1].fetch_max(end, Ordering::Relaxed);
            if unsized_ask {
                arc.unsized_asks.fetch_add(1, Ordering::Relaxed);
            }
        }
        Some(arc)
    } else {
        None
    }
}

#[allow(clippy::too_many_arguments)]
fn tls_cache_insert(
    epoch: u64,
    safe_start: u64,
    safe_end: u64,
    exact_addr: u64,
    exact_end: u64,
    any: bool,
    is_window: bool,
    range_safe: bool,
    shadow: &Arc<Shadow>,
) {
    TLS_SHADOW_CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();
        let slot = cache.next % cache.entries.len();
        cache.next = slot + 1;
        cache.entries[slot] = TlsShadowEntry {
            epoch,
            safe_start,
            safe_end,
            exact_addr,
            exact_end,
            any,
            is_window,
            range_safe,
            shadow: Arc::downgrade(shadow),
        };
    });
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    let mut map = buffers_write();
    map.insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
    bump_shadow_epoch();
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    let mut map = buffers_write();
    if map.remove(&(start, resource)).is_some() {
        bump_shadow_epoch();
    }
}

/// The title buffer holding `address`, as (start, length), without making
/// its copy (diagnostics: which buffer a dropped program's root view names).
#[cfg(feature = "diagnostics")]
pub(crate) fn extent(address: u64) -> Option<(u64, u64)> {
    buffers_read()
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end > address)
        .map(|(k, e)| (k.0, e.end - k.0))
}

/// Title memory `[at, at + len)` is about to show something else or be
/// given up: what is in place there (`crate::inplace`) is taken out of the
/// tables, so that the next ask makes a copy or opens the memory anew, and
/// goes when no list and no slot holds it any more. Returns how many.
pub(crate) fn in_place_let_go(at: u64, len: u64) -> u64 {
    let end = at.saturating_add(len);
    let mut gone: Vec<Arc<Shadow>> = Vec::new();
    {
        let mut map = LOOSE.write().unwrap_or_else(|e| e.into_inner());
        let keys: Vec<(u64, u64)> = map
            .range((at.saturating_sub(WINDOW), 0)..(end, 0))
            .filter(|(key, shadow)| shadow.place.is_some() && key.0 + key.1 > at)
            .map(|(key, _)| *key)
            .collect();
        let removed: Vec<_> = keys
            .into_iter()
            .filter_map(|key| map.remove(&key))
            .collect();
        if !removed.is_empty() {
            bump_shadow_epoch();
        }
        gone.extend(removed);
    }
    {
        // As `lookup` finds a buffer: among the 256 that start below.
        let mut map = buffers_write();
        let mut took = false;
        for (_, entry) in map.range_mut(..(end, 0)).rev().take(256) {
            if entry.end > at && entry.shadow.as_ref().is_some_and(|s| s.place.is_some()) {
                gone.extend(entry.shadow.take());
                took = true;
            }
        }
        if took {
            bump_shadow_epoch();
        }
    }
    let n = gone.len() as u64;
    for shadow in gone {
        crate::host::retire(shadow);
    }
    n
}

/// For the in-place census (`crate::inplace`): the title buffers (start,
/// bytes, false) and the windows outside buffers that have a copy (start,
/// bytes, true).
pub(crate) fn spans() -> Vec<(u64, u64, bool)> {
    let mut spans: Vec<(u64, u64, bool)> = buffers_read()
        .iter()
        .map(|(key, entry)| (key.0, entry.end - key.0, false))
        .collect();
    let loose = LOOSE.read().unwrap_or_else(|e| e.into_inner());
    spans.extend(loose.keys().map(|&(start, len)| (start, len, true)));
    spans
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use. A `len` of 0 or 1 is an ask without a length
/// (a root view: the caller reads on to the copy's end).
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    lookup(address, len, false)
}

/// As [`find`], for an ask that is done with the copy when its list has
/// run (a draw's vertex and index buffers, indirect arguments): outside
/// buffer objects it takes a window that is there already and holds the
/// range ([`holder`]), and makes one of its own only if none does.
///
/// [`find`] names a window by the ask alone, which a descriptor slot needs
/// (the slot shows the window its view found when it was made, and the
/// pass that brings it up to date must find the same one). A draw needs no
/// such thing: it binds the copy it found and brings that copy up to date.
/// Named by the ask alone, every draw's range was a window of its own
/// wherever the windows are cut finer than 4 MiB: draws that bind
/// different ranges of the same memory then each copied their range, and
/// windows nothing asked for again for ten seconds were let go and made
/// (and copied) anew. Run 71, on a 256 KiB grid: 350 MB copied per frame
/// against 120 MB, 7,087 write-watch askings against 2,600, and 5,263
/// windows (4.5 GB) let go in two minutes against 393 in a whole run.
pub(crate) fn find_any(address: u64, len: u64) -> Option<Arc<Shadow>> {
    // With whole windows (the default) there is one window to a cell and
    // nothing to choose: exactly `find`.
    if !fine() {
        return find(address, len);
    }
    lookup(address, len, true)
}

fn lookup(address: u64, len: u64, any: bool) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    let fastpath = crate::descriptor::host::gpu_fastpath_enabled();
    if fastpath && let Some(cached) = tls_cache_lookup(address, end, len <= 1, any) {
        return Some(cached);
    }
    let cur_epoch = SHADOW_EPOCH.0.load(Ordering::Acquire);
    // Shared-read fast path: 99.99% of calls either hit an already-created
    // buffer shadow or miss into `loose()`, so parallel recording threads never
    // serialize on an exclusive lock.
    {
        let map = buffers_read();
        let found = map
            .range(..=(address, usize::MAX))
            .rev()
            .take(64)
            .find(|(_, e)| e.end >= end);
        match found {
            None => {
                drop(map);
                return loose(address, end, len <= 1, any);
            }
            Some((k, entry)) if entry.shadow.is_some() => {
                let res = entry.shadow.clone();
                if fastpath && let Some(ref arc) = res {
                    let next_start = map
                        .range((std::ops::Bound::Excluded(*k), std::ops::Bound::Unbounded))
                        .next()
                        .map(|(nk, _)| nk.0)
                        .unwrap_or(u64::MAX);
                    let safe_end = entry.end.min(next_start);
                    let range_safe = address >= k.0 && end <= safe_end;
                    tls_cache_insert(
                        cur_epoch, k.0, safe_end, address, end, any, false, range_safe, arc,
                    );
                }
                return res;
            }
            Some(_) => {}
        }
    }
    let mut map = buffers_write();
    let key = map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k);
    let Some(key) = key else {
        drop(map);
        return loose(address, end, len <= 1, any);
    };
    let next_start = map
        .range((std::ops::Bound::Excluded(key), std::ops::Bound::Unbounded))
        .next()
        .map(|(nk, _)| nk.0)
        .unwrap_or(u64::MAX);
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let size = len.next_multiple_of(1 << 16);
        // In place, if asked for and the memory can be opened
        // (`crate::inplace::place_for`): the buffer is the title's own
        // memory. Not the big ones `cards` puts in the card's memory: those
        // are read too often to stay in the PC's.
        let big = cards() && len >= card_limits().0;
        if !big && let Some(placed) = crate::inplace::place_for(key.0, len, false) {
            entry.shadow = Some(Arc::new(Shadow::placed(placed, key.0, len, false)));
            bump_shadow_epoch();
            return entry.shadow.clone();
        }
        if !big
            && let Some(shadow) = rebar_copy(crate::rebar::Kind::Buffer, key.0, len, size, false)
        {
            entry.shadow = Some(shadow);
            bump_shadow_epoch();
            return entry.shadow.clone();
        }
        let (buffer, data) = host()?.shared_buffer(size)?;
        crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
        // A big one, with `cards`: shaders get a buffer in the card's own
        // memory, and the one just made is the CPU's side.
        let (buffer, card) = match card_for(len, size) {
            Some(on_card) => {
                crate::host::name(&on_card, || {
                    format!("title buffer {:#x}+{len:#x} on the card", key.0)
                });
                tracing::info!(
                    start = format_args!("{:#x}", key.0),
                    len,
                    "a buffer's host copy is in the card's memory"
                );
                let card = Card {
                    staging: buffer,
                    before: Mutex::default(),
                    dirty: Mutex::default(),
                    sent: Mutex::default(),
                    size,
                };
                (on_card, Some(card))
            }
            None => (buffer, None),
        };
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        let mut shadow = Shadow::new(buffer, data, gpu, key.0, len);
        let is_card = card.is_some();
        shadow.card = card;
        shadow.set_live_class(if is_card {
            0
        } else if len >= card_limits().0 {
            1
        } else {
            2
        });
        let shadow = Arc::new(shadow);
        if shadow.card.is_some() {
            CARDED
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(Arc::downgrade(&shadow));
        }
        entry.shadow = Some(shadow);
        bump_shadow_epoch();
        let safe_end = entry.end.min(next_start);
        if fastpath && let Some(ref arc) = entry.shadow {
            let new_epoch = SHADOW_EPOCH.0.load(Ordering::Acquire);
            let range_safe = address >= key.0 && end <= safe_end;
            tls_cache_insert(
                new_epoch, key.0, safe_end, address, end, any, false, range_safe, arc,
            );
        }
    }
    entry.shadow.clone()
}

/// A host copy in CPU-visible card memory (`D3D12_HEAP_TYPE_GPU_UPLOAD`,
/// `crate::rebar`), if enabled, supported and within budget.
fn rebar_copy(
    kind: crate::rebar::Kind,
    start: u64,
    len: u64,
    size: u64,
    window: bool,
) -> Option<Arc<Shadow>> {
    // Small title buffers (`Kind::Buffer`) are snapshotted and diffed on the
    // CPU after shader writes whenever `VICTORY_COPY_BACK_ONLY_READBACK` is
    // off. Reading write-combined GPU upload memory over PCIe from the CPU
    // stalls copy-back severely (~80 ms/frame for 17 MB), so keep small
    // buffers in system RAM unless readback filtering is active.
    let cpu_read = crate::readback::tagged_in(start, len)
        || (kind == crate::rebar::Kind::Buffer && !crate::readback::enabled());
    if cpu_read
        && kind == crate::rebar::Kind::Buffer
        && !crate::readback::enabled()
        && crate::rebar::level() >= 2
    {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::info!(
                "Resizable BAR: keeping small title buffers in system RAM because VICTORY_COPY_BACK_ONLY_READBACK=0 (avoids uncached PCIe CPU readback stalls)"
            );
        });
    }
    if !crate::rebar::reserve(kind, size, cpu_read) {
        return None;
    }
    let Some((buffer, data)) = host()?.gpu_upload_buffer(size) else {
        crate::rebar::released(size);
        return None;
    };
    crate::host::name(&buffer, || {
        format!(
            "title {} {start:#x}+{len:#x} (rebar)",
            if window { "memory" } else { "buffer" }
        )
    });
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    let mut shadow = Shadow::new(buffer, data, gpu, start, len);
    shadow.window = window;
    shadow.rebar = size;
    shadow.set_live_class(if window {
        4
    } else if len >= card_limits().0 {
        1
    } else {
        2
    });
    Some(Arc::new(shadow))
}

/// Titles also use memory outside any buffer object as GPU memory (
/// the title's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
/// A window nothing names any more is let go ([`let_go_idle`]).
/// shortcut: a window is 4 MiB however little of it is read (the memory
/// line says how much was ever asked for, and how much was ever copied);
/// [`window`] cuts the windows of asks that give a length on a finer grid.
const WINDOW: u64 = 4 << 20;

/// The grid the windows of asks with a length are cut on, in bytes: 4 MiB
/// unless `VICTORY_WINDOW_KB` names a smaller power of two (64 at least).
/// A window is the PC's memory, all of it, however little of it is read:
/// 803 windows held 3.3 GB in run 60's gameplay, and all that was ever
/// asked for in them lay within 1.6 GB. On a grid of 1 MiB or 256 KiB a
/// window ends nearer to what is read. An ask without a length keeps the
/// 4 MiB grid whatever is set: a root view reads from its address to its
/// window's end, and mesh streams bound that way reach megabytes.
/// shortcut: the default stays 4 MiB until a run with the memory line's
/// window counts has shown what the finer grids give; the same memory
/// asked for with different ranges then lies in more than one window.
fn window() -> u64 {
    static GRID: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *GRID.get_or_init(|| {
        let kb = std::env::var("VICTORY_WINDOW_KB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok());
        grid(kb)
    })
}

/// The window grid in bytes for a setting in KiB: a power of two from
/// 64 KiB to 4 MiB, 4 MiB for anything else.
fn grid(kb: Option<u64>) -> u64 {
    match kb {
        Some(kb) if kb.is_power_of_two() && (64..=4096).contains(&kb) => kb << 10,
        _ => WINDOW,
    }
}

/// The window (start, bytes) of title memory for an ask of `[address,
/// end)` on a grid of `grid` bytes.
fn window_of(address: u64, end: u64, grid: u64) -> (u64, u64) {
    let start = address & !(grid - 1);
    (start, end.next_multiple_of(grid) - start)
}

/// Whether the window of an ask without a length begins at the ask instead
/// of at the 4 MiB line below it. Such an ask (a root view) reads from its
/// address to the end of its 4 MiB cell; what lies below the lowest
/// address ever asked for in the cell is memory the host zeroed and nothing
/// reads (513 of run 65's 782 windows were asked for that way, 2,052 MB;
/// of all windows' 3,236 MB, 1,988 MB were ever copied). With this on, the
/// window is the cell's tail from the ask's step on ([`TAIL_STEP`]); an ask
/// lower in the cell later makes a longer tail, which every ask from then
/// on takes, and the shorter one goes when nothing holds it any more.
/// shortcut: off unless `VICTORY_WINDOW_TAILS=1`. A longer tail is another
/// host buffer than the shorter one: what was bound to the shorter stays
/// right only because every list brings up to date the copy it bound
/// (`draw.rs` keeps a root view's copy with its slot). It pays only
/// together with `VICTORY_WINDOW_KB` below 4096: on the 4 MiB grid an ask
/// with a length in the same cell makes the whole cell's window, and the
/// tail is that.
fn tails() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WINDOW_TAILS").is_some_and(|v| v == "1"))
}

/// Tails begin on multiples of this. A longer tail is a new host buffer,
/// filled from title memory from its first byte to its last: begun on the
/// 64 KiB page of each ask, a cell whose asks came from the top down made
/// up to 63 of them, one after the other (part of run 71's copies). On
/// this step it is 15 at most, and a tail holds up to 192 KiB it did not
/// need.
const TAIL_STEP: u64 = 256 << 10;

/// The window (start, bytes) a tail ask at `address` makes: from the ask's
/// step ([`TAIL_STEP`]) to the end of its 4 MiB cell.
fn tail_of(address: u64, end: u64) -> (u64, u64) {
    let (cell, len) = window_of(address, end, WINDOW);
    let start = address & !(TAIL_STEP - 1);
    (start, cell + len - start)
}

/// Of `windows` (start, bytes), the one an ask of `[address, end)` that
/// may take any window takes ([`find_any`]): the longest that holds the
/// whole range.
fn holder(windows: impl Iterator<Item = (u64, u64)>, address: u64, end: u64) -> Option<(u64, u64)> {
    windows
        .filter(|&(start, bytes)| start <= address && start + bytes >= end)
        .max_by_key(|&(start, bytes)| (bytes, std::cmp::Reverse(start)))
}

/// Whether windows may be smaller than their 4 MiB cell (a finer grid, or
/// tails): only then is there more than one window to choose from.
fn fine() -> bool {
    tails() || window() < WINDOW
}

/// For the memory line: the bytes of title memory the windows alive lie
/// over (each byte once), beside the windows' own bytes in
/// [`memory`]: the difference is memory that is in more than one window,
/// which is copied once for each.
#[cfg(feature = "diagnostics")]
pub(crate) fn window_cover() -> u64 {
    let loose = LOOSE.read().unwrap_or_else(|e| e.into_inner());
    cover(loose.keys().copied())
}

/// The bytes `windows` (start, bytes; in order of start) lie over, each
/// byte once.
#[cfg_attr(not(any(test, feature = "diagnostics")), allow(dead_code))]
fn cover(windows: impl Iterator<Item = (u64, u64)>) -> u64 {
    let (mut covered, mut reached) = (0, 0);
    for (start, bytes) in windows {
        let end = start + bytes;
        if end > reached {
            covered += end - start.max(reached);
            reached = end;
        }
    }
    covered
}

/// Of `windows` (start, bytes; in order), the longest tail a tail ask at
/// `address` can take: the first that begins in the ask's cell at or below
/// the ask's page and ends where the cell ends.
fn longest_tail(
    windows: impl Iterator<Item = (u64, u64)>,
    address: u64,
    end: u64,
) -> Option<(u64, u64)> {
    let (cell, len) = window_of(address, end, WINDOW);
    let step = address & !(TAIL_STEP - 1);
    windows
        .filter(|&(start, bytes)| start >= cell && start <= step && start + bytes == cell + len)
        .min()
}

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: RwLock<BTreeMap<(u64, u64), Arc<Shadow>>> = RwLock::new(BTreeMap::new());

fn loose(address: u64, end: u64, unsized_ask: bool, any: bool) -> Option<Arc<Shadow>> {
    let (cell, _) = window_of(address, end, WINDOW);
    let named = window_of(address, end, if unsized_ask { WINDOW } else { window() });
    let resolve = |map: &BTreeMap<(u64, u64), Arc<Shadow>>| -> (u64, u64) {
        if unsized_ask && tails() {
            let there = map
                .range((cell, 0)..=(address, u64::MAX))
                .map(|(key, _)| *key);
            longest_tail(there, address, end).unwrap_or_else(|| tail_of(address, end))
        } else if any && fine() && !unsized_ask {
            // The windows that begin in the ask's cell at or below the ask (one
            // that begins in a cell below and reaches this far is passed over:
            // the ask then makes its own).
            let there = map
                .range((cell, 0)..=(address, u64::MAX))
                .map(|(key, _)| *key);
            holder(there, address, end).unwrap_or(named)
        } else {
            named
        }
    };
    // Under the lock (shared or exclusive), with the handing out: see `let_go_idle`
    // (which takes `LOOSE.write()`).
    let note = |shadow: &Shadow| {
        shadow.used.store(crate::host::moment(), Ordering::Relaxed);
        shadow.asked[0].fetch_min(address, Ordering::Relaxed);
        shadow.asked[1].fetch_max(end, Ordering::Relaxed);
        if unsized_ask {
            shadow.unsized_asks.fetch_add(1, Ordering::Relaxed);
        }
    };
    let fastpath = crate::descriptor::host::gpu_fastpath_enabled();
    let cur_epoch = SHADOW_EPOCH.0.load(Ordering::Acquire);
    // Shared-read fast path: 99.99% of lookups hit an existing window.
    {
        let map = LOOSE.read().unwrap_or_else(|e| e.into_inner());
        let (start, len) = resolve(&map);
        if let Some(shadow) = map.get(&(start, len)) {
            note(shadow);
            if fastpath {
                tls_cache_insert(
                    cur_epoch,
                    start,
                    start + len,
                    address,
                    end,
                    any,
                    true,
                    false,
                    shadow,
                );
            }
            return Some(shadow.clone());
        }
    }
    let mut map = LOOSE.write().unwrap_or_else(|e| e.into_inner());
    let (start, len) = resolve(&map);
    if let Some(shadow) = map.get(&(start, len)) {
        note(shadow);
        let cur_epoch = SHADOW_EPOCH.0.load(Ordering::Acquire);
        if fastpath {
            tls_cache_insert(
                cur_epoch,
                start,
                start + len,
                address,
                end,
                any,
                true,
                false,
                shadow,
            );
        }
        return Some(shadow.clone());
    }
    // In place, if asked for and the window's memory can be opened: the
    // window is the title's own memory, with nothing to copy.
    if let Some(placed) = crate::inplace::place_for(start, len, true) {
        let shadow = Arc::new(Shadow::placed(placed, start, len, true));
        note(&shadow);
        map.insert((start, len), shadow.clone());
        bump_shadow_epoch();
        return Some(shadow);
    }
    if let Some(shadow) = rebar_copy(crate::rebar::Kind::Window, start, len, len, true) {
        note(&shadow);
        map.insert((start, len), shadow.clone());
        bump_shadow_epoch();
        return Some(shadow);
    }
    let (buffer, data) = host()?.shared_buffer(len)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    // With room for windows on the card (`window_card_most`): shaders get a
    // buffer there, and the one just made is the CPU's side.
    let (buffer, card) = match window_card_for(len) {
        Some(on_card) => {
            crate::host::name(&on_card, || {
                format!("title memory {start:#x}+{len:#x} on the card")
            });
            let card = Card {
                staging: buffer,
                before: Mutex::default(),
                dirty: Mutex::default(),
                sent: Mutex::default(),
                size: len,
            };
            (on_card, Some(card))
        }
        None => (buffer, None),
    };
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let mut shadow = Shadow::new(buffer, data, gpu, start, len);
    shadow.window = true;
    let is_card = card.is_some();
    shadow.card = card;
    shadow.set_live_class(if is_card { 3 } else { 4 });
    let shadow = Arc::new(shadow);
    if shadow.card.is_some() {
        CARDED
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(Arc::downgrade(&shadow));
    }
    note(&shadow);
    map.insert((start, len), shadow.clone());
    bump_shadow_epoch();
    Some(shadow)
}

#[cfg(test)]
mod window_tests {
    use super::{WINDOW, cover, grid, holder, longest_tail, tail_of, window_of};

    /// An ask that may take any window takes the longest that holds all of
    /// it, and none that holds only a part.
    #[test]
    fn an_ask_takes_the_longest_window_that_holds_it() {
        let cell = 0xd_1200_0000u64;
        let (ask, end) = (cell + 0x9_0000, cell + 0xb_0000);
        assert_eq!(holder([].into_iter(), ask, end), None);
        let there = [
            (cell, 0x4_0000),             // ends below the ask
            (cell + 0x8_0000, 0x2_0000),  // holds its first half only
            (cell + 0x8_0000, 0x4_0000),  // holds it
            (cell + 0x4_0000, 0x10_0000), // holds it, longer
            (cell + 0xa_0000, 0x40_0000), // begins above it
        ];
        assert_eq!(
            holder(there.into_iter(), ask, end),
            Some((cell + 0x4_0000, 0x10_0000))
        );
        // Of two as long, the one that begins lower.
        let two = [(cell + 0x8_0000, 0x8_0000), (cell + 0x4_0000, 0x8_0000)];
        assert_eq!(
            holder(two.into_iter(), ask, end),
            Some((cell + 0x4_0000, 0x8_0000))
        );
    }

    /// What windows lie over counts each byte once.
    #[test]
    fn overlapping_windows_cover_their_memory_once() {
        assert_eq!(cover([].into_iter()), 0);
        let apart = [(0x10_0000u64, 0x4_0000u64), (0x20_0000, 0x4_0000)];
        assert_eq!(cover(apart.into_iter()), 0x8_0000);
        // One inside another, one overlapping its end, one touching.
        let over = [
            (0x10_0000u64, 0x10_0000u64),
            (0x14_0000, 0x4_0000),
            (0x1c_0000, 0x8_0000),
            (0x24_0000, 0x4_0000),
        ];
        assert_eq!(cover(over.into_iter()), 0x18_0000);
    }

    /// A tail runs from the ask's page to the end of its 4 MiB cell, and an
    /// ask takes the longest tail there is that holds it.
    #[test]
    fn tails_run_to_the_cells_end_and_the_longest_is_taken() {
        let cell = 0xd_1200_0000u64;
        let ask = cell + 0x12_3456;
        assert_eq!(
            tail_of(ask, ask + 1),
            (cell + 0x10_0000, WINDOW - 0x10_0000)
        );
        // The cell's first page: the whole cell, the window there always was.
        assert_eq!(tail_of(cell + 0xffff, cell + 0x1_0000), (cell, WINDOW));
        let end = cell + WINDOW;
        let tail = |from: u64| (cell + from, WINDOW - from);
        // Nothing there, or only tails that begin above the ask.
        assert_eq!(longest_tail([].into_iter(), ask, ask + 1), None);
        assert_eq!(
            longest_tail([tail(0x20_0000)].into_iter(), ask, ask + 1),
            None
        );
        // The longest of those that hold the ask; windows that do not reach
        // the cell's end (cut on a finer grid for asks with a length) and
        // windows of other cells are not tails.
        let there = [
            (cell - WINDOW, WINDOW),
            (cell, 0x4_0000),
            tail(0x8_0000),
            tail(0x10_0000),
            (cell + 0x10_0000, 0x4_0000),
            tail(0x20_0000),
        ];
        assert_eq!(
            longest_tail(there.into_iter(), ask, ask + 1),
            Some(tail(0x8_0000))
        );
        let (start, bytes) = tail(0x8_0000);
        assert!(start <= ask && start + bytes == end);
        // The whole cell's window is the longest tail of all.
        let with_whole = [(cell, WINDOW), tail(0x8_0000)];
        assert_eq!(
            longest_tail(with_whole.into_iter(), ask, ask + 1),
            Some((cell, WINDOW))
        );
    }

    /// The setting takes powers of two from 64 KiB to 4 MiB; anything else
    /// leaves the grid as it was.
    #[test]
    fn the_grid_is_a_power_of_two_or_the_default() {
        assert_eq!(grid(None), WINDOW);
        assert_eq!(grid(Some(4096)), WINDOW);
        assert_eq!(grid(Some(1024)), 1 << 20);
        assert_eq!(grid(Some(256)), 256 << 10);
        assert_eq!(grid(Some(64)), 64 << 10);
        for bad in [0, 32, 100, 1000, 8192] {
            assert_eq!(grid(Some(bad)), WINDOW, "{bad}");
        }
    }

    /// A window holds its ask and is cut on the grid; on the default grid
    /// it is the window there always was.
    #[test]
    fn windows_hold_their_ask_on_the_grid() {
        let (address, end) = (0xd_1234_5678u64, 0xd_1234_5678u64 + 0x9000);
        assert_eq!(window_of(address, end, WINDOW), (0xd_1200_0000, 4 << 20));
        assert_eq!(window_of(address, end, 1 << 20), (0xd_1230_0000, 1 << 20));
        assert_eq!(
            window_of(address, end, 256 << 10),
            (0xd_1234_0000, 256 << 10)
        );
        // An ask across a grid line takes both cells.
        let end = address + 0x4_0000;
        assert_eq!(
            window_of(address, end, 256 << 10),
            (0xd_1234_0000, 512 << 10)
        );
        for g in [64u64 << 10, 256 << 10, 1 << 20, WINDOW] {
            let (start, len) = window_of(address, end, g);
            assert!(start <= address && start + len >= end);
            assert!(start % g == 0 && len % g == 0);
        }
    }

    #[test]
    fn copied_pages_tracks_uncopied_count_and_fast_path_condition() {
        let mut pages = super::CopiedPages::new(4);
        assert!(!pages.all_copied());
        assert_eq!(pages.copied_count(), 0);
        pages.set(0, 10);
        pages.set(0, 20); // updating an already-copied page keeps count stable
        assert_eq!(pages.copied_count(), 1);
        assert!(!pages.all_copied());
        pages.set(1, 10);
        pages.set(2, 10);
        pages.set(3, 10);
        assert!(pages.all_copied());
        assert_eq!(pages.copied_count(), 4);
        assert_eq!(pages.get(2), Some(10));
        assert_eq!(pages.get(4), None);
        pages.set(1, 0);
        assert!(!pages.all_copied());
        assert_eq!(pages.copied_count(), 3);
    }

    #[test]
    fn card_admission_classifies_all_rejection_reasons() {
        use super::{CARD_REJECTS, CardReject, check_card_admission, note_card_reject};
        let least = 256 << 10;
        let most = 1024 << 20;
        let vram_pct = 92;
        let vm = Some([7000 << 20, 6000 << 20, 8000 << 20, 100 << 20]);

        assert_eq!(
            check_card_admission(64 << 10, 64 << 10, least, most, vram_pct, vm, 0),
            Err(CardReject::Threshold)
        );
        assert_eq!(
            check_card_admission(1 << 30, 1 << 30, least, most, vram_pct, vm, 0),
            Err(CardReject::Threshold)
        );
        assert_eq!(
            check_card_admission(500 << 20, 500 << 20, least, most, vram_pct, vm, 0),
            Err(CardReject::DxgiBudget)
        );
        assert_eq!(
            check_card_admission(200 << 20, 200 << 20, least, most, vram_pct, vm, 900 << 20),
            Err(CardReject::CumulativeCap)
        );
        assert_eq!(
            check_card_admission(200 << 20, 200 << 20, least, most, vram_pct, vm, 500 << 20),
            Ok(())
        );

        let before = CARD_REJECTS[2].load(std::sync::atomic::Ordering::Relaxed);
        let before_b = CARD_REJECTS[3].load(std::sync::atomic::Ordering::Relaxed);
        note_card_reject(CardReject::DxgiBudget, 4096);
        assert_eq!(
            CARD_REJECTS[2].load(std::sync::atomic::Ordering::Relaxed),
            before + 1
        );
        assert_eq!(
            CARD_REJECTS[3].load(std::sync::atomic::Ordering::Relaxed),
            before_b + 4096
        );
    }

    #[test]
    fn live_class_accounting_matches_shadow_lifetime() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let Some((buffer, data)) = host.shared_buffer(64 << 10) else {
            return;
        };
        let base = super::LIVE_CLASS[2].load(std::sync::atomic::Ordering::Relaxed);
        {
            let mut s = super::Shadow::new(buffer, data, 0, 0x1000, 12_345);
            s.set_live_class(2);
            assert_eq!(
                super::LIVE_CLASS[2].load(std::sync::atomic::Ordering::Relaxed),
                base + 12_345
            );
        }
        assert_eq!(
            super::LIVE_CLASS[2].load(std::sync::atomic::Ordering::Relaxed),
            base
        );
    }

    #[test]
    fn tls_lookup_cache_hits_and_invalidates_on_epoch() {
        const CHILD_ENV: &str = "VICTORY_REA_CHILD_TLS_SHADOW_TEST";
        if std::env::var_os(CHILD_ENV).is_some() {
            assert!(crate::descriptor::host::gpu_fastpath_enabled());
            if crate::host::host().is_none() {
                return;
            }
            let base_addr = 0x7000_0000_0000u64;
            super::register(101, base_addr, 4096);
            // Seed entry.shadow directly to avoid touching unmapped guest VA.
            {
                let host = crate::host::host().unwrap();
                let (buffer, data) = host.shared_buffer(4096).unwrap();
                let gpu = unsafe { buffer.GetGPUVirtualAddress() };
                let s = std::sync::Arc::new(super::Shadow::new(buffer, data, gpu, base_addr, 4096));
                let mut map = super::buffers_write();
                map.get_mut(&(base_addr, 101)).unwrap().shadow = Some(s);
                super::bump_shadow_epoch();
            }

            let (h0, m0) = super::tls_cache_stats();
            let a = super::lookup(base_addr, 64, false).unwrap();
            let b = super::lookup(base_addr + 256, 64, false).unwrap();
            assert!(std::sync::Arc::ptr_eq(&a, &b));
            let (h1, m1) = super::tls_cache_stats();
            assert!(m1 > m0, "first lookup must miss TLS");
            assert!(h1 > h0, "subrange offset lookup must hit TLS");

            // Epoch bump on register/unregister invalidates TLS cache.
            super::register(102, base_addr + 0x10_000, 4096);
            let _ = super::lookup(base_addr + 128, 64, false).unwrap();
            let (_h2, m2) = super::tls_cache_stats();
            assert!(m2 > m1, "epoch change must invalidate TLS cache");
            super::unregister(102, base_addr + 0x10_000);
            super::unregister(101, base_addr);
            return;
        }
        let exe = std::env::current_exe().unwrap();
        let out = std::process::Command::new(exe)
            .arg("--exact")
            .arg("shadow::tests::tls_lookup_cache_hits_and_invalidates_on_epoch")
            .env(CHILD_ENV, "1")
            .env("VICTORY_REA_GPU_FASTPATH", "1")
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "child test failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    struct WriteWatchAlloc {
        ptr: *mut u8,
        _len: usize,
    }

    impl WriteWatchAlloc {
        fn new(len: usize) -> Option<Self> {
            use windows_sys::Win32::System::Memory::{
                MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
            };
            const MEM_WRITE_WATCH: u32 = 0x0020_0000;
            // SAFETY: allocates private RW write-watched pages owned by this struct.
            let ptr = unsafe {
                VirtualAlloc(
                    std::ptr::null(),
                    len,
                    MEM_COMMIT | MEM_RESERVE | MEM_WRITE_WATCH,
                    PAGE_READWRITE,
                )
            }
            .cast::<u8>();
            (!ptr.is_null()).then_some(Self { ptr, _len: len })
        }

        fn addr(&self) -> usize {
            self.ptr as usize
        }
    }

    impl Drop for WriteWatchAlloc {
        fn drop(&mut self) {
            use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
            // SAFETY: matches VirtualAlloc above.
            unsafe {
                let _ = VirtualFree(self.ptr.cast(), 0, MEM_RELEASE);
            }
        }
    }

    #[test]
    fn adjacent_4k_cpu_write_preserves_gpu_owned_subpage() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let Some(mem) = WriteWatchAlloc::new(64 << 10) else {
            return;
        };
        let Some((buffer, data)) = host.shared_buffer(64 << 10) else {
            return;
        };
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let base = mem.addr();
        // Seed subpage 0 (CPU side) and subpage 1 (4 KiB adjacent).
        unsafe {
            std::ptr::write_bytes(mem.ptr, 0x11, 4096);
            std::ptr::write_bytes(mem.ptr.add(4096), 0x22, 4096);
        }
        let s = std::sync::Arc::new(super::Shadow::new(buffer, data, gpu, base as u64, 64 << 10));
        assert_eq!(s.update(base, 64 << 10), 64 << 10);

        // GPU writes 4 KiB at subpage 1 (+0x1000..+0x2000) directly into shadow without CPU copy-back.
        unsafe {
            std::ptr::write_bytes(data.add(4096), 0xAB, 4096);
        }
        s.note_gpu_write(base + 4096, 4096);

        // CPU now writes adjacent 4 KiB subpage 0 (+0x0000..+0x1000) in the same 64 KiB block.
        unsafe {
            std::ptr::write_bytes(mem.ptr, 0xCD, 4096);
        }
        let copied = s.update(base, 64 << 10);
        assert!(copied > 0, "adjacent 4K CPU write must propagate");

        // Subpage 0 must reflect new CPU bytes 0xCD; subpage 1 MUST still hold GPU bytes 0xAB!
        unsafe {
            assert_eq!(*data, 0xCD);
            assert_eq!(*data.add(4095), 0xCD);
            assert_eq!(
                *data.add(4096),
                0xAB,
                "GPU-owned adjacent 4K subpage was clobbered!"
            );
            assert_eq!(*data.add(8191), 0xAB);
        }
    }

    #[test]
    fn partial_subpage_gpu_write_preserves_bytes_on_refresh() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let Some(mem) = WriteWatchAlloc::new(64 << 10) else {
            return;
        };
        let Some((buffer, data)) = host.shared_buffer(64 << 10) else {
            return;
        };
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let base = mem.addr();
        unsafe {
            std::ptr::write_bytes(mem.ptr, 0x33, 64 << 10);
        }
        let s = std::sync::Arc::new(super::Shadow::new(buffer, data, gpu, base as u64, 64 << 10));
        let _ = s.update(base, 64 << 10);

        // GPU writes partial 512 bytes at +0x1000..+0x1200.
        unsafe {
            std::ptr::write_bytes(data.add(0x1000), 0xEF, 512);
        }
        s.note_gpu_write(base + 0x1000, 512);

        // CPU writes unrelated adjacent 4 KiB subpage (+0x0000..+0x1000) triggering 64K watch hit.
        unsafe {
            std::ptr::write_bytes(mem.ptr, 0x44, 4096);
        }
        let _ = s.update(base, 64 << 10);

        unsafe {
            assert_eq!(*data, 0x44);
            assert_eq!(
                *data.add(0x1000),
                0xEF,
                "partial subpage GPU slice clobbered"
            );
            assert_eq!(*data.add(0x11FF), 0xEF);
            assert_eq!(*data.add(0x1200), 0x33);
        }
    }

    #[test]
    fn gpu_writes_preserve_authority_when_copy_back_skipped_and_materializes_on_cpu_read() {
        const CHILD: &str = "VICTORY_TEST_SHADOW_GPU_OWNED_CHILD";
        const OK: &str = "shadow gpu_writes + materialize assertions passed";
        if std::env::var_os(CHILD).is_none() {
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "shadow::window_tests::gpu_writes_preserve_authority_when_copy_back_skipped_and_materializes_on_cpu_read",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .env("VICTORY_COPY_BACK_ONLY_READBACK", "1")
                .env("VICTORY_REA_GPU_AUTHORITY", "1")
                .output()
                .unwrap();
            assert!(
                out.status.success() && String::from_utf8_lossy(&out.stdout).contains(OK),
                "stdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            return;
        }
        let Some(host) = crate::host::host() else {
            println!("{OK}");
            return;
        };
        let Some(mem) = WriteWatchAlloc::new(64 << 10) else {
            return;
        };
        let Some((buffer, data)) = host.shared_buffer(64 << 10) else {
            return;
        };
        let Some(list) = host.list_for_test() else {
            return;
        };
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let base = mem.addr();
        unsafe {
            std::ptr::write_bytes(mem.ptr, 0x55, 64 << 10);
        }
        let mut s = super::Shadow::new(buffer, data, gpu, base as u64, 64 << 10);
        s.set_live_class(0);
        let s = std::sync::Arc::new(s);
        let _ = s.update(base, 64 << 10);

        // Even with VICTORY_COPY_BACK_ONLY_READBACK=1 skipping copy-back on non-readback VA,
        // `gpu_writes` MUST still defer `Step::GpuWrite { copy_back: false }`.
        s.gpu_writes(&list, base as u64 + 0x2000, 1024);
        list.run_deferred_for_test();
        assert!(
            s.has_gpu_owned.load(std::sync::atomic::Ordering::Acquire),
            "gpu_writes must preserve GPU ownership even when copy_back is false"
        );

        // Simulate GPU output in shadow buffer, then legitimate CPU read materialization.
        unsafe {
            std::ptr::write_bytes(data.add(0x2000), 0x99, 1024);
        }
        let mat = s.materialize_gpu_owned(base + 0x2000, 1024);
        assert_eq!(mat, 1024);
        unsafe {
            assert_eq!(*mem.ptr.add(0x2000), 0x99);
            assert_eq!(*mem.ptr.add(0x23FF), 0x99);
        }
        assert!(!s.has_gpu_owned.load(std::sync::atomic::Ordering::Acquire));
        println!("{OK}");
    }

    #[test]
    fn indirect_argument_buffer_preserved_without_premature_copyback() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let Some(mem) = WriteWatchAlloc::new(64 << 10) else {
            return;
        };
        let Some((buffer, data)) = host.shared_buffer(64 << 10) else {
            return;
        };
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let base = mem.addr();
        let s = std::sync::Arc::new(super::Shadow::new(buffer, data, gpu, base as u64, 64 << 10));
        let _ = s.update(base, 64 << 10);

        // GPU compute culling writes DrawIndexedInstanced / Dispatch indirect args [32..48).
        let args = [256u32, 4u32, 0u32, 1u32];
        unsafe {
            std::ptr::copy_nonoverlapping(args.as_ptr().cast::<u8>(), data.add(0x4000), 16);
        }
        s.note_gpu_write(base + 0x4000, 16);

        // Adjacent CPU write at +0x0000 updates vertex/constant header, must NOT zero indirect args.
        unsafe {
            std::ptr::write_bytes(mem.ptr, 0x77, 4096);
        }
        let _ = s.update(base, 64 << 10);
        let mut readback_args = [0u32; 4];
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.add(0x4000).cast_const(),
                readback_args.as_mut_ptr().cast::<u8>(),
                16,
            );
        }
        assert_eq!(readback_args, args);
    }

    #[test]
    fn buffer_va_reuse_across_frames_resets_stale_gpu_ownership() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let Some(mem) = WriteWatchAlloc::new(64 << 10) else {
            return;
        };
        let base = mem.addr();
        let before_global = super::ACTIVE_GPU_OWNED.load(std::sync::atomic::Ordering::Relaxed);
        {
            let Some((b1, d1)) = host.shared_buffer(64 << 10) else {
                return;
            };
            let g1 = unsafe { b1.GetGPUVirtualAddress() };
            let mut s1_inner = super::Shadow::new(b1, d1, g1, base as u64, 64 << 10);
            s1_inner.set_live_class(0);
            let s1 = std::sync::Arc::new(s1_inner);
            s1.note_gpu_write(base, 4096);
            assert!(
                super::ACTIVE_GPU_OWNED.load(std::sync::atomic::Ordering::Relaxed) > before_global
            );
        }
        assert_eq!(
            super::ACTIVE_GPU_OWNED.load(std::sync::atomic::Ordering::Relaxed),
            before_global,
            "dropping frame N buffer must release ACTIVE_GPU_OWNED"
        );
        unsafe {
            std::ptr::write_bytes(mem.ptr, 0x2A, 4096);
        }
        let Some((b2, d2)) = host.shared_buffer(64 << 10) else {
            return;
        };
        let g2 = unsafe { b2.GetGPUVirtualAddress() };
        let s2 = super::Shadow::new(b2, d2, g2, base as u64, 64 << 10);
        let _ = s2.update(base, 64 << 10);
        unsafe {
            assert_eq!(*d2, 0x2A);
        }
    }

    #[test]
    fn concurrent_cpu_and_gpu_writes_on_independent_regions_remain_coherent() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let Some(mem) = WriteWatchAlloc::new(64 << 10) else {
            return;
        };
        let Some((buffer, data)) = host.shared_buffer(64 << 10) else {
            return;
        };
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let base = mem.addr();
        let s = std::sync::Arc::new(super::Shadow::new(buffer, data, gpu, base as u64, 64 << 10));
        let _ = s.update(base, 64 << 10);

        let s_gpu = s.clone();
        let data_addr = data as usize;
        let gpu_thread = std::thread::spawn(move || {
            for iter in 0..8u8 {
                unsafe {
                    std::ptr::write_bytes((data_addr + 0x2000) as *mut u8, 0x80 + iter, 4096);
                }
                s_gpu.note_gpu_write(base + 0x2000, 4096);
            }
        });
        for iter in 0..8u8 {
            unsafe {
                std::ptr::write_bytes(mem.ptr, 0x10 + iter, 4096);
            }
            let _ = s.update(base, 64 << 10);
        }
        gpu_thread.join().unwrap();
        let _ = s.update(base, 64 << 10);

        unsafe {
            assert_eq!(*data, 0x17);
            assert_eq!(*data.add(0x2000), 0x87);
        }
    }

    #[test]
    fn partial_split_coalesces_adjacent_subpages_and_preserves_gpu_bytes() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let Some(mem) = WriteWatchAlloc::new(64 << 10) else {
            return;
        };
        let Some((buffer, data)) = host.shared_buffer(64 << 10) else {
            return;
        };
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let base = mem.addr();
        unsafe {
            std::ptr::write_bytes(mem.ptr, 0x11, 64 << 10);
        }
        let s = std::sync::Arc::new(super::Shadow::new(buffer, data, gpu, base as u64, 64 << 10));
        assert_eq!(s.update(base, 64 << 10), 64 << 10);

        // Mark subpage 3 (+0x3000..+0x4000) GPU-owned.
        unsafe {
            std::ptr::write_bytes(data.add(0x3000), 0xEE, 4096);
        }
        s.note_gpu_write(base + 0x3000, 4096);

        // Dirty all other 15 subpages (0..3 and 4..16) in title RAM.
        unsafe {
            std::ptr::write_bytes(mem.ptr, 0x55, 3 * 4096);
            std::ptr::write_bytes(mem.ptr.add(0x4000), 0x55, 12 * 4096);
        }
        let calls_before = super::COPY_CALLS.load(std::sync::atomic::Ordering::Relaxed);
        let copied = s.update(base, 64 << 10);
        let calls_after = super::COPY_CALLS.load(std::sync::atomic::Ordering::Relaxed);

        assert_eq!(copied, 15 * 4096);
        let split_calls = calls_after - calls_before;
        assert!(
            split_calls <= 2,
            "adjacent CPU-dirty 4 KiB subpages must coalesce into <=2 copy calls, got {split_calls}"
        );
        unsafe {
            assert_eq!(*data, 0x55);
            assert_eq!(*data.add(0x3000), 0xEE);
            assert_eq!(*data.add(0x4000), 0x55);
        }
    }

    #[test]
    fn warm_note_gpu_write_merges_sorted_intervals_without_re_sorting() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let Some(mem) = WriteWatchAlloc::new(64 << 10) else {
            return;
        };
        let Some((buffer, data)) = host.shared_buffer(64 << 10) else {
            return;
        };
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let base = mem.addr();
        let s = std::sync::Arc::new(super::Shadow::new(buffer, data, gpu, base as u64, 64 << 10));
        // Warm shadow so watched != 0.
        let _ = s.update(base, 64 << 10);

        let started = std::time::Instant::now();
        for i in 0..10_000usize {
            let off = (i % 8) * 1024;
            s.note_gpu_write(base + off, 2048);
        }
        let elapsed = started.elapsed();
        let count = s.gpu_owned.lock().unwrap_or_else(|e| e.into_inner()).len();
        assert_eq!(count, 1, "overlapping intervals must merge into 1 span");
        eprintln!(
            "10,000 warm note_gpu_write calls completed in {:?}",
            elapsed
        );
    }

    #[test]
    fn materialize_and_invalidate_scale_o1_against_sparse_owned_index() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let Some((buffer, data)) = host.shared_buffer(64 << 10) else {
            return;
        };
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let s = std::sync::Arc::new(super::Shadow::new(
            buffer,
            data,
            gpu,
            0x7000_0000_0000,
            64 << 10,
        ));
        s.note_gpu_write(0x7000_0000_0000, 4096);

        // Probe 50,000 times against disjoint addresses outside GPU_OWNED_MIN..GPU_OWNED_MAX.
        let started = std::time::Instant::now();
        for i in 0..50_000usize {
            let probe = 0x1000_0000 + (i & 0xFF) * 4096;
            super::materialize_for_cpu_read(probe, 256);
            super::invalidate_for_cpu_write(probe, 256);
        }
        let elapsed = started.elapsed();
        eprintln!(
            "50,000 materialize+invalidate misses completed in {:?}",
            elapsed
        );
        assert!(
            elapsed < std::time::Duration::from_secs(1),
            "disjoint fast-path took unexpectedly long: {:?}",
            elapsed
        );
    }
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers_read()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

/// Returns cloned `Arc<Shadow>` instances whose `[start, start + len)` range
/// overlaps `[address, address + len)` and currently holds active `gpu_owned` bytes.
fn overlapping_gpu_owned_shadows(address: usize, len: usize) -> Vec<Arc<Shadow>> {
    if len == 0 || ACTIVE_GPU_OWNED.load(Ordering::Acquire) == 0 {
        return Vec::new();
    }
    let addr = address as u64;
    let end = addr.saturating_add(len as u64);
    if end <= GPU_OWNED_MIN.load(Ordering::Acquire) || addr >= GPU_OWNED_MAX.load(Ordering::Acquire)
    {
        return Vec::new();
    }
    let mut hits = Vec::new();
    let list = GPU_OWNED_SHADOWS.read().unwrap_or_else(|e| e.into_inner());
    for &(_, start, s_end, ref weak) in list.iter() {
        if start < end
            && addr < s_end
            && let Some(s) = weak.upgrade()
            && s.has_gpu_owned.load(Ordering::Acquire)
        {
            hits.push(s);
        }
    }
    hits
}

/// Ensures any GPU-owned bytes overlapping `[address, address + len)` are
/// materialized back into title RAM and demoted before a CPU read consumes them.
pub(crate) fn materialize_for_cpu_read(address: usize, len: usize) {
    if len == 0 {
        return;
    }
    crate::authority::observe_cpu_read_or_copy(address, len);
    for shadow in overlapping_gpu_owned_shadows(address, len) {
        let _ = shadow.materialize_gpu_owned(address, len);
    }
}

/// Invalidates GPU ownership on `[address, address + len)` when the CPU
/// explicitly writes that guest VA interval.
pub(crate) fn invalidate_for_cpu_write(address: usize, len: usize) {
    if len == 0 {
        return;
    }
    crate::authority::observe_cpu_write(address, len);
    for shadow in overlapping_gpu_owned_shadows(address, len) {
        shadow.on_cpu_overwrite(address, len);
    }
}
