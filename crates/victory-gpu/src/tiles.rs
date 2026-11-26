//! D3D12 Reserved Resources & Tiled Texture Pool (Front A: Sparse Residency).
//!
//! Eliminates physical VRAM commitment for unmapped / unstreamed mip levels
//! when `VICTORY_REA_SPARSE_TEXTURES=1` is active.
//!
//! Attribution: Informed by `desplmfao` / `Marvisak/VIctory` PR #2389 (`tiles.rs`,
//! GPL-3.0-or-later) and `fxceless/victory-fork` commit `9376fd6`, extended for
//! REA-002 with multi-extent sub-allocation across granular 16 MiB heaps,
//! fence-gated cold-mip reclamation (`VICTORY_REA_SPARSE_RECLAIM=1`), and
//! O(1) incremental accounting / authority tracking.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use crate::accounting;
use crate::authority::{self, AuthorityClass};

/// Standard D3D12 tile size is fixed at 64 KiB across all tiers.
pub const TILE_SIZE: u64 = 65_536;

/// Default minimum resource size before placing on sparse heaps (256 KiB).
const DEFAULT_SPARSE_MIN_BYTES: u64 = 256 << 10;

/// Default backing heap chunk granularity: 256 tiles = 16 MiB (friendly to 8 GB VRAM).
const DEFAULT_HEAP_TILES: u32 = 256;

/// Minimum idle submissions + wall milliseconds before reclaiming a fine standard mip.
const RECLAIM_IDLE_SUBMISSIONS: u64 = 180;
const RECLAIM_IDLE_MS: u64 = 2_000;

/// Maximum packed mips bitmask width supported lock-free.
const MAX_TRACKED_MIPS: usize = 16;

fn parse_bool_env(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| {
            let t = v.trim();
            t == "1" || t.eq_ignore_ascii_case("true") || t.eq_ignore_ascii_case("yes")
        })
        .unwrap_or(default)
}

/// Master switch (`VICTORY_REA_SPARSE_TEXTURES=1`, default OFF).
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = parse_bool_env("VICTORY_REA_SPARSE_TEXTURES", false);
        if on {
            tracing::info!(
                min_kb = min_bytes() >> 10,
                reclaim = reclaim_enabled(),
                "REA-002 sparse texture residency enabled"
            );
        }
        on
    })
}

/// Whether cold high-res standard mips may be unmapped back to the pool (`VICTORY_REA_SPARSE_RECLAIM`,
/// defaults ON whenever `VICTORY_REA_SPARSE_TEXTURES=1` unless explicitly set to `0`).
pub fn reclaim_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| enabled() && parse_bool_env("VICTORY_REA_SPARSE_RECLAIM", true))
}

/// Minimum virtual resource size to manage sparsely (`VICTORY_REA_SPARSE_MIN_KB`, default 256 KiB).
pub fn min_bytes() -> u64 {
    static BYTES: OnceLock<u64> = OnceLock::new();
    *BYTES.get_or_init(|| {
        std::env::var("VICTORY_REA_SPARSE_MIN_KB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|&kb| kb > 0)
            .map_or(DEFAULT_SPARSE_MIN_BYTES, |kb| kb << 10)
    })
}

fn max_pool_bytes() -> Option<u64> {
    static CAP: OnceLock<Option<u64>> = OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("VICTORY_REA_SPARSE_MAX_POOL_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|&mb| mb > 0)
            .map(|mb| mb << 20)
    })
}

/// Device-level Tiled Resources tier reported by `CheckFeatureSupport`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TiledSupport {
    pub tier: i32,
    pub supported: bool,
}

static SUPPORT: OnceLock<TiledSupport> = OnceLock::new();

/// Query and cache `D3D12_FEATURE_DATA_D3D12_OPTIONS::TiledResourcesTier`.
pub fn device_support(device: &ID3D12Device) -> TiledSupport {
    *SUPPORT.get_or_init(|| query_tiled_support(device))
}

/// Non-cached raw device capability query (used by GPU tests on independent devices).
pub fn query_tiled_support(device: &ID3D12Device) -> TiledSupport {
    let mut opts = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    // SAFETY: valid struct sized cleanly for D3D12_FEATURE_D3D12_OPTIONS.
    let ok = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS,
            (&raw mut opts).cast(),
            size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS>() as u32,
        )
    };
    if ok.is_err() {
        return TiledSupport::default();
    }
    let tier = opts.TiledResourcesTier.0;
    let supported = tier >= D3D12_TILED_RESOURCES_TIER_2.0;
    tracing::info!(tier, supported, "host D3D12 tiled resources capability");
    TiledSupport { tier, supported }
}

/// Whether `desc` is eligible for D3D12 Reserved Resource placement.
pub fn eligible(device: &ID3D12Device, desc: &D3D12_RESOURCE_DESC) -> bool {
    if !enabled() || !device_support(device).supported {
        return false;
    }
    if desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D {
        return false;
    }
    if desc.MipLevels == 1 || desc.SampleDesc.Count > 1 {
        return false;
    }
    let disallowed = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET
        | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL
        | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if (desc.Flags & disallowed) != D3D12_RESOURCE_FLAG_NONE {
        return false;
    }
    // Check exact virtual size using the reserved 64 KiB tile layout that
    // CreateReservedResource requires (caller `resource.rs` passes `Layout = UNKNOWN`).
    let mut reserved = *desc;
    reserved.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
    reserved.Alignment = 0;
    let alloc = unsafe { device.GetResourceAllocationInfo(0, std::slice::from_ref(&reserved)) };
    alloc.SizeInBytes >= min_bytes()
}

/// Extent `(heap_chunk_id, start_tile_in_chunk, tile_count)` returned by [`TilePool`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PoolExtent {
    chunk_id: u64,
    start_tile: u32,
    tile_count: u32,
}

#[derive(Clone, Debug)]
struct FreeSpan {
    start: u32,
    len: u32,
}

struct HeapChunk {
    id: u64,
    heap: ID3D12Heap,
    total_tiles: u32,
    free_tiles: u32,
    /// Disjoint sorted free spans `[start, start + len)`.
    free: Vec<FreeSpan>,
}

impl HeapChunk {
    fn new(id: u64, heap: ID3D12Heap, total_tiles: u32) -> Self {
        Self {
            id,
            heap,
            total_tiles,
            free_tiles: total_tiles,
            free: vec![FreeSpan {
                start: 0,
                len: total_tiles,
            }],
        }
    }

    fn take_best(&mut self, want: u32) -> Option<PoolExtent> {
        if self.free_tiles == 0 || self.free.is_empty() || want == 0 {
            return None;
        }
        // Prefer an exact/smallest span that holds `want` contiguously.
        if let Some((idx, _)) = self
            .free
            .iter()
            .enumerate()
            .filter(|(_, s)| s.len >= want)
            .min_by_key(|(_, s)| s.len)
        {
            let span = &mut self.free[idx];
            let start = span.start;
            if span.len == want {
                self.free.remove(idx);
            } else {
                span.start += want;
                span.len -= want;
            }
            self.free_tiles -= want;
            return Some(PoolExtent {
                chunk_id: self.id,
                start_tile: start,
                tile_count: want,
            });
        }
        // Fall back to taking the largest available contiguous run inside this chunk.
        let (idx, _) = self.free.iter().enumerate().max_by_key(|(_, s)| s.len)?;
        let span = self.free.remove(idx);
        self.free_tiles -= span.len;
        Some(PoolExtent {
            chunk_id: self.id,
            start_tile: span.start,
            tile_count: span.len,
        })
    }

    fn release(&mut self, start: u32, len: u32) {
        if len == 0 {
            return;
        }
        self.free_tiles += len;
        let pos = self.free.partition_point(|s| s.start < start);
        self.free.insert(pos, FreeSpan { start, len });
        // Coalesce right neighbor, then left neighbor.
        if pos + 1 < self.free.len()
            && self.free[pos].start + self.free[pos].len == self.free[pos + 1].start
        {
            self.free[pos].len += self.free[pos + 1].len;
            self.free.remove(pos + 1);
        }
        if pos > 0 && self.free[pos - 1].start + self.free[pos - 1].len == self.free[pos].start {
            self.free[pos - 1].len += self.free[pos].len;
            self.free.remove(pos);
        }
    }

    fn is_completely_free(&self) -> bool {
        self.free_tiles == self.total_tiles
    }
}

struct RetiredBatch {
    submission: u64,
    extents: Vec<PoolExtent>,
}

struct TilePoolState {
    next_id: u64,
    chunks: BTreeMap<u64, HeapChunk>,
    retired: Vec<RetiredBatch>,
    total_heap_bytes: u64,
    free_tiles: u64,
}

impl TilePoolState {
    const fn new() -> Self {
        Self {
            next_id: 1,
            chunks: BTreeMap::new(),
            retired: Vec::new(),
            total_heap_bytes: 0,
            free_tiles: 0,
        }
    }

    fn drain_retired(&mut self, finished_submission: u64) {
        if self.retired.is_empty() {
            return;
        }
        let mut remaining = Vec::new();
        for batch in self.retired.drain(..) {
            if batch.submission <= finished_submission {
                for ext in batch.extents {
                    if let Some(chunk) = self.chunks.get_mut(&ext.chunk_id) {
                        chunk.release(ext.start_tile, ext.tile_count);
                        self.free_tiles += u64::from(ext.tile_count);
                    }
                }
            } else {
                remaining.push(batch);
            }
        }
        self.retired = remaining;

        // Release empty heap chunks back to Windows/WDDM once free hysteresis exceeds 32 MiB.
        const HYSTERESIS_FREE_TILES: u64 = 512;
        if self.free_tiles > HYSTERESIS_FREE_TILES {
            let empty_ids: Vec<u64> = self
                .chunks
                .iter()
                .filter(|(_, c)| c.is_completely_free())
                .map(|(&id, _)| id)
                .collect();
            for id in empty_ids {
                if self.free_tiles <= HYSTERESIS_FREE_TILES {
                    break;
                }
                if let Some(chunk) = self.chunks.remove(&id) {
                    let chunk_bytes = u64::from(chunk.total_tiles) * TILE_SIZE;
                    self.total_heap_bytes = self.total_heap_bytes.saturating_sub(chunk_bytes);
                    self.free_tiles = self.free_tiles.saturating_sub(u64::from(chunk.total_tiles));
                    accounting::on_tile_pool_delta(-(chunk_bytes as i64));
                    // `chunk.heap` is already known unmapped since all tiles retired cleanly.
                    drop(chunk);
                }
            }
        }
    }

    fn grow(&mut self, device: &ID3D12Device, min_tiles_needed: u32) -> Option<u64> {
        let want_tiles = DEFAULT_HEAP_TILES.max(min_tiles_needed);
        let want_bytes = u64::from(want_tiles) * TILE_SIZE;
        if let Some(cap) = max_pool_bytes()
            && self.total_heap_bytes + want_bytes > cap
        {
            return None;
        }

        // Try default chunk size first, falling back to smaller 4 MiB chunk if tight on VRAM.
        let candidates = if min_tiles_needed <= 64 && want_tiles > 64 {
            [want_tiles, 64]
        } else {
            [want_tiles, want_tiles]
        };

        for tiles in candidates {
            let bytes = u64::from(tiles) * TILE_SIZE;
            if let Some(cap) = max_pool_bytes()
                && self.total_heap_bytes + bytes > cap
            {
                continue;
            }
            let desc = D3D12_HEAP_DESC {
                SizeInBytes: bytes,
                Properties: D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_DEFAULT,
                    ..Default::default()
                },
                Alignment: D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT as u64,
                Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
            };
            let mut heap: Option<ID3D12Heap> = None;
            // SAFETY: valid D3D12_HEAP_DESC and live device.
            let res = unsafe { device.CreateHeap(&desc, &mut heap) };
            if res.is_ok()
                && let Some(heap) = heap
            {
                let id = self.next_id;
                self.next_id += 1;
                self.chunks.insert(id, HeapChunk::new(id, heap, tiles));
                self.total_heap_bytes += bytes;
                self.free_tiles += u64::from(tiles);
                accounting::on_tile_pool_delta(bytes as i64);
                POOL_HEAPS_ALLOCATED.fetch_add(1, Ordering::Relaxed);
                return Some(id);
            }
        }
        None
    }

    /// Allocate `count` tiles (contiguously in one chunk whenever possible so
    /// `UpdateTileMappings` avoids multi-heap splits).
    fn alloc(
        &mut self,
        device: &ID3D12Device,
        count: u32,
        finished_submission: u64,
    ) -> Option<Vec<(ID3D12Heap, PoolExtent)>> {
        if count == 0 {
            return Some(Vec::new());
        }
        self.drain_retired(finished_submission);

        // Fast path: find one chunk holding all `count` tiles contiguously.
        let single_id = self
            .chunks
            .iter()
            .filter(|(_, c)| c.free.iter().any(|s| s.len >= count))
            .map(|(&id, c)| (id, c.free_tiles))
            .min_by_key(|&(_, f)| f)
            .map(|(id, _)| id);

        if let Some(id) = single_id
            && let Some(chunk) = self.chunks.get_mut(&id)
            && let Some(ext) = chunk.take_best(count)
        {
            self.free_tiles = self.free_tiles.saturating_sub(u64::from(ext.tile_count));
            return Some(vec![(chunk.heap.clone(), ext)]);
        }

        // Otherwise allocate a fresh heap chunk sized for at least `count` tiles.
        if let Some(new_id) = self.grow(device, count)
            && let Some(chunk) = self.chunks.get_mut(&new_id)
            && let Some(ext) = chunk.take_best(count)
        {
            self.free_tiles = self.free_tiles.saturating_sub(u64::from(ext.tile_count));
            return Some(vec![(chunk.heap.clone(), ext)]);
        }

        // Last resort (e.g., pool cap near saturation): piece together across existing chunks.
        if self.free_tiles < u64::from(count) {
            return None;
        }
        let mut remaining = count;
        let mut out = Vec::new();
        let ids: Vec<u64> = self.chunks.keys().copied().collect();
        for id in ids {
            if remaining == 0 {
                break;
            }
            if let Some(chunk) = self.chunks.get_mut(&id) {
                while remaining > 0 {
                    if let Some(ext) = chunk.take_best(remaining) {
                        remaining -= ext.tile_count;
                        self.free_tiles = self.free_tiles.saturating_sub(u64::from(ext.tile_count));
                        out.push((chunk.heap.clone(), ext));
                    } else {
                        break;
                    }
                }
            }
        }
        if remaining == 0 {
            Some(out)
        } else {
            for (_, ext) in out {
                if let Some(chunk) = self.chunks.get_mut(&ext.chunk_id) {
                    chunk.release(ext.start_tile, ext.tile_count);
                    self.free_tiles += u64::from(ext.tile_count);
                }
            }
            None
        }
    }

    fn retire(&mut self, submission: u64, extents: Vec<PoolExtent>) {
        if extents.is_empty() {
            return;
        }
        self.retired.push(RetiredBatch {
            submission,
            extents,
        });
    }
}

static POOL: Mutex<TilePoolState> = Mutex::new(TilePoolState::new());
static TILE_MAP_LOCK: Mutex<()> = Mutex::new(());

static SPARSE_TEXTURES_CREATED: AtomicU64 = AtomicU64::new(0);
static SPARSE_TEXTURES_ALIVE: AtomicU64 = AtomicU64::new(0);
static POOL_HEAPS_ALLOCATED: AtomicU64 = AtomicU64::new(0);
static TILE_MAP_OPS: AtomicU64 = AtomicU64::new(0);
static TILE_UNMAP_OPS: AtomicU64 = AtomicU64::new(0);
static TILES_RECLAIMED_TOTAL: AtomicU64 = AtomicU64::new(0);
static SPARSE_FALLBACKS: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Default)]
struct MipExtents {
    mapped: bool,
    extents: Vec<PoolExtent>,
    tile_count: u32,
}

struct BindingState {
    packed_mapped: bool,
    packed_extents: Vec<PoolExtent>,
    packed_last_submission: u64,
    packed_last_ms: u64,
    standard: Vec<MipExtents>,
}

/// Per-resource sparse mip residency tracker attached to [`crate::resource::HostTexture`].
pub(crate) struct SparseBinding {
    device: ID3D12Device,
    resource: ID3D12Resource,
    pub(crate) virtual_bytes: u64,
    authority_va: AtomicUsize,
    mip_levels: u32,
    array_slices: u32,
    num_standard_mips: u32,
    packed_info: D3D12_PACKED_MIP_INFO,
    subresource_tilings: Vec<D3D12_SUBRESOURCE_TILING>,
    mapped_mips_mask: AtomicU64,
    mapped_tiles_count: AtomicU32,
    mip_last_submission: [AtomicU64; MAX_TRACKED_MIPS],
    mip_last_ms: [AtomicU64; MAX_TRACKED_MIPS],
    dropped: AtomicBool,
    state: Mutex<BindingState>,
}

// SAFETY: D3D12 device/resource COM interfaces are free-threaded; mutations
// inside `SparseBinding` and queue `UpdateTileMappings` synchronize via locks.
unsafe impl Send for SparseBinding {}
unsafe impl Sync for SparseBinding {}

/// Summary of sparse texture residency metrics for telemetry & logs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SparseStats {
    pub enabled: bool,
    pub reclaim_enabled: bool,
    pub sparse_textures_created: u64,
    pub sparse_textures_alive: u64,
    pub virtual_reserved_mb: u64,
    pub physical_mapped_mb: u64,
    pub pool_capacity_mb: u64,
    pub pool_heaps_allocated: u64,
    pub tile_map_ops: u64,
    pub tile_unmap_ops: u64,
    pub tiles_reclaimed_total: u64,
    pub sparse_fallbacks: u64,
}

impl SparseBinding {
    /// Create a D3D12 Reserved Resource for `desc` and query its tiling metadata.
    /// Zero physical tiles are mapped initially.
    pub(crate) fn create(
        device: &ID3D12Device,
        desc: &D3D12_RESOURCE_DESC,
        initial_state: D3D12_RESOURCE_STATES,
        clear_value: Option<*const D3D12_CLEAR_VALUE>,
    ) -> Option<(ID3D12Resource, Arc<Self>)> {
        let mut reserved = *desc;
        reserved.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
        reserved.Alignment = 0;

        let mut resource: Option<ID3D12Resource> = None;
        // SAFETY: `reserved` uses `64KB_UNDEFINED_SWIZZLE` for a valid 2D texture.
        let res = unsafe {
            device.CreateReservedResource(&reserved, initial_state, clear_value, &mut resource)
        };
        if let Err(e) = res {
            tracing::warn!(error = %e, width = reserved.Width, height = reserved.Height, mips = reserved.MipLevels, "CreateReservedResource failed, falling back to committed");
            SPARSE_FALLBACKS.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let resource = resource?;
        // Resolve concrete mip/array counts (e.g. when title requested `MipLevels = 0`).
        // SAFETY: `resource` was just created cleanly.
        let actual = unsafe { resource.GetDesc() };
        let array_slices = u32::from(actual.DepthOrArraySize.max(1));
        let mip_levels = u32::from(actual.MipLevels.max(1));
        let total_subresources = mip_levels * array_slices;

        let mut total_tiles = 0u32;
        let mut packed_info = D3D12_PACKED_MIP_INFO::default();
        let mut tile_shape = D3D12_TILE_SHAPE::default();
        let mut num_tilings = total_subresources;
        let mut tilings = vec![D3D12_SUBRESOURCE_TILING::default(); total_subresources as usize];

        // SAFETY: `tilings` has capacity for `total_subresources`.
        unsafe {
            device.GetResourceTiling(
                &resource,
                Some(&mut total_tiles),
                Some(&mut packed_info),
                Some(&mut tile_shape),
                Some(&mut num_tilings),
                0,
                tilings.as_mut_ptr(),
            );
        }
        tilings.truncate(num_tilings as usize);

        if total_tiles == 0 {
            SPARSE_FALLBACKS.fetch_add(1, Ordering::Relaxed);
            return None;
        }

        let virtual_bytes = u64::from(total_tiles) * TILE_SIZE;
        let num_standard_mips = u32::from(packed_info.NumStandardMips).min(mip_levels);

        SPARSE_TEXTURES_CREATED.fetch_add(1, Ordering::Relaxed);
        SPARSE_TEXTURES_ALIVE.fetch_add(1, Ordering::Relaxed);

        let binding = Arc::new(Self {
            device: device.clone(),
            resource: resource.clone(),
            virtual_bytes,
            authority_va: AtomicUsize::new(0),
            mip_levels,
            array_slices,
            num_standard_mips,
            packed_info,
            subresource_tilings: tilings,
            mapped_mips_mask: AtomicU64::new(0),
            mapped_tiles_count: AtomicU32::new(0),
            mip_last_submission: [const { AtomicU64::new(0) }; MAX_TRACKED_MIPS],
            mip_last_ms: [const { AtomicU64::new(0) }; MAX_TRACKED_MIPS],
            dropped: AtomicBool::new(false),
            state: Mutex::new(BindingState {
                packed_mapped: false,
                packed_extents: Vec::new(),
                packed_last_submission: 0,
                packed_last_ms: 0,
                standard: vec![MipExtents::default(); num_standard_mips as usize],
            }),
        });

        Some((resource, binding))
    }

    pub(crate) fn set_authority_va(&self, title_va: usize) {
        if title_va != 0 {
            self.authority_va.store(title_va, Ordering::Relaxed);
            authority::register_range(
                title_va,
                self.virtual_bytes as usize,
                AuthorityClass::SparseTiled,
                true,
            );
        }
    }

    pub(crate) fn mapped_bytes(&self) -> u64 {
        u64::from(self.mapped_tiles_count.load(Ordering::Relaxed)) * TILE_SIZE
    }

    #[cfg(test)]
    pub(crate) fn mapped_mips_mask(&self) -> u64 {
        self.mapped_mips_mask.load(Ordering::Relaxed)
    }

    fn touch_mips(&self, first_mip: u32, now_submission: u64, now_ms: u64) {
        let clamped_first = first_mip.min(self.mip_levels.saturating_sub(1));
        for m in clamped_first..self.mip_levels {
            let idx = (m as usize).min(MAX_TRACKED_MIPS - 1);
            self.mip_last_submission[idx].store(now_submission, Ordering::Relaxed);
            self.mip_last_ms[idx].store(now_ms, Ordering::Relaxed);
        }
    }

    fn required_mask(&self, first_mip: u32) -> u64 {
        let first = first_mip.min(self.mip_levels.saturating_sub(1));
        let mut mask = 0u64;
        for m in first..self.mip_levels.min(64) {
            mask |= 1u64 << m;
        }
        // Packed mip tail is always required whenever any mip of the texture is active.
        if self.packed_info.NumPackedMips > 0 {
            for m in self.num_standard_mips..self.mip_levels.min(64) {
                mask |= 1u64 << m;
            }
        }
        mask
    }

    /// Ensure packed mip tail and every standard mip in `first_mip..num_standard_mips`
    /// have physical VRAM tiles mapped before command lists referencing them execute.
    pub(crate) fn ensure_resident(
        &self,
        queue: &ID3D12CommandQueue,
        first_mip: u32,
        now_submission: u64,
        finished_submission: u64,
        now_ms: u64,
    ) -> u64 {
        if self.dropped.load(Ordering::Relaxed) {
            return 0;
        }
        self.touch_mips(first_mip, now_submission, now_ms);
        let want_mask = self.required_mask(first_mip);
        if (self.mapped_mips_mask.load(Ordering::Acquire) & want_mask) == want_mask {
            return 0;
        }

        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let mut newly_mapped_tiles = 0u32;

        // 1. Ensure packed mip tail is mapped across every array slice.
        let need_packed = !st.packed_mapped && self.packed_info.NumTilesForPackedMips > 0;
        if need_packed {
            let per_slice_tiles = self.packed_info.NumTilesForPackedMips;
            let total_packed_tiles = per_slice_tiles * self.array_slices;
            let allocated = {
                let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
                pool.alloc(&self.device, total_packed_tiles, finished_submission)
            };
            if let Some(allocs) = allocated {
                if self.map_packed_slices(queue, &allocs, per_slice_tiles) {
                    st.packed_extents = allocs.into_iter().map(|(_, ext)| ext).collect();
                    st.packed_mapped = true;
                    st.packed_last_submission = now_submission;
                    st.packed_last_ms = now_ms;
                    newly_mapped_tiles += total_packed_tiles;
                    let mut mask = 0u64;
                    for m in self.num_standard_mips..self.mip_levels.min(64) {
                        mask |= 1u64 << m;
                    }
                    self.mapped_mips_mask.fetch_or(mask, Ordering::Release);
                } else {
                    let extents = allocs.into_iter().map(|(_, ext)| ext).collect();
                    POOL.lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .retire(now_submission, extents);
                }
            }
        }

        // 2. Ensure requested standard mips `[first_mip .. num_standard_mips)` are mapped.
        let start_mip = first_mip.min(self.num_standard_mips);
        for mip in start_mip..self.num_standard_mips {
            if st.standard[mip as usize].mapped {
                continue;
            }
            let per_slice_tiles = self.tiles_for_standard_mip(mip, 0);
            if per_slice_tiles == 0 {
                st.standard[mip as usize].mapped = true;
                if mip < 64 {
                    self.mapped_mips_mask
                        .fetch_or(1u64 << mip, Ordering::Release);
                }
                continue;
            }
            let total_mip_tiles = self.total_tiles_for_standard_mip(mip);
            let allocated = {
                let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
                pool.alloc(&self.device, total_mip_tiles, finished_submission)
            };
            if let Some(allocs) = allocated {
                if self.map_standard_mip_slices(queue, mip, &allocs) {
                    st.standard[mip as usize].extents =
                        allocs.into_iter().map(|(_, ext)| ext).collect();
                    st.standard[mip as usize].mapped = true;
                    st.standard[mip as usize].tile_count = total_mip_tiles;
                    newly_mapped_tiles += total_mip_tiles;
                    if mip < 64 {
                        self.mapped_mips_mask
                            .fetch_or(1u64 << mip, Ordering::Release);
                    }
                } else {
                    let extents = allocs.into_iter().map(|(_, ext)| ext).collect();
                    POOL.lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .retire(now_submission, extents);
                }
            }
        }

        if newly_mapped_tiles > 0 {
            self.mapped_tiles_count
                .fetch_add(newly_mapped_tiles, Ordering::Relaxed);
            let delta_bytes = u64::from(newly_mapped_tiles) * TILE_SIZE;
            accounting::on_sparse_tiles_delta(delta_bytes as i64);
            delta_bytes
        } else {
            0
        }
    }

    fn tiles_for_standard_mip(&self, mip: u32, slice: u32) -> u32 {
        let sub = (slice * self.mip_levels + mip) as usize;
        self.subresource_tilings.get(sub).map_or(0, |t| {
            t.WidthInTiles * u32::from(t.HeightInTiles) * u32::from(t.DepthInTiles.max(1))
        })
    }

    fn total_tiles_for_standard_mip(&self, mip: u32) -> u32 {
        (0..self.array_slices)
            .map(|s| self.tiles_for_standard_mip(mip, s))
            .sum()
    }

    fn map_packed_slices(
        &self,
        queue: &ID3D12CommandQueue,
        allocs: &[(ID3D12Heap, PoolExtent)],
        per_slice_tiles: u32,
    ) -> bool {
        if self.num_standard_mips >= self.mip_levels || per_slice_tiles == 0 {
            return true;
        }
        // Most common path: single contiguous allocation covering all array slices' packed tails.
        let mut slice_idx = 0u32;
        let mut tile_in_slice = 0u32;

        let _lock = TILE_MAP_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        for (heap, ext) in allocs {
            let mut rem = ext.tile_count;
            let mut heap_cursor = ext.start_tile;
            let mut coords = Vec::new();
            let mut regions = Vec::new();
            let mut flags = Vec::new();
            let mut heap_offsets = Vec::new();
            let mut tile_counts = Vec::new();

            while rem > 0 && slice_idx < self.array_slices {
                let take = rem.min(per_slice_tiles - tile_in_slice);
                let subresource = slice_idx * self.mip_levels + self.num_standard_mips;
                coords.push(D3D12_TILED_RESOURCE_COORDINATE {
                    X: tile_in_slice,
                    Y: 0,
                    Z: 0,
                    Subresource: subresource,
                });
                regions.push(D3D12_TILE_REGION_SIZE {
                    NumTiles: take,
                    UseBox: false.into(),
                    Width: 0,
                    Height: 0,
                    Depth: 0,
                });
                flags.push(D3D12_TILE_RANGE_FLAG_NONE);
                heap_offsets.push(heap_cursor);
                tile_counts.push(take);

                heap_cursor += take;
                rem -= take;
                tile_in_slice += take;
                if tile_in_slice >= per_slice_tiles {
                    tile_in_slice = 0;
                    slice_idx += 1;
                }
            }

            if !coords.is_empty() {
                // SAFETY: resource, heap, and tile coordinate slices are valid and synchronized.
                unsafe {
                    queue.UpdateTileMappings(
                        &self.resource,
                        coords.len() as u32,
                        Some(coords.as_ptr()),
                        Some(regions.as_ptr()),
                        Some(heap),
                        flags.len() as u32,
                        Some(flags.as_ptr()),
                        Some(heap_offsets.as_ptr()),
                        Some(tile_counts.as_ptr()),
                        D3D12_TILE_MAPPING_FLAG_NONE,
                    );
                }
                TILE_MAP_OPS.fetch_add(1, Ordering::Relaxed);
            }
        }
        true
    }

    fn map_standard_mip_slices(
        &self,
        queue: &ID3D12CommandQueue,
        mip: u32,
        allocs: &[(ID3D12Heap, PoolExtent)],
    ) -> bool {
        let mut slice_idx = 0u32;
        let mut tile_in_slice = 0u32;
        let _lock = TILE_MAP_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        for (heap, ext) in allocs {
            let mut rem = ext.tile_count;
            let mut heap_cursor = ext.start_tile;
            let mut coords = Vec::new();
            let mut regions = Vec::new();
            let mut flags = Vec::new();
            let mut heap_offsets = Vec::new();
            let mut tile_counts = Vec::new();

            while rem > 0 && slice_idx < self.array_slices {
                let sub = slice_idx * self.mip_levels + mip;
                let sub_tiling = self
                    .subresource_tilings
                    .get(sub as usize)
                    .copied()
                    .unwrap_or_default();
                let w = sub_tiling.WidthInTiles.max(1);
                let h = u32::from(sub_tiling.HeightInTiles.max(1));
                let sub_total = sub_tiling.WidthInTiles
                    * u32::from(sub_tiling.HeightInTiles)
                    * u32::from(sub_tiling.DepthInTiles.max(1));
                if sub_total == 0 {
                    slice_idx += 1;
                    tile_in_slice = 0;
                    continue;
                }
                let take = rem.min(sub_total - tile_in_slice);
                let x = tile_in_slice % w;
                let y = (tile_in_slice / w) % h;
                let z = tile_in_slice / (w * h);

                coords.push(D3D12_TILED_RESOURCE_COORDINATE {
                    X: x,
                    Y: y,
                    Z: z,
                    Subresource: sub,
                });
                regions.push(D3D12_TILE_REGION_SIZE {
                    NumTiles: take,
                    UseBox: false.into(),
                    Width: 0,
                    Height: 0,
                    Depth: 0,
                });
                flags.push(D3D12_TILE_RANGE_FLAG_NONE);
                heap_offsets.push(heap_cursor);
                tile_counts.push(take);

                heap_cursor += take;
                rem -= take;
                tile_in_slice += take;
                if tile_in_slice >= sub_total {
                    tile_in_slice = 0;
                    slice_idx += 1;
                }
            }

            if !coords.is_empty() {
                // SAFETY: valid D3D12 resource, heap, and coordinate spans.
                unsafe {
                    queue.UpdateTileMappings(
                        &self.resource,
                        coords.len() as u32,
                        Some(coords.as_ptr()),
                        Some(regions.as_ptr()),
                        Some(heap),
                        flags.len() as u32,
                        Some(flags.as_ptr()),
                        Some(heap_offsets.as_ptr()),
                        Some(tile_counts.as_ptr()),
                        D3D12_TILE_MAPPING_FLAG_NONE,
                    );
                }
                TILE_MAP_OPS.fetch_add(1, Ordering::Relaxed);
            }
        }
        true
    }

    fn unmap_standard_mip(&self, queue: &ID3D12CommandQueue, mip: u32) {
        let mut coords = Vec::with_capacity(self.array_slices as usize);
        let mut regions = Vec::with_capacity(self.array_slices as usize);
        let mut flags = Vec::with_capacity(self.array_slices as usize);
        let mut counts = Vec::with_capacity(self.array_slices as usize);

        for s in 0..self.array_slices {
            let tile_count = self.tiles_for_standard_mip(mip, s);
            if tile_count == 0 {
                continue;
            }
            coords.push(D3D12_TILED_RESOURCE_COORDINATE {
                X: 0,
                Y: 0,
                Z: 0,
                Subresource: s * self.mip_levels + mip,
            });
            regions.push(D3D12_TILE_REGION_SIZE {
                NumTiles: tile_count,
                UseBox: false.into(),
                Width: 0,
                Height: 0,
                Depth: 0,
            });
            flags.push(D3D12_TILE_RANGE_FLAG_NULL);
            counts.push(tile_count);
        }

        if coords.is_empty() {
            return;
        }
        let _lock = TILE_MAP_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: unmapping ranges to NULL with `pHeap = None` releases physical tile
        // references cleanly on Tier 2/3 resources without device removal.
        unsafe {
            queue.UpdateTileMappings(
                &self.resource,
                coords.len() as u32,
                Some(coords.as_ptr()),
                Some(regions.as_ptr()),
                None,
                flags.len() as u32,
                Some(flags.as_ptr()),
                None,
                Some(counts.as_ptr()),
                D3D12_TILE_MAPPING_FLAG_NONE,
            );
        }
        TILE_UNMAP_OPS.fetch_add(1, Ordering::Relaxed);
    }

    /// Reclaim higher-detail standard mips (`0..num_standard_mips`, never touching packed
    /// mip tail) that have been idle longer than `RECLAIM_IDLE_SUBMISSIONS` & `RECLAIM_IDLE_MS`.
    /// Returns reclaimed bytes.
    pub(crate) fn reclaim_cold_mips(
        &self,
        queue: &ID3D12CommandQueue,
        now_submission: u64,
        finished_submission: u64,
        now_ms: u64,
        max_reclaim_bytes: u64,
    ) -> u64 {
        if !reclaim_enabled() || self.dropped.load(Ordering::Relaxed) || max_reclaim_bytes == 0 {
            return 0;
        }

        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let mut reclaimed_bytes = 0u64;
        let mut to_retire = Vec::new();

        for mip in 0..self.num_standard_mips {
            if reclaimed_bytes >= max_reclaim_bytes {
                break;
            }
            if !st.standard[mip as usize].mapped {
                continue;
            }
            let idx = (mip as usize).min(MAX_TRACKED_MIPS - 1);
            let last_sub = self.mip_last_submission[idx].load(Ordering::Relaxed);
            let last_ms = self.mip_last_ms[idx].load(Ordering::Relaxed);
            let sub_idle = finished_submission.saturating_sub(last_sub) >= RECLAIM_IDLE_SUBMISSIONS;
            let time_idle = now_ms.saturating_sub(last_ms) >= RECLAIM_IDLE_MS;
            if !(sub_idle && time_idle) {
                continue;
            }

            let tile_count = st.standard[mip as usize].tile_count;
            self.unmap_standard_mip(queue, mip);
            let extents = std::mem::take(&mut st.standard[mip as usize].extents);
            st.standard[mip as usize].mapped = false;
            st.standard[mip as usize].tile_count = 0;
            if mip < 64 {
                self.mapped_mips_mask
                    .fetch_and(!(1u64 << mip), Ordering::Release);
            }
            if tile_count > 0 {
                self.mapped_tiles_count.fetch_sub(
                    tile_count.min(self.mapped_tiles_count.load(Ordering::Relaxed)),
                    Ordering::Relaxed,
                );
                let bytes = u64::from(tile_count) * TILE_SIZE;
                reclaimed_bytes += bytes;
                TILES_RECLAIMED_TOTAL.fetch_add(u64::from(tile_count), Ordering::Relaxed);
                accounting::on_sparse_tiles_delta(-(bytes as i64));
            }
            to_retire.extend(extents);
        }

        if !to_retire.is_empty() {
            let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
            pool.retire(now_submission, to_retire);
            pool.drain_retired(finished_submission);
        }
        reclaimed_bytes
    }

    pub(crate) fn retire_all(&self, retire_submission: u64) {
        if self.dropped.swap(true, Ordering::AcqRel) {
            return;
        }
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let mut all_extents = std::mem::take(&mut st.packed_extents);
        st.packed_mapped = false;
        for m in &mut st.standard {
            m.mapped = false;
            m.tile_count = 0;
            all_extents.extend(std::mem::take(&mut m.extents));
        }
        self.mapped_mips_mask.store(0, Ordering::Release);
        let mapped_tiles = self.mapped_tiles_count.swap(0, Ordering::AcqRel);
        if mapped_tiles > 0 {
            let delta = u64::from(mapped_tiles) * TILE_SIZE;
            accounting::on_sparse_tiles_delta(-(delta as i64));
        }
        if !all_extents.is_empty() {
            let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
            pool.retire(retire_submission, all_extents);
        }
    }
}

impl Drop for SparseBinding {
    fn drop(&mut self) {
        let va = self.authority_va.swap(0, Ordering::Relaxed);
        if va != 0 {
            authority::unregister_range(va, self.virtual_bytes as usize);
        }
        let sub = crate::host::started_submission();
        self.retire_all(sub);
        SPARSE_TEXTURES_ALIVE.fetch_sub(1, Ordering::Relaxed);
        // Ensure raw COM release of reserved resource happens cleanly; any pool chunks stay
        // retired until `sub` completes on the GPU queue.
        let _ = self.resource.as_raw();
        let _ = &self.device;
    }
}

/// Drain any tile pool heaps whose unmap/drop submission fence has already passed.
pub(crate) fn maintenance_drain(finished_submission: u64) {
    if !enabled() {
        return;
    }
    if let Ok(mut pool) = POOL.try_lock() {
        pool.drain_retired(finished_submission);
    }
}

/// Snapshot sparse texture statistics for telemetry / logging.
pub fn stats() -> SparseStats {
    let snap = accounting::snapshot();
    let pool_bytes = POOL
        .try_lock()
        .map(|p| p.total_heap_bytes)
        .unwrap_or(snap.tile_pool_allocated_bytes);
    SparseStats {
        enabled: enabled(),
        reclaim_enabled: reclaim_enabled(),
        sparse_textures_created: SPARSE_TEXTURES_CREATED.load(Ordering::Relaxed),
        sparse_textures_alive: SPARSE_TEXTURES_ALIVE.load(Ordering::Relaxed),
        virtual_reserved_mb: snap.reserved_textures_bytes >> 20,
        physical_mapped_mb: snap.mapped_tile_bytes >> 20,
        pool_capacity_mb: pool_bytes >> 20,
        pool_heaps_allocated: POOL_HEAPS_ALLOCATED.load(Ordering::Relaxed),
        tile_map_ops: TILE_MAP_OPS.load(Ordering::Relaxed),
        tile_unmap_ops: TILE_UNMAP_OPS.load(Ordering::Relaxed),
        tiles_reclaimed_total: TILES_RECLAIMED_TOTAL.load(Ordering::Relaxed),
        sparse_fallbacks: SPARSE_FALLBACKS.load(Ordering::Relaxed),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};

    #[test]
    fn default_flag_is_off_for_baseline_parity() {
        // Under default test harness env (VICTORY_REA_SPARSE_TEXTURES unset), must be OFF.
        assert!(!enabled());
    }

    #[test]
    fn gpu_reserved_resource_map_copy_reclaim_cycle() {
        // Real D3D12 Tier 2/3 hardware verification independent of process-wide env flag.
        let mut device: Option<ID3D12Device> = None;
        let created = unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device) };
        if created.is_err() {
            // Skip gracefully on headless runners without DX12 hardware adapter.
            return;
        }
        let device = device.unwrap();
        let sup = query_tiled_support(&device);
        if !sup.supported {
            return;
        }

        let queue: ID3D12CommandQueue = unsafe {
            device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .expect("direct command queue")
        };

        // Pass UNKNOWN layout just as `resource.rs::host_texture` does (`layout = 0`).
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Alignment: 0,
            Width: 1024,
            Height: 1024,
            DepthOrArraySize: 1,
            MipLevels: 11,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_NONE,
        };

        let (res, binding) =
            SparseBinding::create(&device, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None)
                .expect("reserved resource + tiling info");

        assert!(binding.virtual_bytes >= 5 << 20);
        assert_eq!(binding.mapped_bytes(), 0);
        assert_eq!(binding.mapped_mips_mask(), 0);
        assert!(binding.num_standard_mips > 0);

        // Map only tail + mip 4.. (skipping mips 0..3 -> saves >90% VRAM).
        let first = 4.min(binding.mip_levels - 1);
        let mapped = binding.ensure_resident(&queue, first, 1, 0, 1000);
        assert!(mapped > 0);
        assert_ne!(binding.mapped_mips_mask(), 0);
        assert!(
            binding.mapped_bytes() < binding.virtual_bytes / 2,
            "partial mip mapping ({} B) should use far less physical VRAM than virtual ({} B)",
            binding.mapped_bytes(),
            binding.virtual_bytes
        );

        // Execute a real GPU fence wait on the queue after mapping.
        let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap() };
        unsafe { queue.Signal(&fence, 1).unwrap() };
        while unsafe { fence.GetCompletedValue() } < 1 {
            std::thread::yield_now();
        }

        // Unmap standard mip if any standard mip in >= first was mapped.
        if first < binding.num_standard_mips {
            binding.unmap_standard_mip(&queue, first);
            unsafe { queue.Signal(&fence, 2).unwrap() };
            while unsafe { fence.GetCompletedValue() } < 2 {
                std::thread::yield_now();
            }
        }

        binding.retire_all(2);
        POOL.lock().unwrap().drain_retired(2);
        drop(res);
    }

    #[test]
    #[ignore = "rea002 microbenchmark"]
    fn rea002_bench_sparse_residency_and_dxr() {
        let mut device: Option<ID3D12Device> = None;
        if unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device) }.is_err() {
            return;
        }
        let device = device.unwrap();
        let sup = query_tiled_support(&device);
        assert!(sup.supported);
        let queue: ID3D12CommandQueue = unsafe {
            device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .unwrap()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Alignment: 0,
            Width: 4096,
            Height: 4096,
            DepthOrArraySize: 1,
            MipLevels: 13,
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_BC7_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_NONE,
        };
        let t_create = std::time::Instant::now();
        let (res, binding) =
            SparseBinding::create(&device, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None)
                .expect("4k BC7 reserved resource");
        let create_us = t_create.elapsed().as_micros();

        let t_tail = std::time::Instant::now();
        let tail_first = binding.num_standard_mips.min(binding.mip_levels - 1);
        let _ = binding.ensure_resident(&queue, tail_first, 1, 0, 100);
        let tail_us = t_tail.elapsed().as_micros();
        let tail_mapped = binding.mapped_bytes();

        let t_mip3 = std::time::Instant::now();
        let _ = binding.ensure_resident(&queue, 3, 2, 0, 101);
        let mip3_us = t_mip3.elapsed().as_micros();
        let mip3_mapped = binding.mapped_bytes();

        let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap() };
        unsafe { queue.Signal(&fence, 1).unwrap() };
        while unsafe { fence.GetCompletedValue() } < 1 {
            std::thread::yield_now();
        }
        let virt = binding.virtual_bytes;
        let saving_tail_pct = 100.0 * (1.0 - tail_mapped as f64 / virt as f64);
        let saving_mip3_pct = 100.0 * (1.0 - mip3_mapped as f64 / virt as f64);

        let dxr = crate::dxr_proto::run_isolated_smoke(&device).ok();
        println!(
            "REA002_BENCH_SPARSE_4K_BC7 tier={:?} virt_bytes={} tail_mapped_bytes={} tail_saving_pct={:.2} \
             mip3_mapped_bytes={} mip3_saving_pct={:.2} create_us={create_us} map_tail_us={tail_us} map_mip3_us={mip3_us} \
             dxr_tier={} dxr_build_us={} dxr_blas_bytes={} dxr_tlas_bytes={}",
            sup.tier,
            virt,
            tail_mapped,
            saving_tail_pct,
            mip3_mapped,
            saving_mip3_pct,
            dxr.map(|d| d.tier).unwrap_or(0),
            dxr.map(|d| d.last_build_us).unwrap_or(0),
            dxr.map(|d| d.blas_bytes).unwrap_or(0),
            dxr.map(|d| d.tlas_bytes).unwrap_or(0),
        );

        binding.retire_all(1);
        POOL.lock().unwrap().drain_retired(1);
        drop(res);
    }
}
