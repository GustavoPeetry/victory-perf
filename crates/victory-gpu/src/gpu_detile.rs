//! Opt-in GPU compute texture detiling backend (`VICTORY_REA003_GPU_DETILE=1`, off by default).
//!
//! Uses table-driven HLSL `cs_5_1` kernels fed by [`victory_formats::texture::Layout`]
//! micro-tile lookup patterns (`pattern_xy` / `pattern_z`) to detile console swizzled
//! textures (2D, array, 3D, mip-tails) into 4-byte-aligned linear staging slices without
//! `RWByteAddressBuffer` write races.

use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use victory_formats::texture::{Dimension, Layout, Swizzle};
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::{
    D3D_ROOT_SIGNATURE_VERSION_1_0, D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_DESC,
    D3D12_COMMAND_QUEUE_FLAG_NONE, D3D12_COMPUTE_PIPELINE_STATE_DESC, D3D12_FENCE_FLAG_NONE,
    D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_TYPE_READBACK, D3D12_HEAP_TYPE_UPLOAD,
    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_NONE,
    D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE,
    D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_ROOT_CONSTANTS,
    D3D12_ROOT_DESCRIPTOR, D3D12_ROOT_PARAMETER, D3D12_ROOT_PARAMETER_0,
    D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS, D3D12_ROOT_PARAMETER_TYPE_SRV,
    D3D12_ROOT_PARAMETER_TYPE_UAV, D3D12_ROOT_SIGNATURE_DESC, D3D12_ROOT_SIGNATURE_FLAG_NONE,
    D3D12_SHADER_BYTECODE, D3D12_SHADER_VISIBILITY_ALL, D3D12SerializeRootSignature,
    ID3D12CommandAllocator, ID3D12CommandList, ID3D12CommandQueue, ID3D12Device, ID3D12Fence,
    ID3D12GraphicsCommandList, ID3D12PipelineState, ID3D12RootSignature,
};
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::{Interface, s};

use crate::dxr_backend::{align_up, readback_vec, transition_barrier, uav_barrier_all};
use crate::dxr_proto::{create_buffer, upload_slice};

/// Gate flag for GPU compute detiling (`VICTORY_REA003_GPU_DETILE=1`, off by default).
pub fn gpu_detile_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("VICTORY_REA003_GPU_DETILE")
            .ok()
            .is_some_and(|v| v.trim() == "1")
    })
}

/// Cumulative metrics from GPU detile operations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuDetileStats {
    pub dispatches: u64,
    pub bytes_detiled: u64,
    pub upload_ring_hits: u64,
    pub fallback_cpu: u64,
    pub last_dispatch_us: u64,
}

static DISPATCHES: AtomicU64 = AtomicU64::new(0);
static BYTES_DETILED: AtomicU64 = AtomicU64::new(0);
static RING_HITS: AtomicU64 = AtomicU64::new(0);
static FALLBACK_CPU: AtomicU64 = AtomicU64::new(0);
static LAST_US: AtomicU64 = AtomicU64::new(0);

pub fn stats() -> GpuDetileStats {
    GpuDetileStats {
        dispatches: DISPATCHES.load(Ordering::Relaxed),
        bytes_detiled: BYTES_DETILED.load(Ordering::Relaxed),
        upload_ring_hits: RING_HITS.load(Ordering::Relaxed),
        fallback_cpu: FALLBACK_CPU.load(Ordering::Relaxed),
        last_dispatch_us: LAST_US.load(Ordering::Relaxed),
    }
}

pub(crate) fn note_cpu_fallback() {
    FALLBACK_CPU.fetch_add(1, Ordering::Relaxed);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DetileConstants {
    width: u32,
    height: u32,
    depth: u32,
    elem_bytes: u32,
    row_pitch: u32,
    depth_pitch: u32,
    bw: u32,
    bh: u32,
    bz: u32,
    align_log2: u32,
    thick: u32,
    is_linear: u32,
    slice_size: u32,
    mip_offset: u32,
    mip_pitch: u32,
    first_slice: u32,
    tail_x: u32,
    tail_y: u32,
    gx_count: u32,
    pad: u32,
}

const DETILE_HLSL: &str = r#"
ByteAddressBuffer g_tiled : register(t0);
ByteAddressBuffer g_lut : register(t1);
RWByteAddressBuffer g_linear : register(u0);

cbuffer DetileCB : register(b0)
{
    uint g_width;
    uint g_height;
    uint g_depth;
    uint g_elem_bytes;
    uint g_row_pitch;
    uint g_depth_pitch;
    uint g_bw;
    uint g_bh;
    uint g_bz;
    uint g_align_log2;
    uint g_thick;
    uint g_is_linear;
    uint g_slice_size;
    uint g_mip_offset;
    uint g_mip_pitch;
    uint g_first_slice;
    uint g_tail_x;
    uint g_tail_y;
    uint g_gx_count;
    uint g_pad;
};

uint tiled_byte_offset(uint x, uint y, uint slice_idx, uint inner_z)
{
    uint base = slice_idx * g_slice_size + g_mip_offset;
    if (g_is_linear != 0u)
    {
        return base + (y * g_mip_pitch + x) * g_elem_bytes;
    }
    uint tx = x + g_tail_x;
    uint ty = y + g_tail_y;
    uint blocks_x = tx / g_bw;
    uint blocks_y = ty / g_bh;
    uint blocks_per_row = g_mip_pitch / g_bw;
    uint lx = tx % g_bw;
    uint ly = ty % g_bh;
    uint pat_x = g_lut.Load(lx * 4u);
    uint pat_y = g_lut.Load((g_bw + ly) * 4u);
    uint pat_z = g_lut.Load((g_bw + g_bh + inner_z) * 4u);
    uint row_off = ((blocks_y * blocks_per_row) << g_align_log2) + pat_y + pat_z;
    uint col_off = (blocks_x << g_align_log2) + pat_x;
    return base + row_off + col_off;
}

uint load_u8(uint byte_off)
{
    uint word = g_tiled.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

uint load_u16(uint byte_off)
{
    uint word = g_tiled.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFFFu;
}

[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= g_gx_count || dtid.y >= g_height || dtid.z >= g_depth)
    {
        return;
    }

    uint z_abs = g_first_slice + dtid.z;
    uint slice_idx = (g_thick != 0u) ? (z_abs / g_bz) : z_abs;
    uint inner_z = (g_thick != 0u) ? (z_abs % g_bz) : 0u;

    uint row_base = dtid.z * g_depth_pitch + dtid.y * g_row_pitch;

    if (g_elem_bytes == 1u)
    {
        uint x0 = dtid.x * 4u;
        uint out_off = row_base + dtid.x * 4u;
        uint packed = (x0 + 3u < g_width) ? 0u : g_linear.Load(out_off);
        [unroll]
        for (uint k = 0u; k < 4u; ++k)
        {
            uint x = x0 + k;
            if (x < g_width)
            {
                uint b = load_u8(tiled_byte_offset(x, dtid.y, slice_idx, inner_z));
                uint shift = k * 8u;
                packed = (packed & ~(0xFFu << shift)) | (b << shift);
            }
        }
        g_linear.Store(out_off, packed);
    }
    else if (g_elem_bytes == 2u)
    {
        uint x0 = dtid.x * 2u;
        uint out_off = row_base + dtid.x * 4u;
        uint packed = (x0 + 1u < g_width) ? 0u : g_linear.Load(out_off);
        uint h0 = load_u16(tiled_byte_offset(x0, dtid.y, slice_idx, inner_z));
        packed = (packed & ~0xFFFFu) | h0;
        if (x0 + 1u < g_width)
        {
            uint h1 = load_u16(tiled_byte_offset(x0 + 1u, dtid.y, slice_idx, inner_z));
            packed = (packed & 0xFFFFu) | (h1 << 16u);
        }
        g_linear.Store(out_off, packed);
    }
    else if (g_elem_bytes == 4u)
    {
        uint src = tiled_byte_offset(dtid.x, dtid.y, slice_idx, inner_z);
        uint out_off = row_base + dtid.x * 4u;
        g_linear.Store(out_off, g_tiled.Load(src));
    }
    else if (g_elem_bytes == 8u)
    {
        uint src = tiled_byte_offset(dtid.x, dtid.y, slice_idx, inner_z);
        uint out_off = row_base + dtid.x * 8u;
        g_linear.Store2(out_off, g_tiled.Load2(src));
    }
    else if (g_elem_bytes == 16u)
    {
        uint src = tiled_byte_offset(dtid.x, dtid.y, slice_idx, inner_z);
        uint out_off = row_base + dtid.x * 16u;
        g_linear.Store4(out_off, g_tiled.Load4(src));
    }
}
"#;

/// Subresource coordinates and byte strides passed to [`GpuDetiler::detile_to_slice`].
#[derive(Clone, Copy, Debug)]
pub struct DetileSubresource {
    pub mip: u32,
    pub slice: u32,
    pub row_pitch: usize,
    pub depth_pitch: usize,
}

/// Self-contained D3D12 compute detiler matching [`victory_formats::texture::Layout::detile`].
pub struct GpuDetiler {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    alloc: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    fence_value: u64,
    event: HANDLE,
    root_sig: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

impl GpuDetiler {
    pub fn new(device: &ID3D12Device) -> Result<Self, String> {
        let cs_bytes = compile_cs51(DETILE_HLSL)?;
        let params = [
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Constants: D3D12_ROOT_CONSTANTS {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                        Num32BitValues: 20,
                    },
                },
            },
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                    },
                },
            },
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR {
                        ShaderRegister: 1,
                        RegisterSpace: 0,
                    },
                },
            },
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                    },
                },
            },
        ];
        let rs_desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: params.len() as u32,
            pParameters: params.as_ptr(),
            NumStaticSamplers: 0,
            pStaticSamplers: std::ptr::null(),
            Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
        };
        let mut blob: Option<ID3DBlob> = None;
        let mut err: Option<ID3DBlob> = None;
        // SAFETY: `params` stays live across `D3D12SerializeRootSignature`.
        unsafe {
            D3D12SerializeRootSignature(
                &rs_desc,
                D3D_ROOT_SIGNATURE_VERSION_1_0,
                &mut blob,
                Some(&mut err),
            )
            .map_err(|e| format!("D3D12SerializeRootSignature(gpu_detile): {e}"))?;
        }
        let blob = blob.ok_or_else(|| "missing gpu_detile root sig blob".to_owned())?;
        // SAFETY: valid device and serialized root signature blob.
        let root_sig: ID3D12RootSignature = unsafe {
            let slice = std::slice::from_raw_parts(
                blob.GetBufferPointer().cast::<u8>(),
                blob.GetBufferSize(),
            );
            device
                .CreateRootSignature(0, slice)
                .map_err(|e| format!("CreateRootSignature(gpu_detile): {e}"))?
        };
        let pso_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::ManuallyDrop::new(Some(root_sig.clone())),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: cs_bytes.as_ptr().cast::<c_void>(),
                BytecodeLength: cs_bytes.len(),
            },
            ..Default::default()
        };
        // SAFETY: `cs_bytes` and `root_sig` live across PSO creation.
        let (queue, alloc, list, fence, event, pso) = unsafe {
            let pso: ID3D12PipelineState = device
                .CreateComputePipelineState(&pso_desc)
                .map_err(|e| format!("CreateComputePipelineState(gpu_detile): {e}"))?;
            let queue: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
                    ..Default::default()
                })
                .map_err(|e| format!("CreateCommandQueue(gpu_detile): {e}"))?;
            let alloc: ID3D12CommandAllocator = device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .map_err(|e| format!("CreateCommandAllocator(gpu_detile): {e}"))?;
            let list: ID3D12GraphicsCommandList = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
                .map_err(|e| format!("CreateCommandList(gpu_detile): {e}"))?;
            list.Close().map_err(|e| format!("initial Close: {e}"))?;
            let fence: ID3D12Fence = device
                .CreateFence(0, D3D12_FENCE_FLAG_NONE)
                .map_err(|e| format!("CreateFence(gpu_detile): {e}"))?;
            let event = CreateEventW(None, false, false, None)
                .map_err(|e| format!("CreateEventW(gpu_detile): {e}"))?;
            (queue, alloc, list, fence, event, pso)
        };
        Ok(Self {
            device: device.clone(),
            queue,
            alloc,
            list,
            fence,
            fence_value: 0,
            event,
            root_sig,
            pso,
        })
    }

    /// Detiles `tiled` subresource into `linear`, matching [`Layout::detile`].
    pub fn detile_to_slice(
        &mut self,
        layout: &Layout,
        tiled: &[u8],
        sub: DetileSubresource,
        linear: &mut [u8],
    ) -> Result<u64, String> {
        let DetileSubresource {
            mip,
            slice,
            row_pitch,
            depth_pitch,
        } = sub;
        let m = layout
            .mips
            .get(mip as usize)
            .ok_or_else(|| "invalid mip".to_owned())?;
        let elem_bytes = layout.desc.element_bytes;
        if !matches!(elem_bytes, 1 | 2 | 4 | 8 | 16) {
            return Err("unsupported element_bytes for GPU detile".to_owned());
        }
        let is_linear = layout.desc.swizzle == Swizzle::LINEAR;
        if !is_linear && !supported_swizzle(layout.desc.swizzle) {
            return Err("unsupported swizzle mode for GPU detile".to_owned());
        }
        if !row_pitch.is_multiple_of(4) || !depth_pitch.is_multiple_of(4) {
            return Err("GPU detile requires 4-byte aligned row_pitch and depth_pitch".to_owned());
        }
        let first_slice = if layout.desc.dimension == Dimension::Tex3d {
            if slice != 0 {
                return Err("3D slice must be 0".to_owned());
            }
            0
        } else {
            slice
        };
        if first_slice >= layout.slices {
            return Err("slice out of range".to_owned());
        }
        let need_linear = (m.depth as usize - 1)
            .checked_mul(depth_pitch)
            .and_then(|d| {
                (m.height as usize - 1)
                    .checked_mul(row_pitch)
                    .and_then(|r| d.checked_add(r))
            })
            .and_then(|o| o.checked_add(m.width as usize * elem_bytes as usize))
            .ok_or_else(|| "linear buffer overflow".to_owned())?;
        if tiled.len() < layout.bytes_from(mip) as usize
            || linear.len() < need_linear
            || row_pitch < m.width as usize * elem_bytes as usize
        {
            return Err("buffer too small for detile".to_owned());
        }
        if tiled.len() > u32::MAX as usize || linear.len() > u32::MAX as usize {
            return Err("buffer exceeds 32-bit byte address range".to_owned());
        }

        let (pat_x, pat_y) = layout.pattern_xy();
        let pat_z = layout.pattern_z();
        let mut lut: Vec<u32> = Vec::with_capacity(
            pat_x.len().max(layout.block[0] as usize)
                + pat_y.len().max(layout.block[1] as usize)
                + pat_z.len().max(layout.block[2] as usize),
        );
        if is_linear {
            lut.push(0);
        } else {
            lut.extend_from_slice(pat_x);
            lut.resize(layout.block[0] as usize, 0);
            lut.extend_from_slice(pat_y);
            lut.resize((layout.block[0] + layout.block[1]) as usize, 0);
            lut.extend_from_slice(pat_z);
            lut.resize(
                (layout.block[0] + layout.block[1] + layout.block[2]) as usize,
                0,
            );
        }

        // Prepare tiled buffer via upload_ring when active, else transient upload resource.
        let tiled_bytes = align_up(tiled.len() as u64, 256);
        let pooled_tiled = crate::upload_ring::checkout_upload(
            &self.device,
            tiled_bytes,
            crate::host::finished_submission(),
        );
        let tiled_res = if let Some((res, ptr, _)) = &pooled_tiled {
            RING_HITS.fetch_add(1, Ordering::Relaxed);
            // SAFETY: pooled upload resource mapped for at least `tiled.len()` bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(tiled.as_ptr(), *ptr, tiled.len());
            }
            res.clone()
        } else {
            let res = create_buffer(
                &self.device,
                tiled_bytes,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            upload_slice(&res, tiled)?;
            res
        };

        let lut_bytes = align_up((lut.len() * 4) as u64, 256);
        let lut_res = create_buffer(
            &self.device,
            lut_bytes,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        upload_slice(&lut_res, &lut)?;

        let out_bytes = align_up(linear.len() as u64, 256);
        // Preserve caller padding bytes outside written pixels by seeding UAV via upload copy if non-zero.
        let out_seed = create_buffer(
            &self.device,
            out_bytes,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        upload_slice(&out_seed, linear)?;

        let out_uav = create_buffer(
            &self.device,
            out_bytes,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;
        let out_rb = create_buffer(
            &self.device,
            out_bytes,
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;

        let gx_count = match elem_bytes {
            1 => m.width.div_ceil(4),
            2 => m.width.div_ceil(2),
            _ => m.width,
        };
        let (tail_x, tail_y) = m.tail.map_or((0, 0), |t| (t.x, t.y));
        let consts = DetileConstants {
            width: m.width,
            height: m.height,
            depth: m.depth,
            elem_bytes,
            row_pitch: row_pitch as u32,
            depth_pitch: depth_pitch as u32,
            bw: layout.block[0],
            bh: layout.block[1],
            bz: layout.block[2],
            align_log2: layout.alignment.trailing_zeros(),
            thick: u32::from(layout.thick),
            is_linear: u32::from(is_linear),
            slice_size: layout.slice_size as u32,
            mip_offset: m.offset as u32,
            mip_pitch: m.pitch,
            first_slice,
            tail_x,
            tail_y,
            gx_count,
            pad: 0,
        };
        let gx = gx_count.div_ceil(8);
        let gy = m.height.div_ceil(8);
        let gz = m.depth;
        let started = Instant::now();
        let next_fence = self.fence_value + 1;

        // SAFETY: all upload, UAV, and readback resources remain alive until fence wait completes.
        unsafe {
            let tiled_va = tiled_res.GetGPUVirtualAddress();
            let lut_va = lut_res.GetGPUVirtualAddress();
            let out_va = out_uav.GetGPUVirtualAddress();

            self.alloc
                .Reset()
                .map_err(|e| format!("Reset alloc: {e}"))?;
            self.list
                .Reset(&self.alloc, None)
                .map_err(|e| format!("Reset list: {e}"))?;
            self.list.CopyResource(&out_uav, &out_seed);
            self.list.ResourceBarrier(&[transition_barrier(
                &out_uav,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
            self.list.SetComputeRootSignature(&self.root_sig);
            self.list.SetPipelineState(&self.pso);
            self.list
                .SetComputeRoot32BitConstants(0, 20, (&raw const consts).cast(), 0);
            self.list.SetComputeRootShaderResourceView(1, tiled_va);
            self.list.SetComputeRootShaderResourceView(2, lut_va);
            self.list.SetComputeRootUnorderedAccessView(3, out_va);
            self.list.Dispatch(gx, gy, gz);
            self.list.ResourceBarrier(&[
                uav_barrier_all(),
                transition_barrier(
                    &out_uav,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                ),
            ]);
            self.list.CopyResource(&out_rb, &out_uav);
            self.list.Close().map_err(|e| format!("Close list: {e}"))?;

            let cmd: ID3D12CommandList = self.list.cast().map_err(|e| format!("cast list: {e}"))?;
            self.queue.ExecuteCommandLists(&[Some(cmd)]);
            self.queue
                .Signal(&self.fence, next_fence)
                .map_err(|e| format!("Signal: {e}"))?;
            if self.fence.GetCompletedValue() < next_fence {
                self.fence
                    .SetEventOnCompletion(next_fence, self.event)
                    .map_err(|e| format!("SetEventOnCompletion: {e}"))?;
                WaitForSingleObject(self.event, INFINITE);
            }
        }
        self.fence_value = next_fence;
        let elapsed_us = started.elapsed().as_micros().max(1) as u64;

        if let Some((res, ptr, bucket)) = pooled_tiled {
            crate::upload_ring::return_upload(res, ptr, bucket, crate::host::finished_submission());
        }

        let gpu_bytes = readback_vec::<u8>(&out_rb, linear.len())?;
        linear.copy_from_slice(&gpu_bytes);

        DISPATCHES.fetch_add(1, Ordering::Relaxed);
        BYTES_DETILED.fetch_add(linear.len() as u64, Ordering::Relaxed);
        LAST_US.store(elapsed_us, Ordering::Relaxed);
        Ok(elapsed_us)
    }
}

impl Drop for GpuDetiler {
    fn drop(&mut self) {
        if !self.event.is_invalid() {
            // SAFETY: owned Win32 event handle closed once on Drop.
            unsafe {
                let _ = CloseHandle(self.event);
            }
        }
    }
}

// SAFETY: `GpuDetiler` and its Win32 event/COM objects are accessed exclusively under `SHARED_DETILER`.
unsafe impl Send for GpuDetiler {}

fn supported_swizzle(s: Swizzle) -> bool {
    matches!(
        s,
        Swizzle::LINEAR
            | Swizzle::SW_256B_S
            | Swizzle::SW_256B_D
            | Swizzle::SW_4KB_S
            | Swizzle::SW_4KB_D
            | Swizzle::SW_64KB_S
            | Swizzle::SW_64KB_D
            | Swizzle::SW_64KB_S_X
            | Swizzle::SW_64KB_D_X
            | Swizzle::SW_64KB_R_X
            | Swizzle::SW_64KB_Z_X
    )
}

static SHARED_DETILER: Mutex<Option<GpuDetiler>> = Mutex::new(None);

/// Attempts GPU detile using a lazy host-device [`GpuDetiler`]; returns `false` on unsupported inputs
/// so the caller falls back to CPU `Layout::detile`.
pub fn try_detile_to_slice(
    layout: &Layout,
    tiled: &[u8],
    sub: DetileSubresource,
    linear: &mut [u8],
) -> bool {
    let Some(host) = crate::host::host() else {
        return false;
    };
    let mut guard = SHARED_DETILER.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        let Ok(detiler) = GpuDetiler::new(&host.device) else {
            return false;
        };
        *guard = Some(detiler);
    }
    let Some(detiler) = guard.as_mut() else {
        return false;
    };
    detiler.detile_to_slice(layout, tiled, sub, linear).is_ok()
}

fn compile_cs51(source: &str) -> Result<Vec<u8>, String> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: `source` and static literals remain valid across `D3DCompile`.
    let hr = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            s!("rea003_detile"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = hr {
        let msg = errors
            .as_ref()
            .map(|b| unsafe {
                let p = b.GetBufferPointer().cast::<u8>();
                String::from_utf8_lossy(std::slice::from_raw_parts(p, b.GetBufferSize()))
                    .into_owned()
            })
            .unwrap_or_default();
        return Err(format!("D3DCompile(gpu_detile): {e} {msg}"));
    }
    let blob = code.ok_or_else(|| "D3DCompile returned empty blob".to_owned())?;
    // SAFETY: valid blob pointer/size until vec copy finishes.
    Ok(unsafe {
        std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
            .to_vec()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_formats::texture::Desc;

    fn pattern_bytes(len: usize) -> Vec<u8> {
        (0..len)
            .map(|i| ((i.wrapping_mul(37)) ^ (i >> 5) ^ 0xA5) as u8)
            .collect()
    }

    #[test]
    fn gpu_detile_defaults_off_and_matches_cpu_across_modes_and_element_sizes() {
        assert!(!gpu_detile_enabled());
        let Some(host) = crate::host::host() else {
            return;
        };
        let mut detiler = GpuDetiler::new(&host.device).expect("GpuDetiler::new");

        let swizzles = [
            Swizzle::LINEAR,
            Swizzle::SW_256B_S,
            Swizzle::SW_4KB_S,
            Swizzle::SW_64KB_S,
            Swizzle::SW_64KB_D,
        ];
        for swizzle in swizzles {
            for elem in [1u32, 2, 4, 8, 16] {
                let layout = Layout::new(&Desc {
                    dimension: Dimension::Tex2d,
                    width: 67,
                    height: 39,
                    depth: 1,
                    mips: 6,
                    element_bytes: elem,
                    block: 1,
                    samples: 1,
                    swizzle,
                })
                .expect("layout");
                let tiled = pattern_bytes(layout.size as usize);

                // Validate both mip 0 (full) and tail mip (last mip in tail / small).
                for mip in [0u32, layout.mips.len() as u32 - 1] {
                    let m = &layout.mips[mip as usize];
                    let row_pitch = ((m.width * elem).max(4) as usize).next_multiple_of(256);
                    let depth_pitch = row_pitch * m.height as usize;

                    let mut cpu_out = vec![0xCCu8; depth_pitch];
                    layout
                        .detile(&tiled, mip, 0, &mut cpu_out, row_pitch, depth_pitch)
                        .expect("CPU detile");

                    let mut gpu_out = vec![0xCCu8; depth_pitch];
                    detiler
                        .detile_to_slice(
                            &layout,
                            &tiled,
                            DetileSubresource {
                                mip,
                                slice: 0,
                                row_pitch,
                                depth_pitch,
                            },
                            &mut gpu_out,
                        )
                        .expect("GPU detile");

                    assert_eq!(
                        gpu_out, cpu_out,
                        "mismatch swizzle={swizzle:?} elem={elem} mip={mip} tail={:?}",
                        m.tail
                    );
                }
            }
        }

        // Measure throughput on a realistic 1024x1024 RGBA8 (4B) SW_64KB_S texture.
        let bench_layout = Layout::new(&Desc {
            dimension: Dimension::Tex2d,
            width: 1024,
            height: 1024,
            depth: 1,
            mips: 1,
            element_bytes: 4,
            block: 1,
            samples: 1,
            swizzle: Swizzle::SW_64KB_S,
        })
        .expect("bench layout");
        let bench_tiled = pattern_bytes(bench_layout.size as usize);
        let row_pitch = 1024 * 4;
        let depth_pitch = row_pitch * 1024;
        let mut cpu_bench = vec![0u8; depth_pitch];
        let mut gpu_bench = vec![0u8; depth_pitch];

        let cpu_t0 = Instant::now();
        bench_layout
            .detile(&bench_tiled, 0, 0, &mut cpu_bench, row_pitch, depth_pitch)
            .unwrap();
        let cpu_us = cpu_t0.elapsed().as_micros().max(1);

        let gpu_us = detiler
            .detile_to_slice(
                &bench_layout,
                &bench_tiled,
                DetileSubresource {
                    mip: 0,
                    slice: 0,
                    row_pitch,
                    depth_pitch,
                },
                &mut gpu_bench,
            )
            .unwrap();
        assert_eq!(gpu_bench, cpu_bench);
        println!(
            "GPU_DETILE_1024_RGBA8_SW64KBS cpu_us={cpu_us} gpu_us={gpu_us} stats={:?}",
            stats()
        );
    }
}
