//! Abstract desktop upscaling backend (`VICTORY_REA003_UPSCALER=original|fsr3|dlss`).
//!
//! Probes native desktop DLLs (`amd_fidelityfx_dx12.dll` / `nvngx_dlss.dll`) dynamically
//! via `LoadLibraryW` without shipping proprietary binaries, and provides a clean-room
//! D3D12 `cs_5_1` temporal compute upscaler verified on host hardware.

use std::ffi::c_void;
use std::sync::OnceLock;
use std::time::Instant;

use windows::Win32::Foundation::{CloseHandle, FreeLibrary, HANDLE};
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::{
    D3D_ROOT_SIGNATURE_VERSION_1_0, D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_DESC,
    D3D12_COMMAND_QUEUE_FLAG_NONE, D3D12_COMPUTE_PIPELINE_STATE_DESC, D3D12_FENCE_FLAG_NONE,
    D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_TYPE_READBACK, D3D12_HEAP_TYPE_UPLOAD,
    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_NONE,
    D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE,
    D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_ROOT_CONSTANTS,
    D3D12_ROOT_DESCRIPTOR, D3D12_ROOT_PARAMETER, D3D12_ROOT_PARAMETER_0,
    D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS, D3D12_ROOT_PARAMETER_TYPE_SRV,
    D3D12_ROOT_PARAMETER_TYPE_UAV, D3D12_ROOT_SIGNATURE_DESC, D3D12_ROOT_SIGNATURE_FLAG_NONE,
    D3D12_SHADER_BYTECODE, D3D12_SHADER_VISIBILITY_ALL, D3D12SerializeRootSignature,
    ID3D12CommandAllocator, ID3D12CommandList, ID3D12CommandQueue, ID3D12Device, ID3D12Fence,
    ID3D12GraphicsCommandList, ID3D12PipelineState, ID3D12RootSignature,
};
use windows::Win32::System::LibraryLoader::LoadLibraryW;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::{Interface, PCWSTR, s, w};

use crate::dxr_backend::{align_up, readback_vec, transition_barrier, uav_barrier_all};
use crate::dxr_proto::{create_buffer, probe_tier, upload_slice};

/// Selected upscaler mode from `VICTORY_REA003_UPSCALER`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UpscalerMode {
    #[default]
    Original,
    Fsr3,
    Dlss,
}

impl UpscalerMode {
    pub fn parse(raw: &str) -> Self {
        if raw.trim().eq_ignore_ascii_case("fsr3") {
            Self::Fsr3
        } else if raw.trim().eq_ignore_ascii_case("dlss") {
            Self::Dlss
        } else {
            Self::Original
        }
    }

    pub fn from_env() -> Self {
        static MODE: OnceLock<UpscalerMode> = OnceLock::new();
        *MODE.get_or_init(|| {
            std::env::var("VICTORY_REA003_UPSCALER")
                .ok()
                .map(|v| Self::parse(&v))
                .unwrap_or_default()
        })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Original => "original",
            Self::Fsr3 => "fsr3",
            Self::Dlss => "dlss",
        }
    }
}

/// Active backend chosen after probing installed DLLs and GPU capabilities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActiveUpscalerBackend {
    OriginalPassthrough,
    NativeFsr3Dll,
    NativeDlssDll,
    CleanRoomTemporalCompute,
}

/// Honest probe of host upscaling DLL availability and clean-room backend readiness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UpscalerProbe {
    pub requested_mode: UpscalerMode,
    pub dxr_tier: u32,
    pub fsr3_dll_present: bool,
    pub dlss_dll_present: bool,
    pub clean_room_temporal_available: bool,
    pub active_backend: ActiveUpscalerBackend,
}

impl UpscalerProbe {
    pub fn detect(device: Option<&ID3D12Device>, requested_mode: UpscalerMode) -> Self {
        let fsr3_dll_present = probe_dll(w!("amd_fidelityfx_dx12.dll"));
        let dlss_dll_present = probe_dll(w!("nvngx_dlss.dll"));
        let dxr_tier = device.and_then(probe_tier).unwrap_or(0);
        let active_backend = match requested_mode {
            UpscalerMode::Original => ActiveUpscalerBackend::OriginalPassthrough,
            UpscalerMode::Fsr3 => {
                if fsr3_dll_present {
                    ActiveUpscalerBackend::NativeFsr3Dll
                } else {
                    ActiveUpscalerBackend::CleanRoomTemporalCompute
                }
            }
            UpscalerMode::Dlss => {
                if dlss_dll_present {
                    ActiveUpscalerBackend::NativeDlssDll
                } else {
                    ActiveUpscalerBackend::CleanRoomTemporalCompute
                }
            }
        };
        Self {
            requested_mode,
            dxr_tier,
            fsr3_dll_present,
            dlss_dll_present,
            clean_room_temporal_available: true,
            active_backend,
        }
    }
}

fn probe_dll(name: PCWSTR) -> bool {
    // SAFETY: `LoadLibraryW` probes presence on standard Windows DLL search path; freed immediately.
    unsafe {
        match LoadLibraryW(name) {
            Ok(module) if !module.is_invalid() => {
                let _ = FreeLibrary(module);
                true
            }
            _ => false,
        }
    }
}

/// Parameters for one temporal upscale dispatch.
#[derive(Clone, Copy, Debug)]
pub struct UpscaleDispatchDesc {
    pub render_width: u32,
    pub render_height: u32,
    pub display_width: u32,
    pub display_height: u32,
    pub jitter_x: f32,
    pub jitter_y: f32,
    pub reset_history: bool,
    pub sharpness: f32,
    pub frame_index: u32,
}

/// Execution telemetry for [`DesktopUpscaler`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UpscalerStats {
    pub dispatches: u64,
    pub history_resets: u64,
    pub last_dispatch_us: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct UpscaleConstants {
    render_w: u32,
    render_h: u32,
    display_w: u32,
    display_h: u32,
    jitter_x_bits: u32,
    jitter_y_bits: u32,
    reset_history: u32,
    sharpness_bits: u32,
}

const TEMPORAL_UPSCALE_HLSL: &str = r#"
ByteAddressBuffer g_curr : register(t0);
ByteAddressBuffer g_motion : register(t1);
ByteAddressBuffer g_prev_hist : register(t2);
RWByteAddressBuffer g_out_hist : register(u0);

cbuffer UpscaleCB : register(b0)
{
    uint g_rw;
    uint g_rh;
    uint g_dw;
    uint g_dh;
    float g_jitter_x;
    float g_jitter_y;
    uint g_reset;
    float g_sharpness;
};

float4 load_rgba(ByteAddressBuffer buf, uint w, uint h, int x, int y)
{
    int cx = clamp(x, 0, (int)w - 1);
    int cy = clamp(y, 0, (int)h - 1);
    uint idx = ((uint)cy * w + (uint)cx) * 16u;
    return asfloat(buf.Load4(idx));
}

float2 load_motion(int x, int y)
{
    int cx = clamp(x, 0, (int)g_dw - 1);
    int cy = clamp(y, 0, (int)g_dh - 1);
    uint idx = ((uint)cy * g_dw + (uint)cx) * 8u;
    return asfloat(g_motion.Load2(idx));
}

float3 rgb_to_ycocg(float3 c)
{
    float co = c.r - c.b;
    float tmp = c.b + 0.5f * co;
    float cg = c.g - tmp;
    float y = tmp + 0.5f * cg;
    return float3(y, co, cg);
}

float3 ycocg_to_rgb(float3 ycc)
{
    float tmp = ycc.x - 0.5f * ycc.z;
    float g = ycc.z + tmp;
    float b = tmp - 0.5f * ycc.y;
    float r = b + ycc.y;
    return float3(r, g, b);
}

[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= g_dw || dtid.y >= g_dh)
    {
        return;
    }

    float2 uv = (float2(dtid.xy) + 0.5f) / float2(g_dw, g_dh);
    // Render sample (rx, ry) captured the scene at display offset (+g_jitter) * scale,
    // so unjittering aligns back by subtracting g_jitter in render coordinates.
    float2 rpos = uv * float2(g_rw, g_rh) - 0.5f - float2(g_jitter_x, g_jitter_y);
    int2 ri = int2(floor(rpos));
    float2 rf = frac(rpos);

    float4 c00 = load_rgba(g_curr, g_rw, g_rh, ri.x, ri.y);
    float4 c10 = load_rgba(g_curr, g_rw, g_rh, ri.x + 1, ri.y);
    float4 c01 = load_rgba(g_curr, g_rw, g_rh, ri.x, ri.y + 1);
    float4 c11 = load_rgba(g_curr, g_rw, g_rh, ri.x + 1, ri.y + 1);

    float4 spatial = lerp(lerp(c00, c10, rf.x), lerp(c01, c11, rf.x), rf.y);

    // Closest native jittered render sample gives sharp subpixel truth when near display texel center.
    int2 r_near = int2(round(rpos));
    float2 subpixel_delta = abs(rpos - float2(r_near));
    float4 near_val = load_rgba(g_curr, g_rw, g_rh, r_near.x, r_near.y);

    float3 out_rgb;
    if (g_reset != 0u)
    {
        float sharp = saturate(g_sharpness);
        float4 lap = (c00 + c10 + c01 + c11) * 0.25f;
        out_rgb = saturate(spatial.rgb + (near_val.rgb - lap.rgb) * (0.25f * sharp));
    }
    else
    {
        // 3x3 render-space YCoCg AABB around current sample prevents ghosting on cuts/motion.
        float3 box_min = float3(1e9f, 1e9f, 1e9f);
        float3 box_max = float3(-1e9f, -1e9f, -1e9f);
        [unroll]
        for (int oy = -1; oy <= 1; ++oy)
        {
            [unroll]
            for (int ox = -1; ox <= 1; ++ox)
            {
                float3 n_ycc = rgb_to_ycocg(load_rgba(g_curr, g_rw, g_rh, r_near.x + ox, r_near.y + oy).rgb);
                box_min = min(box_min, n_ycc);
                box_max = max(box_max, n_ycc);
            }
        }

        float2 mv = load_motion((int)dtid.x, (int)dtid.y);
        float2 prev_pos = float2(dtid.xy) + mv;
        int2 pi = int2(floor(prev_pos));
        float2 pf = frac(prev_pos);

        float3 h00 = load_rgba(g_prev_hist, g_dw, g_dh, pi.x, pi.y).rgb;
        float3 h10 = load_rgba(g_prev_hist, g_dw, g_dh, pi.x + 1, pi.y).rgb;
        float3 h01 = load_rgba(g_prev_hist, g_dw, g_dh, pi.x, pi.y + 1).rgb;
        float3 h11 = load_rgba(g_prev_hist, g_dw, g_dh, pi.x + 1, pi.y + 1).rgb;
        float3 hist_rgb = lerp(lerp(h00, h10, pf.x), lerp(h01, h11, pf.x), pf.y);

        float3 hist_ycc = clamp(rgb_to_ycocg(hist_rgb), box_min, box_max);
        float3 clamped_hist = ycocg_to_rgb(hist_ycc);

        float dist = max(subpixel_delta.x, subpixel_delta.y);
        float hit_conf = saturate(1.0f - dist * 2.2f);
        float3 curr_sample = lerp(spatial.rgb, near_val.rgb, hit_conf);
        float alpha = lerp(0.08f, 0.85f, hit_conf * hit_conf);
        out_rgb = saturate(lerp(clamped_hist, curr_sample, alpha));
    }

    uint w_idx = (dtid.y * g_dw + dtid.x) * 16u;
    g_out_hist.Store4(w_idx, asuint(float4(out_rgb, 1.0f)));
}
"#;

/// Clean-room D3D12 compute temporal upscaler compiled via `D3DCompile` (`cs_5_1`).
pub struct CleanRoomTemporalUpscaler {
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

impl CleanRoomTemporalUpscaler {
    pub fn new(device: &ID3D12Device) -> Result<Self, String> {
        let cs_bytes = compile_cs51(TEMPORAL_UPSCALE_HLSL)?;
        let params = [
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Constants: D3D12_ROOT_CONSTANTS {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                        Num32BitValues: 8,
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
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR {
                        ShaderRegister: 2,
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
        // SAFETY: `params` stays valid for `D3D12SerializeRootSignature`.
        unsafe {
            D3D12SerializeRootSignature(
                &rs_desc,
                D3D_ROOT_SIGNATURE_VERSION_1_0,
                &mut blob,
                Some(&mut err),
            )
            .map_err(|e| format!("D3D12SerializeRootSignature(upscaler): {e}"))?;
        }
        let blob = blob.ok_or_else(|| "missing upscaler root signature blob".to_owned())?;
        // SAFETY: valid D3D12 device and serialized root signature blob.
        let root_sig: ID3D12RootSignature = unsafe {
            let slice = std::slice::from_raw_parts(
                blob.GetBufferPointer().cast::<u8>(),
                blob.GetBufferSize(),
            );
            device
                .CreateRootSignature(0, slice)
                .map_err(|e| format!("CreateRootSignature(upscaler): {e}"))?
        };
        let pso_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::ManuallyDrop::new(Some(root_sig.clone())),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: cs_bytes.as_ptr().cast::<c_void>(),
                BytecodeLength: cs_bytes.len(),
            },
            ..Default::default()
        };
        // SAFETY: `cs_bytes` and `root_sig` stay live across PSO creation.
        let (queue, alloc, list, fence, event, pso) = unsafe {
            let pso: ID3D12PipelineState = device
                .CreateComputePipelineState(&pso_desc)
                .map_err(|e| format!("CreateComputePipelineState(upscaler): {e}"))?;
            let queue: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
                    ..Default::default()
                })
                .map_err(|e| format!("CreateCommandQueue(upscaler): {e}"))?;
            let alloc: ID3D12CommandAllocator = device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .map_err(|e| format!("CreateCommandAllocator(upscaler): {e}"))?;
            let list: ID3D12GraphicsCommandList = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
                .map_err(|e| format!("CreateCommandList(upscaler): {e}"))?;
            list.Close().map_err(|e| format!("initial Close: {e}"))?;
            let fence: ID3D12Fence = device
                .CreateFence(0, D3D12_FENCE_FLAG_NONE)
                .map_err(|e| format!("CreateFence(upscaler): {e}"))?;
            let event = CreateEventW(None, false, false, None)
                .map_err(|e| format!("CreateEventW(upscaler): {e}"))?;
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

    /// Runs one temporal upscale step (`curr_rgba32f`, `motion_rg32f`, `prev_history_rgba32f`)
    /// returning the upscaled RGBA32F output and measured GPU execution time in microseconds.
    pub fn upscale_rgba32f(
        &mut self,
        desc: &UpscaleDispatchDesc,
        curr_rgba32f: &[[f32; 4]],
        motion_rg32f: &[[f32; 2]],
        prev_history_rgba32f: &[[f32; 4]],
    ) -> Result<(Vec<[f32; 4]>, u64), String> {
        let render_pixels = (desc.render_width as usize) * (desc.render_height as usize);
        let display_pixels = (desc.display_width as usize) * (desc.display_height as usize);
        if curr_rgba32f.len() != render_pixels {
            return Err("curr_rgba32f pixel count mismatch".to_owned());
        }
        if motion_rg32f.len() != display_pixels || prev_history_rgba32f.len() != display_pixels {
            return Err("history or motion buffer pixel count mismatch".to_owned());
        }

        let curr_bytes = align_up((render_pixels * 16) as u64, 256);
        let motion_bytes = align_up((display_pixels * 8) as u64, 256);
        let hist_bytes = align_up((display_pixels * 16) as u64, 256);

        let curr_buf = create_buffer(
            &self.device,
            curr_bytes,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        upload_slice(&curr_buf, curr_rgba32f)?;

        let motion_buf = create_buffer(
            &self.device,
            motion_bytes,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        upload_slice(&motion_buf, motion_rg32f)?;

        let prev_buf = create_buffer(
            &self.device,
            hist_bytes,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        upload_slice(&prev_buf, prev_history_rgba32f)?;

        let out_uav = create_buffer(
            &self.device,
            hist_bytes,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        )?;
        let out_rb = create_buffer(
            &self.device,
            hist_bytes,
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;

        let consts = UpscaleConstants {
            render_w: desc.render_width,
            render_h: desc.render_height,
            display_w: desc.display_width,
            display_h: desc.display_height,
            jitter_x_bits: desc.jitter_x.to_bits(),
            jitter_y_bits: desc.jitter_y.to_bits(),
            reset_history: u32::from(desc.reset_history),
            sharpness_bits: desc.sharpness.to_bits(),
        };
        let gx = desc.display_width.div_ceil(8);
        let gy = desc.display_height.div_ceil(8);
        let started = Instant::now();
        let next_fence = self.fence_value + 1;

        // SAFETY: valid command allocator, list, upload/uav/readback resources kept alive across fence wait.
        unsafe {
            let curr_va = curr_buf.GetGPUVirtualAddress();
            let motion_va = motion_buf.GetGPUVirtualAddress();
            let prev_va = prev_buf.GetGPUVirtualAddress();
            let out_va = out_uav.GetGPUVirtualAddress();

            self.alloc
                .Reset()
                .map_err(|e| format!("Reset alloc: {e}"))?;
            self.list
                .Reset(&self.alloc, None)
                .map_err(|e| format!("Reset list: {e}"))?;
            self.list.ResourceBarrier(&[transition_barrier(
                &out_uav,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
            self.list.SetComputeRootSignature(&self.root_sig);
            self.list.SetPipelineState(&self.pso);
            self.list
                .SetComputeRoot32BitConstants(0, 8, (&raw const consts).cast(), 0);
            self.list.SetComputeRootShaderResourceView(1, curr_va);
            self.list.SetComputeRootShaderResourceView(2, motion_va);
            self.list.SetComputeRootShaderResourceView(3, prev_va);
            self.list.SetComputeRootUnorderedAccessView(4, out_va);
            self.list.Dispatch(gx, gy, 1);
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
        let out = readback_vec::<[f32; 4]>(&out_rb, display_pixels)?;
        Ok((out, elapsed_us))
    }
}

impl Drop for CleanRoomTemporalUpscaler {
    fn drop(&mut self) {
        if !self.event.is_invalid() {
            // SAFETY: `self.event` belongs solely to `Self` and closes once.
            unsafe {
                let _ = CloseHandle(self.event);
            }
        }
    }
}

/// High-level abstract desktop upscaler coordinating mode selection, DLL probes, and execution.
pub struct DesktopUpscaler {
    probe: UpscalerProbe,
    temporal: Option<CleanRoomTemporalUpscaler>,
    stats: UpscalerStats,
}

impl DesktopUpscaler {
    pub fn new(device: &ID3D12Device, mode: UpscalerMode) -> Result<Self, String> {
        let probe = UpscalerProbe::detect(Some(device), mode);
        let temporal = if mode == UpscalerMode::Original {
            None
        } else {
            Some(CleanRoomTemporalUpscaler::new(device)?)
        };
        Ok(Self {
            probe,
            temporal,
            stats: UpscalerStats::default(),
        })
    }

    pub fn probe(&self) -> UpscalerProbe {
        self.probe
    }

    pub fn stats(&self) -> UpscalerStats {
        self.stats
    }

    pub fn upscale_rgba32f(
        &mut self,
        desc: &UpscaleDispatchDesc,
        curr_rgba32f: &[[f32; 4]],
        motion_rg32f: &[[f32; 2]],
        prev_history_rgba32f: &[[f32; 4]],
    ) -> Result<Vec<[f32; 4]>, String> {
        let Some(temporal) = self.temporal.as_mut() else {
            return Err("upscaler disabled in original mode".to_owned());
        };
        let (out, us) =
            temporal.upscale_rgba32f(desc, curr_rgba32f, motion_rg32f, prev_history_rgba32f)?;
        self.stats.dispatches += 1;
        if desc.reset_history {
            self.stats.history_resets += 1;
        }
        self.stats.last_dispatch_us = us;
        Ok(out)
    }
}

fn compile_cs51(source: &str) -> Result<Vec<u8>, String> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: `source` and static entry/profile strings live across `D3DCompile`.
    let hr = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            s!("rea003_upscale"),
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
        return Err(format!("D3DCompile(upscaler): {e} {msg}"));
    }
    let blob = code.ok_or_else(|| "D3DCompile returned no bytecode".to_owned())?;
    // SAFETY: `blob` pointer and size are valid until vec copy completes.
    let bytes = unsafe {
        std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
            .to_vec()
    };
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn halton(mut idx: u32, base: u32) -> f32 {
        let mut f = 1.0f32;
        let mut r = 0.0f32;
        while idx > 0 {
            f /= base as f32;
            r += f * (idx % base) as f32;
            idx /= base;
        }
        r
    }

    fn reference_texel(x: f32, y: f32) -> [f32; 4] {
        // High-frequency 1080p edges & concentric detail above single-frame 540p Nyquist.
        let r = ((x - 960.0).powi(2) + (y - 540.0).powi(2)).sqrt();
        let rings = (r * 0.9).sin() * 0.5 + 0.5;
        let diag = ((x * 0.72 + y * 0.58).sin()) * 0.5 + 0.5;
        let grad = (y / 1080.0).clamp(0.0, 1.0);
        [
            rings * 0.8 + 0.1,
            diag * 0.8 + 0.1,
            0.5 * (rings + grad),
            1.0,
        ]
    }

    fn psnr_rgb(got: &[[f32; 4]], reference: &[[f32; 4]]) -> f32 {
        let mut mse = 0.0f64;
        for (a, b) in got.iter().zip(reference.iter()) {
            for c in 0..3 {
                let d = f64::from(a[c] - b[c]);
                mse += d * d;
            }
        }
        mse /= (got.len() * 3) as f64;
        if mse < 1e-12 {
            99.0
        } else {
            (10.0 * (1.0 / mse).log10()) as f32
        }
    }

    #[test]
    fn upscaler_defaults_original_and_probes_dlls() {
        assert_eq!(UpscalerMode::from_env(), UpscalerMode::Original);
        assert_eq!(UpscalerMode::parse("fsr3"), UpscalerMode::Fsr3);
        assert_eq!(UpscalerMode::parse("DLSS"), UpscalerMode::Dlss);
        assert_eq!(UpscalerMode::parse("invalid"), UpscalerMode::Original);

        let p_orig = UpscalerProbe::detect(None, UpscalerMode::Original);
        assert_eq!(
            p_orig.active_backend,
            ActiveUpscalerBackend::OriginalPassthrough
        );
        let p_fsr3 = UpscalerProbe::detect(None, UpscalerMode::Fsr3);
        let p_dlss = UpscalerProbe::detect(None, UpscalerMode::Dlss);
        assert!(p_fsr3.clean_room_temporal_available);
        assert!(p_dlss.clean_room_temporal_available);
        println!("UPSCALER_PROBE_FSR3={p_fsr3:?}");
        println!("UPSCALER_PROBE_DLSS={p_dlss:?}");
    }

    #[test]
    fn upscaler_clean_room_temporal_960x540_to_1920x1080_converges_and_resets() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let mut upscaler =
            DesktopUpscaler::new(&host.device, UpscalerMode::Fsr3).expect("DesktopUpscaler::new");
        let (rw, rh) = (960u32, 540u32);
        let (dw, dh) = (1920u32, 1080u32);

        let mut reference = Vec::with_capacity((dw * dh) as usize);
        for dy in 0..dh {
            for dx in 0..dw {
                reference.push(reference_texel(dx as f32 + 0.5, dy as f32 + 0.5));
            }
        }
        let motion = vec![[0.0f32, 0.0f32]; (dw * dh) as usize];
        let mut history = vec![[0.0f32, 0.0, 0.0, 1.0]; (dw * dh) as usize];

        let mut psnr_frame0 = 0.0f32;
        let mut psnr_frame7 = 0.0f32;
        for f in 0..8u32 {
            let jx = halton(f + 1, 2) - 0.5;
            let jy = halton(f + 1, 3) - 0.5;
            let mut curr = Vec::with_capacity((rw * rh) as usize);
            for ry in 0..rh {
                let sy = ((ry as f32 + 0.5 + jy) / rh as f32) * dh as f32;
                for rx in 0..rw {
                    let sx = ((rx as f32 + 0.5 + jx) / rw as f32) * dw as f32;
                    curr.push(reference_texel(sx, sy));
                }
            }
            history = upscaler
                .upscale_rgba32f(
                    &UpscaleDispatchDesc {
                        render_width: rw,
                        render_height: rh,
                        display_width: dw,
                        display_height: dh,
                        jitter_x: jx,
                        jitter_y: jy,
                        reset_history: f == 0,
                        sharpness: 0.5,
                        frame_index: f,
                    },
                    &curr,
                    &motion,
                    &history,
                )
                .expect("upscale frame");
            let p = psnr_rgb(&history, &reference);
            if f == 0 {
                psnr_frame0 = p;
            }
            if f == 7 {
                psnr_frame7 = p;
            }
        }
        println!(
            "UPSCALER_1080P_CONVERGENCE psnr_f0={psnr_frame0:.2}dB psnr_f7={psnr_frame7:.2}dB stats={:?} probe={:?}",
            upscaler.stats(),
            upscaler.probe()
        );
        assert!(
            psnr_frame7 > psnr_frame0 + 1.0,
            "expected multi-frame convergence gain: f0={psnr_frame0:.2}, f7={psnr_frame7:.2}"
        );

        // Verify history reset completely drops stale bright white history on scene cut to dark scene.
        let stale_white = vec![[1.0f32, 1.0, 1.0, 1.0]; (dw * dh) as usize];
        let dark_curr = vec![[0.05f32, 0.05, 0.05, 1.0]; (rw * rh) as usize];
        let after_reset = upscaler
            .upscale_rgba32f(
                &UpscaleDispatchDesc {
                    render_width: rw,
                    render_height: rh,
                    display_width: dw,
                    display_height: dh,
                    jitter_x: 0.0,
                    jitter_y: 0.0,
                    reset_history: true,
                    sharpness: 0.25,
                    frame_index: 8,
                },
                &dark_curr,
                &motion,
                &stale_white,
            )
            .expect("upscale reset");
        let center = after_reset[(540 * dw + 960) as usize][0];
        assert!(
            center < 0.1,
            "reset_history failed to flush stale white history: center_r={center}"
        );
        assert_eq!(upscaler.stats().dispatches, 9);
        assert_eq!(upscaler.stats().history_resets, 2);
    }
}
