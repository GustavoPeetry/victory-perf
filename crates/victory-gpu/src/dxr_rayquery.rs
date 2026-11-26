//! Shader Model 6.5 `RayQuery` compute pipeline built via WinSDK `dxc.exe` (`-T cs_6_5`).
//!
//! Executes inline ray tracing directly against [`DxrTlas`] without mixing
//! native DXR GPU virtual addresses with console `BvhNodes::Memory` descriptors.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::{
    D3D_ROOT_SIGNATURE_VERSION_1_0, D3D12_COMPUTE_PIPELINE_STATE_DESC, D3D12_HEAP_TYPE_DEFAULT,
    D3D12_HEAP_TYPE_READBACK, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
    D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST,
    D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_GENERIC_READ,
    D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_ROOT_CONSTANTS, D3D12_ROOT_DESCRIPTOR,
    D3D12_ROOT_PARAMETER, D3D12_ROOT_PARAMETER_0, D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
    D3D12_ROOT_PARAMETER_TYPE_SRV, D3D12_ROOT_PARAMETER_TYPE_UAV, D3D12_ROOT_SIGNATURE_DESC,
    D3D12_ROOT_SIGNATURE_FLAG_NONE, D3D12_SHADER_BYTECODE, D3D12_SHADER_VISIBILITY_ALL,
    D3D12SerializeRootSignature, ID3D12Device, ID3D12PipelineState, ID3D12RootSignature,
};

use crate::dxr_backend::{
    AS_ALIGNMENT_BYTES, DxrBackend, DxrTlas, align_up, readback_vec, transition_barrier,
    uav_barrier_all,
};
use crate::dxr_proto::{create_buffer, upload_slice};

/// Semantic correspondence between RDNA2 `image_bvh*_intersect_ray` outputs and DXR 1.1 RayQuery.
#[derive(Clone, Copy, Debug)]
pub struct RayQuerySemanticMapping {
    pub rdna2_component: &'static str,
    pub dxr_rayquery_equivalent: &'static str,
    pub behavior: &'static str,
}

const SEMANTIC_MAPPING: &[RayQuerySemanticMapping] = &[
    RayQuerySemanticMapping {
        rdna2_component: "v[0] ray distance / result[0]",
        dxr_rayquery_equivalent: "CommittedRayT() when CommittedStatus() == COMMITTED_TRIANGLE_HIT",
        behavior: "Closest committed hit parametric t along unnormalized ray direction; falls back to TMax on miss.",
    },
    RayQuerySemanticMapping {
        rdna2_component: "v[1], v[2] barycentrics (bvh::RETURN_BARYCENTRICS)",
        dxr_rayquery_equivalent: "CommittedTriangleBarycentrics() (.x = u, .y = v)",
        behavior: "Standard triangle interpolation P = (1 - u - v)*v0 + u*v1 + v*v2 (v1-v0 weight u, v2-v0 weight v).",
    },
    RayQuerySemanticMapping {
        rdna2_component: "v[3] hit status / primitive / instance state",
        dxr_rayquery_equivalent: "CommittedPrimitiveIndex(), CommittedInstanceID(), CommittedGeometryIndex()",
        behavior: "Native BLAS triangle index and TLAS 24-bit InstanceID without console BVH4 pointer chasing.",
    },
    RayQuerySemanticMapping {
        rdna2_component: "s[rsrc:rsrc+3] word1 bit 31 (BOX_SORT)",
        dxr_rayquery_equivalent: "Hardware BVH traversal order + RAY_FLAG_* passed to TraceRayInline",
        behavior: "Hardware-accelerated traversal using per-ray flags and 8-bit InstanceInclusionMask.",
    },
    RayQuerySemanticMapping {
        rdna2_component: "image_bvh64_intersect_ray descriptor VA",
        dxr_rayquery_equivalent: "RaytracingAccelerationStructure g_tlas root SRV (TLAS GPU VA)",
        behavior: "Strictly isolated from title BVH4 addresses; uses native ID3D12Device5 AS handles.",
    },
];

pub fn semantic_mapping() -> &'static [RayQuerySemanticMapping] {
    SEMANTIC_MAPPING
}

/// 48-byte input ray matching the HLSL `ByteAddressBuffer g_rays` layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayInput {
    pub origin: [f32; 3],
    pub tmin: f32,
    pub dir: [f32; 3],
    pub tmax: f32,
    pub instance_mask: u32,
    pub ray_flags: u32,
    pub pad: [u32; 2],
}

impl RayInput {
    pub fn new(origin: [f32; 3], dir: [f32; 3], tmin: f32, tmax: f32) -> Self {
        Self {
            origin,
            tmin,
            dir,
            tmax,
            instance_mask: 0xFF,
            ray_flags: 0,
            pad: [0, 0],
        }
    }

    pub fn with_mask(mut self, mask: u8) -> Self {
        self.instance_mask = u32::from(mask);
        self
    }
}

/// 32-byte hit output matching the HLSL `RWByteAddressBuffer g_hits` layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RayHitOutput {
    pub hit: u32,
    pub t: f32,
    pub bary_u: f32,
    pub bary_v: f32,
    pub primitive_index: u32,
    pub instance_id: u32,
    pub geometry_index: u32,
    pub front_face: u32,
}

const RAYQUERY_CS_HLSL: &str = r#"
RaytracingAccelerationStructure g_tlas : register(t0);
ByteAddressBuffer g_rays : register(t1);
RWByteAddressBuffer g_hits : register(u0);

cbuffer TraceParams : register(b0)
{
    uint g_ray_count;
    uint g_pad0;
    uint g_pad1;
    uint g_pad2;
};

[numthreads(64, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    uint idx = dtid.x;
    if (idx >= g_ray_count)
    {
        return;
    }

    uint r_off = idx * 48u;
    uint4 r0 = g_rays.Load4(r_off + 0u);
    uint4 r1 = g_rays.Load4(r_off + 16u);
    uint2 r2 = g_rays.Load2(r_off + 32u);

    RayDesc ray;
    ray.Origin = asfloat(r0.xyz);
    ray.TMin = asfloat(r0.w);
    ray.Direction = asfloat(r1.xyz);
    ray.TMax = asfloat(r1.w);

    uint instance_mask = r2.x & 0xFFu;
    uint ray_flags = r2.y;

    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(g_tlas, ray_flags, instance_mask, ray);
    while (q.Proceed())
    {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            q.CommitNonOpaqueTriangleHit();
        }
    }

    uint out_hit = 0u;
    float out_t = ray.TMax;
    float out_u = 0.0f;
    float out_v = 0.0f;
    uint out_prim = 0u;
    uint out_inst = 0u;
    uint out_geom = 0u;
    uint out_ff = 0u;

    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        out_hit = 1u;
        out_t = q.CommittedRayT();
        float2 bary = q.CommittedTriangleBarycentrics();
        out_u = bary.x;
        out_v = bary.y;
        out_prim = q.CommittedPrimitiveIndex();
        out_inst = q.CommittedInstanceID();
        out_geom = q.CommittedGeometryIndex();
        out_ff = q.CommittedTriangleFrontFace() ? 1u : 0u;
    }

    uint w_off = idx * 32u;
    g_hits.Store4(w_off + 0u, uint4(out_hit, asuint(out_t), asuint(out_u), asuint(out_v)));
    g_hits.Store4(w_off + 16u, uint4(out_prim, out_inst, out_geom, out_ff));
}
"#;

static TEMP_SEQ: AtomicU64 = AtomicU64::new(0);

struct TempCleanup(Vec<PathBuf>);

impl Drop for TempCleanup {
    fn drop(&mut self) {
        for p in &self.0 {
            let _ = std::fs::remove_file(p);
        }
    }
}

/// Locates `dxc.exe` from `VICTORY_REA003_DXC_PATH`, Windows SDK 10 bin directories, or `PATH`.
pub fn find_dxc_exe() -> Option<PathBuf> {
    if let Ok(explicit) = std::env::var("VICTORY_REA003_DXC_PATH") {
        let p = PathBuf::from(explicit.trim());
        if p.is_file() {
            return Some(p);
        }
    }
    let known = [
        r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\dxc.exe",
        r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64\dxc.exe",
    ];
    for k in known {
        let p = Path::new(k);
        if p.is_file() {
            return Some(p.to_path_buf());
        }
    }
    let kits_bin = Path::new(r"C:\Program Files (x86)\Windows Kits\10\bin");
    if let Ok(entries) = std::fs::read_dir(kits_bin) {
        let mut versions: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path().join("x64").join("dxc.exe"))
            .filter(|p| p.is_file())
            .collect();
        versions.sort();
        if let Some(latest) = versions.pop() {
            return Some(latest);
        }
    }
    if let Ok(path_env) = std::env::var("PATH") {
        for dir in std::env::split_paths(&path_env) {
            let candidate = dir.join("dxc.exe");
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

/// Compiles `hlsl` with `dxc.exe -T <profile> -E <entry>` into validated DXIL container bytes.
pub fn compile_dxil_with_dxc(hlsl: &str, profile: &str, entry: &str) -> Result<Vec<u8>, String> {
    let dxc = find_dxc_exe().ok_or_else(|| "Windows SDK dxc.exe not found".to_owned())?;
    let id = TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let dir = std::env::temp_dir();
    let src = dir.join(format!("victory_rea003_rq_{pid}_{id}.hlsl"));
    let out = dir.join(format!("victory_rea003_rq_{pid}_{id}.dxil"));
    let _cleanup = TempCleanup(vec![src.clone(), out.clone()]);

    std::fs::write(&src, hlsl).map_err(|e| format!("write HLSL temp: {e}"))?;
    let output = Command::new(&dxc)
        .args(["-T", profile, "-E", entry, "-O3", "-Fo"])
        .arg(&out)
        .arg(&src)
        .output()
        .map_err(|e| format!("spawn {}: {e}", dxc.display()))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(format!(
            "dxc failed ({:?}): {stderr} {stdout}",
            output.status.code()
        ));
    }
    std::fs::read(&out).map_err(|e| format!("read DXIL temp: {e}"))
}

/// Compute pipeline executing Shader Model 6.5 `RayQuery` against a native [`DxrTlas`].
pub struct DxrRayQueryPipeline {
    root_sig: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    dxil_len: usize,
}

impl DxrRayQueryPipeline {
    /// Compiles the `cs_6_5` RayQuery shader and builds root signature + PSO on `device`.
    pub fn new(device: &ID3D12Device, tier: u32) -> Result<Self, String> {
        if tier < 11 {
            return Err(format!(
                "DXR 1.1 RayQuery requires tier >= 11 (adapter tier={tier})"
            ));
        }
        let dxil = compile_dxil_with_dxc(RAYQUERY_CS_HLSL, "cs_6_5", "main")?;
        let params = [
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Constants: D3D12_ROOT_CONSTANTS {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                        Num32BitValues: 4,
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
        // SAFETY: `params` stays pinned across `D3D12SerializeRootSignature`.
        unsafe {
            D3D12SerializeRootSignature(
                &rs_desc,
                D3D_ROOT_SIGNATURE_VERSION_1_0,
                &mut blob,
                Some(&mut err),
            )
            .map_err(|e| {
                let msg = err
                    .as_ref()
                    .map(|b| {
                        let p = b.GetBufferPointer().cast::<u8>();
                        let n = b.GetBufferSize();
                        String::from_utf8_lossy(std::slice::from_raw_parts(p, n)).into_owned()
                    })
                    .unwrap_or_default();
                format!("D3D12SerializeRootSignature: {e} {msg}")
            })?;
        }
        let blob = blob.ok_or_else(|| "missing root signature blob".to_owned())?;
        // SAFETY: serialized blob pointer/size belong to `blob` and live across `CreateRootSignature`.
        let root_sig: ID3D12RootSignature = unsafe {
            let slice = std::slice::from_raw_parts(
                blob.GetBufferPointer().cast::<u8>(),
                blob.GetBufferSize(),
            );
            device
                .CreateRootSignature(0, slice)
                .map_err(|e| format!("CreateRootSignature(RayQuery): {e}"))?
        };
        let pso_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::ManuallyDrop::new(Some(root_sig.clone())),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: dxil.as_ptr().cast(),
                BytecodeLength: dxil.len(),
            },
            ..Default::default()
        };
        // SAFETY: `dxil` and `root_sig` remain alive for the `CreateComputePipelineState` call.
        let pso: ID3D12PipelineState = unsafe {
            device
                .CreateComputePipelineState(&pso_desc)
                .map_err(|e| format!("CreateComputePipelineState(RayQuery): {e}"))?
        };
        Ok(Self {
            root_sig,
            pso,
            dxil_len: dxil.len(),
        })
    }

    pub fn dxil_byte_len(&self) -> usize {
        self.dxil_len
    }

    /// Dispatches `rays` against `tlas` on `backend` and returns GPU hit outputs in input order.
    pub fn trace_rays(
        &self,
        backend: &mut DxrBackend,
        tlas: &DxrTlas,
        rays: &[RayInput],
    ) -> Result<Vec<RayHitOutput>, String> {
        if rays.is_empty() {
            return Ok(Vec::new());
        }
        let ray_count = u32::try_from(rays.len()).map_err(|_| "ray count overflow")?;
        let in_bytes = align_up(
            u64::from(ray_count) * std::mem::size_of::<RayInput>() as u64,
            AS_ALIGNMENT_BYTES,
        );
        let out_bytes = align_up(
            u64::from(ray_count) * std::mem::size_of::<RayHitOutput>() as u64,
            AS_ALIGNMENT_BYTES,
        );
        let rays_buf = create_buffer(
            &backend.device,
            in_bytes,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        upload_slice(&rays_buf, rays)?;

        let hits_uav = create_buffer(
            &backend.device,
            out_bytes,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        )?;
        let hits_rb = create_buffer(
            &backend.device,
            out_bytes,
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;

        // SAFETY: `rays_buf` and `hits_uav` are valid buffers kept alive past fence wait.
        let (rays_va, hits_va) = unsafe {
            (
                rays_buf.GetGPUVirtualAddress(),
                hits_uav.GetGPUVirtualAddress(),
            )
        };
        let tlas_va = tlas.gpu_va;
        let groups = ray_count.div_ceil(64);
        let constants = [ray_count, 0u32, 0u32, 0u32];

        backend.submit_and_wait(|list4| {
            // SAFETY: valid root signature, compute PSO, root bindings, barriers, and readback copy.
            unsafe {
                list4.ResourceBarrier(&[transition_barrier(
                    &hits_uav,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )]);
                list4.SetComputeRootSignature(&self.root_sig);
                list4.SetPipelineState(&self.pso);
                list4.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
                list4.SetComputeRootShaderResourceView(1, tlas_va);
                list4.SetComputeRootShaderResourceView(2, rays_va);
                list4.SetComputeRootUnorderedAccessView(3, hits_va);
                list4.Dispatch(groups, 1, 1);
                list4.ResourceBarrier(&[
                    uav_barrier_all(),
                    transition_barrier(
                        &hits_uav,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    ),
                ]);
                list4.CopyResource(&hits_rb, &hits_uav);
            }
            Ok(())
        })?;

        readback_vec::<RayHitOutput>(&hits_rb, rays.len())
    }
}
