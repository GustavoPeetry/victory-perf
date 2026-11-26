//! Isolated experimental DXR 1.1 / FFX (`slot=0x218`) prototype (P2).
//!
//! Strictly gated behind `VICTORY_REA_DXR_PROTO=1` and `VICTORY_REA_FFX_TRACE=1`.
//! Both switches default to OFF so normal title execution stays bit-identical.

use std::ffi::c_void;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use windows::Win32::Foundation::CloseHandle;
use windows::Win32::Graphics::Direct3D12::{
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC,
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS, D3D12_COMMAND_LIST_TYPE_DIRECT,
    D3D12_COMMAND_QUEUE_DESC, D3D12_COMMAND_QUEUE_FLAG_NONE, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
    D3D12_ELEMENTS_LAYOUT_ARRAY, D3D12_FEATURE_D3D12_OPTIONS5,
    D3D12_FEATURE_DATA_D3D12_OPTIONS5, D3D12_FENCE_FLAG_NONE, D3D12_HEAP_FLAG_NONE,
    D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_TYPE_UPLOAD,
    D3D12_MEMORY_POOL_UNKNOWN, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO,
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL, D3D12_RAYTRACING_GEOMETRY_DESC,
    D3D12_RAYTRACING_GEOMETRY_DESC_0, D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
    D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC, D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
    D3D12_RAYTRACING_INSTANCE_DESC, D3D12_RAYTRACING_TIER_1_0, D3D12_RAYTRACING_TIER_1_1,
    D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_BARRIER_0, D3D12_RESOURCE_BARRIER_FLAG_NONE,
    D3D12_RESOURCE_BARRIER_TYPE_UAV, D3D12_RESOURCE_DESC, D3D12_RESOURCE_DIMENSION_BUFFER,
    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_NONE,
    D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_GENERIC_READ,
    D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
    D3D12_RESOURCE_UAV_BARRIER, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ID3D12CommandAllocator,
    ID3D12CommandQueue, ID3D12Device, ID3D12Device5, ID3D12Fence, ID3D12GraphicsCommandList4,
    ID3D12Resource,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT_R32G32B32_FLOAT, DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC,
};
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::Interface;

static DXR_PROTO_ENABLED: OnceLock<bool> = OnceLock::new();
static FFX_TRACE_ENABLED: OnceLock<bool> = OnceLock::new();
static PROBED_ONCE: std::sync::Once = std::sync::Once::new();

static DXR_TIER: AtomicU32 = AtomicU32::new(0);
static BLAS_BUILDS: AtomicU64 = AtomicU64::new(0);
static TLAS_BUILDS: AtomicU64 = AtomicU64::new(0);
static BLAS_BYTES: AtomicU64 = AtomicU64::new(0);
static TLAS_BYTES: AtomicU64 = AtomicU64::new(0);
static LAST_BUILD_US: AtomicU64 = AtomicU64::new(0);
static FFX_0X218_CALLS: AtomicU64 = AtomicU64::new(0);

/// Snapshot of experimental DXR / FFX telemetry counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DxrStats {
    pub tier: u32,
    pub blas_builds: u64,
    pub tlas_builds: u64,
    pub blas_bytes: u64,
    pub tlas_bytes: u64,
    pub last_build_us: u64,
    pub ffx_0x218_calls: u64,
}

/// Returns true when `VICTORY_REA_DXR_PROTO=1` is active.
pub fn dxr_proto_enabled() -> bool {
    *DXR_PROTO_ENABLED
        .get_or_init(|| std::env::var_os("VICTORY_REA_DXR_PROTO").is_some_and(|v| v == "1"))
}

/// Returns true when `VICTORY_REA_FFX_TRACE=1` is active.
pub fn ffx_trace_enabled() -> bool {
    *FFX_TRACE_ENABLED
        .get_or_init(|| std::env::var_os("VICTORY_REA_FFX_TRACE").is_some_and(|v| v == "1"))
}

/// Returns the current DXR / FFX counter snapshot.
pub fn stats() -> DxrStats {
    DxrStats {
        tier: DXR_TIER.load(Ordering::Relaxed),
        blas_builds: BLAS_BUILDS.load(Ordering::Relaxed),
        tlas_builds: TLAS_BUILDS.load(Ordering::Relaxed),
        blas_bytes: BLAS_BYTES.load(Ordering::Relaxed),
        tlas_bytes: TLAS_BYTES.load(Ordering::Relaxed),
        last_build_us: LAST_BUILD_US.load(Ordering::Relaxed),
        ffx_0x218_calls: FFX_0X218_CALLS.load(Ordering::Relaxed),
    }
}

/// Queries `D3D12_FEATURE_DATA_D3D12_OPTIONS5` for `RaytracingTier`.
///
/// Returned integer codes follow D3D12 (`0` = Not Supported, `10` = Tier 1.0,
/// `11` = Tier 1.1).
pub fn probe_tier(device: &ID3D12Device) -> Option<u32> {
    let mut opts5 = D3D12_FEATURE_DATA_D3D12_OPTIONS5::default();
    // SAFETY: `device` is live and `opts5` matches `D3D12_FEATURE_D3D12_OPTIONS5`.
    let hr = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS5,
            (&raw mut opts5).cast::<c_void>(),
            std::mem::size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS5>() as u32,
        )
    };
    if hr.is_err() {
        return None;
    }
    let tier_code = if opts5.RaytracingTier == D3D12_RAYTRACING_TIER_1_1 {
        11
    } else if opts5.RaytracingTier == D3D12_RAYTRACING_TIER_1_0 {
        10
    } else {
        0
    };
    DXR_TIER.store(tier_code, Ordering::Relaxed);
    Some(tier_code)
}

/// Probes host DXR tier once when `VICTORY_REA_DXR_PROTO=1` is enabled.
pub(crate) fn maybe_probe_host() {
    if !dxr_proto_enabled() {
        return;
    }
    PROBED_ONCE.call_once(|| {
        if let Some(host) = crate::host::host() {
            let tier = probe_tier(&host.device).unwrap_or(0);
            tracing::info!(
                dxr_tier = tier,
                "DXR prototype: probed host ID3D12Device5 RaytracingTier (0=none, 10=1.0, 11=1.1)"
            );
        }
    });
}

/// Observes a title call into unimplemented console pipeline slot `0x218`
/// (`FFX/Brixelizer` research path).
pub(crate) fn on_slot_0x218_call(calls: u32, desc: *const u8) {
    FFX_0X218_CALLS.fetch_add(1, Ordering::Relaxed);
    if !ffx_trace_enabled() {
        return;
    }
    if calls <= 8 || calls.is_power_of_two() {
        let mut head = [0u64; 8];
        // SAFETY: destination is a stack array of 64 bytes; `read_into` copies safely.
        let byte_slice = unsafe {
            std::slice::from_raw_parts_mut(
                head.as_mut_ptr().cast::<u8>(),
                std::mem::size_of_val(&head),
            )
        };
        let readable = !desc.is_null() && crate::host::read_into(desc, byte_slice);
        tracing::info!(
            calls,
            readable,
            desc_words = format_args!("{head:#x?}"),
            "REA-002 FFX trace: observed device slot 0x218 pipeline call (conservative E_NOTIMPL preserved)"
        );
    }
}

fn create_buffer(
    device: &ID3D12Device,
    size: u64,
    heap_type: windows::Win32::Graphics::Direct3D12::D3D12_HEAP_TYPE,
    flags: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_FLAGS,
    initial_state: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES,
) -> Result<ID3D12Resource, String> {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: heap_type,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
        CreationNodeMask: 0,
        VisibleNodeMask: 0,
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: size.max(256),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
    };
    let mut resource: Option<ID3D12Resource> = None;
    // SAFETY: valid heap/resource descriptors and non-null output pointer.
    unsafe {
        device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                initial_state,
                None,
                &mut resource,
            )
            .map_err(|e| format!("CreateCommittedResource({size}): {e}"))?;
    }
    resource.ok_or_else(|| "CreateCommittedResource returned None".to_owned())
}

fn upload_slice<T: Copy>(resource: &ID3D12Resource, data: &[T]) -> Result<(), String> {
    let mut mapped: *mut c_void = std::ptr::null_mut();
    // SAFETY: UPLOAD buffer mapped across full range, written and unmapped before close.
    unsafe {
        resource
            .Map(0, None, Some(&mut mapped))
            .map_err(|e| format!("Map upload buffer: {e}"))?;
        std::ptr::copy_nonoverlapping(data.as_ptr(), mapped.cast::<T>(), data.len());
        resource.Unmap(0, None);
    }
    Ok(())
}

/// Runs an isolated 1-triangle BLAS + 1-instance TLAS build on `device` to
/// validate real hardware DXR 1.1 acceleration-structure support without
/// mutating title rendering state.
pub fn run_isolated_smoke(device: &ID3D12Device) -> Result<DxrStats, String> {
    let started = std::time::Instant::now();
    let tier = probe_tier(device).ok_or_else(|| "OPTIONS5 probe failed".to_owned())?;
    if tier < 10 {
        return Err(format!("device does not report DXR support (tier={tier})"));
    }
    let device5: ID3D12Device5 = device
        .cast()
        .map_err(|e| format!("cast to ID3D12Device5 failed: {e}"))?;

    // Single triangle vertices (3 * float3 = 36 bytes).
    let vertices: [[f32; 3]; 3] = [[0.0, 0.5, 0.0], [0.5, -0.5, 0.0], [-0.5, -0.5, 0.0]];
    let vb = create_buffer(
        device,
        256,
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    )?;
    upload_slice(&vb, &vertices)?;
    let vb_gpu = unsafe { vb.GetGPUVirtualAddress() };

    let geom = D3D12_RAYTRACING_GEOMETRY_DESC {
        Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
        Flags: D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
        Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
            Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                Transform3x4: 0,
                IndexFormat: DXGI_FORMAT_UNKNOWN,
                VertexFormat: DXGI_FORMAT_R32G32B32_FLOAT,
                IndexCount: 0,
                VertexCount: 3,
                IndexBuffer: 0,
                VertexBuffer: windows::Win32::Graphics::Direct3D12::D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                    StartAddress: vb_gpu,
                    StrideInBytes: 12,
                },
            },
        },
    };
    let mut blas_inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
        Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
        Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
        NumDescs: 1,
        DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
        ..Default::default()
    };
    blas_inputs.Anonymous.pGeometryDescs = &geom;

    let mut blas_info = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
    // SAFETY: valid ID3D12Device5 and inputs struct.
    unsafe {
        device5.GetRaytracingAccelerationStructurePrebuildInfo(&blas_inputs, &mut blas_info);
    }
    let blas_res_bytes = blas_info.ResultDataMaxSizeInBytes.max(256);
    let blas_scratch_bytes = blas_info.ScratchDataSizeInBytes.max(256);

    let blas = create_buffer(
        device,
        blas_res_bytes,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
    )?;
    let blas_scratch = create_buffer(
        device,
        blas_scratch_bytes,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COMMON,
    )?;
    let blas_gpu = unsafe { blas.GetGPUVirtualAddress() };

    let mut inst = D3D12_RAYTRACING_INSTANCE_DESC {
        Transform: [
            1.0, 0.0, 0.0, 0.0, // row 0
            0.0, 1.0, 0.0, 0.0, // row 1
            0.0, 0.0, 1.0, 0.0, // row 2
        ],
        AccelerationStructure: blas_gpu,
        ..Default::default()
    };
    // InstanceMask = 0xFF in bits 24..31 of `_bitfield1`.
    inst._bitfield1 = 0xFFu32 << 24;
    let inst_buf = create_buffer(
        device,
        256,
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    )?;
    upload_slice(&inst_buf, &[inst])?;
    let inst_gpu = unsafe { inst_buf.GetGPUVirtualAddress() };

    let mut tlas_inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
        Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
        Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
        NumDescs: 1,
        DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
        ..Default::default()
    };
    tlas_inputs.Anonymous.InstanceDescs = inst_gpu;

    let mut tlas_info = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
    // SAFETY: valid ID3D12Device5 and inputs struct.
    unsafe {
        device5.GetRaytracingAccelerationStructurePrebuildInfo(&tlas_inputs, &mut tlas_info);
    }
    let tlas_res_bytes = tlas_info.ResultDataMaxSizeInBytes.max(256);
    let tlas_scratch_bytes = tlas_info.ScratchDataSizeInBytes.max(256);

    let tlas = create_buffer(
        device,
        tlas_res_bytes,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
    )?;
    let tlas_scratch = create_buffer(
        device,
        tlas_scratch_bytes,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COMMON,
    )?;

    // Build on a self-contained direct queue so no title queues/lists are touched.
    // SAFETY: valid device calls creating queue, allocator, list4, fence, and event.
    let (queue, _alloc, list4, fence) = unsafe {
        let queue: ID3D12CommandQueue = device
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
                ..Default::default()
            })
            .map_err(|e| format!("CreateCommandQueue: {e}"))?;
        let alloc: ID3D12CommandAllocator = device
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            .map_err(|e| format!("CreateCommandAllocator: {e}"))?;
        let list4: ID3D12GraphicsCommandList4 = device
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
            .map_err(|e| format!("CreateCommandList: {e}"))?;
        let fence: ID3D12Fence = device
            .CreateFence(0, D3D12_FENCE_FLAG_NONE)
            .map_err(|e| format!("CreateFence: {e}"))?;
        (queue, alloc, list4, fence)
    };

    let blas_desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
        DestAccelerationStructureData: blas_gpu,
        Inputs: blas_inputs,
        SourceAccelerationStructureData: 0,
        ScratchAccelerationStructureData: unsafe { blas_scratch.GetGPUVirtualAddress() },
    };
    let tlas_desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
        DestAccelerationStructureData: unsafe { tlas.GetGPUVirtualAddress() },
        Inputs: tlas_inputs,
        SourceAccelerationStructureData: 0,
        ScratchAccelerationStructureData: unsafe { tlas_scratch.GetGPUVirtualAddress() },
    };
    let uav_barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    };

    // SAFETY: buffers stay alive past Signal + fence wait below.
    unsafe {
        list4.BuildRaytracingAccelerationStructure(&blas_desc, None);
        list4.ResourceBarrier(&[uav_barrier]);
        list4.BuildRaytracingAccelerationStructure(&tlas_desc, None);
        list4.Close().map_err(|e| format!("Close list4: {e}"))?;
        let cmd_list: windows::Win32::Graphics::Direct3D12::ID3D12CommandList =
            list4.cast().map_err(|e| format!("cast CommandList: {e}"))?;
        queue.ExecuteCommandLists(&[Some(cmd_list)]);
        queue.Signal(&fence, 1).map_err(|e| format!("Signal: {e}"))?;
        if fence.GetCompletedValue() < 1 {
            let event = CreateEventW(None, false, false, None)
                .map_err(|e| format!("CreateEventW: {e}"))?;
            fence
                .SetEventOnCompletion(1, event)
                .map_err(|e| format!("SetEventOnCompletion: {e}"))?;
            WaitForSingleObject(event, INFINITE);
            let _ = CloseHandle(event);
        }
    }

    let elapsed_us = started.elapsed().as_micros().max(1) as u64;
    BLAS_BUILDS.fetch_add(1, Ordering::Relaxed);
    TLAS_BUILDS.fetch_add(1, Ordering::Relaxed);
    BLAS_BYTES.fetch_add(blas_res_bytes, Ordering::Relaxed);
    TLAS_BYTES.fetch_add(tlas_res_bytes, Ordering::Relaxed);
    LAST_BUILD_US.store(elapsed_us, Ordering::Relaxed);
    Ok(stats())
}

#[cfg(test)]
mod tests {
    #[test]
    fn dxr_flags_default_off_and_stats_start_coherent() {
        assert!(!super::dxr_proto_enabled());
        assert!(!super::ffx_trace_enabled());
        let s = super::stats();
        assert_eq!(s.ffx_0x218_calls, s.ffx_0x218_calls);
    }

    #[test]
    fn dxr_isolated_blas_tlas_build_on_host_gpu() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU
        };
        let tier = super::probe_tier(&host.device).unwrap_or(0);
        if tier < 10 {
            return;
        }
        let s = super::run_isolated_smoke(&host.device).expect("DXR BLAS/TLAS smoke build");
        assert!(s.tier >= 10);
        assert!(s.blas_builds >= 1);
        assert!(s.tlas_builds >= 1);
        assert!(s.blas_bytes >= 256);
        assert!(s.tlas_bytes >= 256);
        assert!(s.last_build_us > 0);
    }

    #[test]
    fn slot_0x218_increments_ffx_counter() {
        let before = super::stats().ffx_0x218_calls;
        super::on_slot_0x218_call(1, std::ptr::null());
        let after = super::stats().ffx_0x218_calls;
        assert!(after > before);
    }
}
