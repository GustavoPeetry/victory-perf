//! Clean-room native DXR 1.1+ acceleration structure backend (`VICTORY_REA003_DXR=1`).
//!
//! Strictly isolated from console `BvhNodes::Memory` addresses: all BLAS/TLAS handles,
//! scratch buffers, postbuild compaction readbacks, and command execution live inside host
//! D3D12 resources managed by [`DxrBackend`].

use std::sync::OnceLock;
use std::time::Instant;

use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::Graphics::Direct3D12::{
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC,
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS, D3D12_COMMAND_LIST_TYPE_DIRECT,
    D3D12_COMMAND_QUEUE_DESC, D3D12_COMMAND_QUEUE_FLAG_NONE, D3D12_ELEMENTS_LAYOUT_ARRAY,
    D3D12_FENCE_FLAG_NONE, D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE, D3D12_HEAP_TYPE_DEFAULT,
    D3D12_HEAP_TYPE_READBACK, D3D12_HEAP_TYPE_UPLOAD,
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION,
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE,
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_NONE,
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE,
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS,
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT,
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE,
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE_DESC,
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC,
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO,
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL, D3D12_RAYTRACING_GEOMETRY_DESC,
    D3D12_RAYTRACING_GEOMETRY_DESC_0, D3D12_RAYTRACING_GEOMETRY_FLAG_NONE,
    D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE, D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC,
    D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES, D3D12_RAYTRACING_INSTANCE_DESC,
    D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_BARRIER_0, D3D12_RESOURCE_BARRIER_FLAG_NONE,
    D3D12_RESOURCE_BARRIER_TYPE_UAV, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
    D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST,
    D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_GENERIC_READ,
    D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    D3D12_RESOURCE_UAV_BARRIER, ID3D12CommandAllocator, ID3D12CommandList, ID3D12CommandQueue,
    ID3D12Device, ID3D12Device5, ID3D12Fence, ID3D12GraphicsCommandList4, ID3D12Resource,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R32G32B32_FLOAT, DXGI_FORMAT_UNKNOWN,
};
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::Interface;

use crate::dxr_proto::{create_buffer, probe_tier, upload_slice};

/// Minimum alignment required for D3D12 raytracing acceleration structure buffers.
pub const AS_ALIGNMENT_BYTES: u64 = 256;

/// Gate flag for REA-003 Frente D native DXR acceleration structures and RayQuery paths.
pub fn dxr_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("VICTORY_REA003_DXR")
            .ok()
            .is_some_and(|v| v.trim() == "1")
    })
}

/// Runtime metrics exposed by [`DxrBackend`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DxrBackendStats {
    pub tier: u32,
    pub blas_builds: u64,
    pub blas_updates: u64,
    pub blas_compactions: u64,
    pub tlas_builds: u64,
    pub tlas_updates: u64,
    pub active_blas_bytes: u64,
    pub compacted_saved_bytes: u64,
    pub active_tlas_bytes: u64,
    pub scratch_capacity_bytes: u64,
    pub last_op_us: u64,
}

/// Geometry description for building or updating a triangle BLAS.
#[derive(Clone, Copy, Debug)]
pub struct DxrTriangleMeshDesc {
    pub vertex_buffer_gpu: u64,
    pub vertex_count: u32,
    pub vertex_stride_bytes: u64,
    pub index_buffer_gpu: Option<(u64, u32, DXGI_FORMAT)>,
    pub opaque: bool,
    pub allow_update: bool,
    pub allow_compaction: bool,
    pub prefer_fast_trace: bool,
}

/// Instance entry uploaded into a TLAS build/update.
#[derive(Clone, Copy, Debug)]
pub struct DxrInstance {
    pub blas_gpu_va: u64,
    pub transform_3x4: [f32; 12],
    pub instance_id: u32,
    pub mask: u8,
    pub hit_group_offset: u32,
    pub flags: u32,
}

impl DxrInstance {
    pub fn identity(blas_gpu_va: u64, instance_id: u32) -> Self {
        Self {
            blas_gpu_va,
            transform_3x4: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            instance_id,
            mask: 0xFF,
            hit_group_offset: 0,
            flags: 0,
        }
    }

    fn to_d3d12(self) -> D3D12_RAYTRACING_INSTANCE_DESC {
        let mut d = D3D12_RAYTRACING_INSTANCE_DESC {
            Transform: self.transform_3x4,
            AccelerationStructure: self.blas_gpu_va,
            ..Default::default()
        };
        // Low 24 bits: InstanceID; bits 24..31: InstanceMask.
        d._bitfield1 = (self.instance_id & 0x00FF_FFFF) | (u32::from(self.mask) << 24);
        // Low 24 bits: InstanceContributionToHitGroupIndex; bits 24..31: Flags.
        d._bitfield2 = (self.hit_group_offset & 0x00FF_FFFF) | ((self.flags & 0xFF) << 24);
        d
    }
}

/// Owned Bottom-Level Acceleration Structure on host VRAM.
pub struct DxrBlas {
    pub buffer: ID3D12Resource,
    pub gpu_va: u64,
    pub bytes: u64,
    pub uncompacted_bytes: u64,
    pub compacted: bool,
    allow_update: bool,
    allow_compaction: bool,
    build_flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS,
}

/// Owned Top-Level Acceleration Structure on host VRAM.
pub struct DxrTlas {
    pub buffer: ID3D12Resource,
    pub gpu_va: u64,
    pub bytes: u64,
    pub instance_count: u32,
    instances_upload: ID3D12Resource,
    instances_capacity: u32,
    allow_update: bool,
    build_flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS,
}

/// Grow-on-demand scratch arena reused across BLAS/TLAS builds and updates.
struct DxrScratchArena {
    buffer: Option<ID3D12Resource>,
    gpu_va: u64,
    capacity: u64,
}

impl DxrScratchArena {
    fn new() -> Self {
        Self {
            buffer: None,
            gpu_va: 0,
            capacity: 0,
        }
    }

    fn ensure(&mut self, device: &ID3D12Device, needed: u64) -> Result<u64, String> {
        let aligned = align_up(needed.max(AS_ALIGNMENT_BYTES), AS_ALIGNMENT_BYTES);
        if aligned <= self.capacity && self.buffer.is_some() {
            return Ok(self.gpu_va);
        }
        let next = aligned.max(self.capacity.saturating_mul(2)).max(64 * 1024);
        let buf = create_buffer(
            device,
            next,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        )?;
        // SAFETY: `buf` was just created on a valid D3D12 buffer resource.
        let gpu_va = unsafe { buf.GetGPUVirtualAddress() };
        self.buffer = Some(buf);
        self.gpu_va = gpu_va;
        self.capacity = next;
        Ok(gpu_va)
    }
}

/// Self-contained DXR 1.1+ backend for building, refitting, compacting, and executing traces.
pub struct DxrBackend {
    pub device: ID3D12Device,
    pub device5: ID3D12Device5,
    pub queue: ID3D12CommandQueue,
    pub alloc: ID3D12CommandAllocator,
    pub list4: ID3D12GraphicsCommandList4,
    fence: ID3D12Fence,
    fence_value: u64,
    event: HANDLE,
    scratch: DxrScratchArena,
    stats: DxrBackendStats,
}

impl DxrBackend {
    /// Initializes an isolated DXR direct queue, command list, and scratch arena.
    pub fn new(device: &ID3D12Device) -> Result<Self, String> {
        let tier = probe_tier(device).ok_or_else(|| "probe DXR tier failed".to_owned())?;
        if tier < 10 {
            return Err(format!("adapter lacks DXR support (tier={tier})"));
        }
        let device5: ID3D12Device5 = device
            .cast()
            .map_err(|e| format!("cast ID3D12Device5: {e}"))?;
        // SAFETY: all COM objects come from the validated host `device` and stay owned by `Self`.
        let (queue, alloc, list4, fence, event) = unsafe {
            let queue: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
                    ..Default::default()
                })
                .map_err(|e| format!("CreateCommandQueue(DXR): {e}"))?;
            let alloc: ID3D12CommandAllocator = device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .map_err(|e| format!("CreateCommandAllocator(DXR): {e}"))?;
            let list4: ID3D12GraphicsCommandList4 = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
                .map_err(|e| format!("CreateCommandList(DXR): {e}"))?;
            list4.Close().map_err(|e| format!("initial Close: {e}"))?;
            let fence: ID3D12Fence = device
                .CreateFence(0, D3D12_FENCE_FLAG_NONE)
                .map_err(|e| format!("CreateFence(DXR): {e}"))?;
            let event = CreateEventW(None, false, false, None)
                .map_err(|e| format!("CreateEventW(DXR): {e}"))?;
            (queue, alloc, list4, fence, event)
        };
        Ok(Self {
            device: device.clone(),
            device5,
            queue,
            alloc,
            list4,
            fence,
            fence_value: 0,
            event,
            scratch: DxrScratchArena::new(),
            stats: DxrBackendStats {
                tier,
                ..Default::default()
            },
        })
    }

    pub fn tier(&self) -> u32 {
        self.stats.tier
    }

    pub fn stats(&self) -> DxrBackendStats {
        let mut s = self.stats;
        s.scratch_capacity_bytes = self.scratch.capacity;
        s
    }

    /// Uploads triangle vertices (and optional `u32` indices) into host UPLOAD buffers.
    pub fn create_triangle_buffers(
        &self,
        vertices: &[[f32; 3]],
        indices: Option<&[u32]>,
    ) -> Result<(ID3D12Resource, Option<ID3D12Resource>), String> {
        let vb_bytes = (std::mem::size_of_val(vertices) as u64).max(256);
        let vb = create_buffer(
            &self.device,
            vb_bytes,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        if !vertices.is_empty() {
            upload_slice(&vb, vertices)?;
        }
        let ib = if let Some(idx) = indices {
            let ib_bytes = (std::mem::size_of_val(idx) as u64).max(256);
            let buf = create_buffer(
                &self.device,
                ib_bytes,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            if !idx.is_empty() {
                upload_slice(&buf, idx)?;
            }
            Some(buf)
        } else {
            None
        };
        Ok((vb, ib))
    }

    /// Builds a Bottom-Level Acceleration Structure from triangle buffers.
    pub fn build_blas(&mut self, mesh: &DxrTriangleMeshDesc) -> Result<DxrBlas, String> {
        let geom = geometry_desc(mesh);
        let build_flags = blas_build_flags(mesh);
        let mut inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
            Flags: build_flags,
            NumDescs: 1,
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
            ..Default::default()
        };
        inputs.Anonymous.pGeometryDescs = &geom;

        let info = self.prebuild_info(&inputs);
        let res_bytes = align_up(
            info.ResultDataMaxSizeInBytes.max(AS_ALIGNMENT_BYTES),
            AS_ALIGNMENT_BYTES,
        );
        let scratch_gpu = self
            .scratch
            .ensure(&self.device, info.ScratchDataSizeInBytes)?;

        let buffer = create_buffer(
            &self.device,
            res_bytes,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
        )?;
        // SAFETY: `buffer` is a valid D3D12 buffer resource.
        let gpu_va = unsafe { buffer.GetGPUVirtualAddress() };

        let desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
            DestAccelerationStructureData: gpu_va,
            Inputs: inputs,
            SourceAccelerationStructureData: 0,
            ScratchAccelerationStructureData: scratch_gpu,
        };
        let us = self.submit_and_wait(|list4| {
            // SAFETY: descriptors and scratch/destination buffers remain alive until fence wait finishes.
            unsafe {
                list4.BuildRaytracingAccelerationStructure(&desc, None);
                list4.ResourceBarrier(&[uav_barrier_all()]);
            }
            Ok(())
        })?;

        self.stats.blas_builds += 1;
        self.stats.active_blas_bytes = self.stats.active_blas_bytes.saturating_add(res_bytes);
        self.stats.last_op_us = us;

        Ok(DxrBlas {
            buffer,
            gpu_va,
            bytes: res_bytes,
            uncompacted_bytes: res_bytes,
            compacted: false,
            allow_update: mesh.allow_update,
            allow_compaction: mesh.allow_compaction,
            build_flags,
        })
    }

    /// Performs an in-place DXR refit (`PERFORM_UPDATE`) on an updatable uncompacted BLAS.
    pub fn update_blas(
        &mut self,
        blas: &mut DxrBlas,
        mesh: &DxrTriangleMeshDesc,
    ) -> Result<(), String> {
        if !blas.allow_update {
            return Err("BLAS was not built with allow_update".to_owned());
        }
        if blas.compacted {
            return Err("compacted BLAS cannot be updated in-place".to_owned());
        }
        let geom = geometry_desc(mesh);
        let update_flags =
            blas.build_flags | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
        let mut inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
            Flags: update_flags,
            NumDescs: 1,
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
            ..Default::default()
        };
        inputs.Anonymous.pGeometryDescs = &geom;

        let info = self.prebuild_info(&inputs);
        let scratch_gpu = self.scratch.ensure(
            &self.device,
            info.UpdateScratchDataSizeInBytes
                .max(info.ScratchDataSizeInBytes),
        )?;

        let desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
            DestAccelerationStructureData: blas.gpu_va,
            Inputs: inputs,
            SourceAccelerationStructureData: blas.gpu_va,
            ScratchAccelerationStructureData: scratch_gpu,
        };
        let us = self.submit_and_wait(|list4| {
            // SAFETY: in-place BLAS update with valid scratch and fence synchronization.
            unsafe {
                list4.BuildRaytracingAccelerationStructure(&desc, None);
                list4.ResourceBarrier(&[uav_barrier_all()]);
            }
            Ok(())
        })?;

        self.stats.blas_updates += 1;
        self.stats.last_op_us = us;
        Ok(())
    }

    /// Queries compacted size and compacts the BLAS into a tight DEFAULT buffer when smaller.
    pub fn compact_blas(&mut self, blas: &mut DxrBlas) -> Result<bool, String> {
        if !blas.allow_compaction || blas.compacted {
            return Ok(false);
        }
        let info_bytes = align_up(
            std::mem::size_of::<
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE_DESC,
            >() as u64,
            AS_ALIGNMENT_BYTES,
        );
        let postbuild_uav = create_buffer(
            &self.device,
            info_bytes,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )?;
        let postbuild_rb = create_buffer(
            &self.device,
            info_bytes,
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;
        // SAFETY: `postbuild_uav` is a valid buffer resource.
        let postbuild_gpu = unsafe { postbuild_uav.GetGPUVirtualAddress() };
        let postbuild_desc = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC {
            DestBuffer: postbuild_gpu,
            InfoType: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE,
        };
        let blas_va = blas.gpu_va;
        let _us_query = self.submit_and_wait(|list4| {
            // SAFETY: valid postbuild emit + transition + CopyResource into readback buffer.
            unsafe {
                list4.EmitRaytracingAccelerationStructurePostbuildInfo(&postbuild_desc, &[blas_va]);
                list4.ResourceBarrier(&[
                    uav_barrier_all(),
                    transition_barrier(
                        &postbuild_uav,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    ),
                ]);
                list4.CopyResource(&postbuild_rb, &postbuild_uav);
            }
            Ok(())
        })?;

        let compacted_raw = readback_pod::<
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE_DESC,
        >(&postbuild_rb)?
        .CompactedSizeInBytes;
        let compacted_bytes = align_up(compacted_raw.max(AS_ALIGNMENT_BYTES), AS_ALIGNMENT_BYTES);
        if compacted_bytes == 0 || compacted_bytes >= blas.bytes {
            return Ok(false);
        }

        let compact_buf = create_buffer(
            &self.device,
            compacted_bytes,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
        )?;
        // SAFETY: `compact_buf` is a valid D3D12 buffer resource.
        let compact_va = unsafe { compact_buf.GetGPUVirtualAddress() };
        let us_copy = self.submit_and_wait(|list4| {
            // SAFETY: source and destination AS buffers remain live until the fence completes.
            unsafe {
                list4.CopyRaytracingAccelerationStructure(
                    compact_va,
                    blas_va,
                    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT,
                );
                list4.ResourceBarrier(&[uav_barrier_all()]);
            }
            Ok(())
        })?;

        let saved = blas.bytes - compacted_bytes;
        self.stats.active_blas_bytes = self.stats.active_blas_bytes.saturating_sub(saved);
        self.stats.compacted_saved_bytes = self.stats.compacted_saved_bytes.saturating_add(saved);
        self.stats.blas_compactions += 1;
        self.stats.last_op_us = us_copy;

        blas.buffer = compact_buf;
        blas.gpu_va = compact_va;
        blas.bytes = compacted_bytes;
        blas.compacted = true;
        Ok(true)
    }

    /// Builds a Top-Level Acceleration Structure referencing `instances`.
    pub fn build_tlas(
        &mut self,
        instances: &[DxrInstance],
        allow_update: bool,
    ) -> Result<DxrTlas, String> {
        let count = u32::try_from(instances.len()).map_err(|_| "too many instances")?;
        let cap = count.max(1);
        let inst_bytes = align_up(
            u64::from(cap) * std::mem::size_of::<D3D12_RAYTRACING_INSTANCE_DESC>() as u64,
            AS_ALIGNMENT_BYTES,
        );
        let instances_upload = create_buffer(
            &self.device,
            inst_bytes,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        if !instances.is_empty() {
            let d3d_insts: Vec<_> = instances.iter().map(|i| i.to_d3d12()).collect();
            upload_slice(&instances_upload, &d3d_insts)?;
        }
        // SAFETY: `instances_upload` is a valid buffer resource.
        let inst_gpu = unsafe { instances_upload.GetGPUVirtualAddress() };
        let build_flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE
            | if allow_update {
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE
            } else {
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_NONE
            };
        let mut inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
            Flags: build_flags,
            NumDescs: count,
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
            ..Default::default()
        };
        inputs.Anonymous.InstanceDescs = if count == 0 { 0 } else { inst_gpu };

        let info = self.prebuild_info(&inputs);
        let res_bytes = align_up(
            info.ResultDataMaxSizeInBytes.max(AS_ALIGNMENT_BYTES),
            AS_ALIGNMENT_BYTES,
        );
        let scratch_gpu = self
            .scratch
            .ensure(&self.device, info.ScratchDataSizeInBytes)?;

        let buffer = create_buffer(
            &self.device,
            res_bytes,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
        )?;
        // SAFETY: `buffer` is a valid buffer resource.
        let gpu_va = unsafe { buffer.GetGPUVirtualAddress() };
        let desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
            DestAccelerationStructureData: gpu_va,
            Inputs: inputs,
            SourceAccelerationStructureData: 0,
            ScratchAccelerationStructureData: scratch_gpu,
        };

        let us = self.submit_and_wait(|list4| {
            // SAFETY: valid TLAS descriptors and scratch/result allocations.
            unsafe {
                list4.BuildRaytracingAccelerationStructure(&desc, None);
                list4.ResourceBarrier(&[uav_barrier_all()]);
            }
            Ok(())
        })?;

        self.stats.tlas_builds += 1;
        self.stats.active_tlas_bytes = self.stats.active_tlas_bytes.saturating_add(res_bytes);
        self.stats.last_op_us = us;

        Ok(DxrTlas {
            buffer,
            gpu_va,
            bytes: res_bytes,
            instance_count: count,
            instances_upload,
            instances_capacity: cap,
            allow_update,
            build_flags,
        })
    }

    /// Updates an existing TLAS in-place with updated transforms/masks/handles.
    pub fn update_tlas(
        &mut self,
        tlas: &mut DxrTlas,
        instances: &[DxrInstance],
    ) -> Result<(), String> {
        if !tlas.allow_update {
            return Err("TLAS was not built with allow_update".to_owned());
        }
        let count = u32::try_from(instances.len()).map_err(|_| "too many instances")?;
        if count > tlas.instances_capacity {
            return Err(format!(
                "TLAS update instance count ({count}) exceeds build capacity ({})",
                tlas.instances_capacity
            ));
        }
        if !instances.is_empty() {
            let d3d_insts: Vec<_> = instances.iter().map(|i| i.to_d3d12()).collect();
            upload_slice(&tlas.instances_upload, &d3d_insts)?;
        }
        // SAFETY: `instances_upload` is valid.
        let inst_gpu = unsafe { tlas.instances_upload.GetGPUVirtualAddress() };
        let update_flags =
            tlas.build_flags | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
        let mut inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
            Flags: update_flags,
            NumDescs: count,
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
            ..Default::default()
        };
        inputs.Anonymous.InstanceDescs = if count == 0 { 0 } else { inst_gpu };

        let info = self.prebuild_info(&inputs);
        let scratch_gpu = self.scratch.ensure(
            &self.device,
            info.UpdateScratchDataSizeInBytes
                .max(info.ScratchDataSizeInBytes),
        )?;

        let desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
            DestAccelerationStructureData: tlas.gpu_va,
            Inputs: inputs,
            SourceAccelerationStructureData: tlas.gpu_va,
            ScratchAccelerationStructureData: scratch_gpu,
        };
        let us = self.submit_and_wait(|list4| {
            // SAFETY: in-place TLAS update with valid scratch and fence synchronization.
            unsafe {
                list4.BuildRaytracingAccelerationStructure(&desc, None);
                list4.ResourceBarrier(&[uav_barrier_all()]);
            }
            Ok(())
        })?;

        tlas.instance_count = count;
        self.stats.tlas_updates += 1;
        self.stats.last_op_us = us;
        Ok(())
    }

    /// Executes custom commands on the backend's direct command list4 and waits on the fence.
    pub fn submit_and_wait(
        &mut self,
        record: impl FnOnce(&ID3D12GraphicsCommandList4) -> Result<(), String>,
    ) -> Result<u64, String> {
        let started = Instant::now();
        // SAFETY: command allocator and list4 were already completed or freshly initialized.
        unsafe {
            self.alloc
                .Reset()
                .map_err(|e| format!("Reset alloc: {e}"))?;
            self.list4
                .Reset(&self.alloc, None)
                .map_err(|e| format!("Reset list4: {e}"))?;
        }
        record(&self.list4)?;
        let next_fence = self.fence_value + 1;
        // SAFETY: closes list4, submits, signals fence, and blocks until completion.
        unsafe {
            self.list4
                .Close()
                .map_err(|e| format!("Close list4: {e}"))?;
            let cmd_list: ID3D12CommandList = self
                .list4
                .cast()
                .map_err(|e| format!("cast CommandList: {e}"))?;
            self.queue.ExecuteCommandLists(&[Some(cmd_list)]);
            self.queue
                .Signal(&self.fence, next_fence)
                .map_err(|e| format!("Signal fence: {e}"))?;
            if self.fence.GetCompletedValue() < next_fence {
                self.fence
                    .SetEventOnCompletion(next_fence, self.event)
                    .map_err(|e| format!("SetEventOnCompletion: {e}"))?;
                WaitForSingleObject(self.event, INFINITE);
            }
        }
        self.fence_value = next_fence;
        Ok(started.elapsed().as_micros().max(1) as u64)
    }

    fn prebuild_info(
        &self,
        inputs: &D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS,
    ) -> D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO {
        let mut info = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
        // SAFETY: valid `ID3D12Device5` and properly initialized inputs descriptor.
        unsafe {
            self.device5
                .GetRaytracingAccelerationStructurePrebuildInfo(inputs, &mut info);
        }
        info
    }
}

impl Drop for DxrBackend {
    fn drop(&mut self) {
        if !self.event.is_invalid() {
            // SAFETY: `self.event` was created via `CreateEventW` and is closed once on Drop.
            unsafe {
                let _ = CloseHandle(self.event);
            }
        }
    }
}

pub(crate) fn uav_barrier_all() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    }
}

pub(crate) fn transition_barrier(
    resource: &ID3D12Resource,
    before: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES,
    after: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_TRANSITION_BARRIER,
    };
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: std::mem::ManuallyDrop::new(Some(resource.clone())),
                Subresource:
                    windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

pub(crate) fn readback_pod<T: Copy + Default>(resource: &ID3D12Resource) -> Result<T, String> {
    let mut ptr = std::ptr::null_mut();
    let mut out = T::default();
    // SAFETY: maps readback resource created for at least `size_of::<T>()` bytes and copies once.
    unsafe {
        resource
            .Map(0, None, Some(&mut ptr))
            .map_err(|e| format!("Map(READBACK): {e}"))?;
        std::ptr::copy_nonoverlapping(ptr.cast::<T>(), &mut out, 1);
        resource.Unmap(0, None);
    }
    Ok(out)
}

pub(crate) fn readback_vec<T: Copy>(
    resource: &ID3D12Resource,
    len: usize,
) -> Result<Vec<T>, String> {
    if len == 0 {
        return Ok(Vec::new());
    }
    let mut ptr = std::ptr::null_mut();
    let mut out = Vec::with_capacity(len);
    // SAFETY: caller ensures `resource` holds at least `len * size_of::<T>()` bytes.
    unsafe {
        resource
            .Map(0, None, Some(&mut ptr))
            .map_err(|e| format!("Map(READBACK vec): {e}"))?;
        let slice = std::slice::from_raw_parts(ptr.cast::<T>(), len);
        out.extend_from_slice(slice);
        resource.Unmap(0, None);
    }
    Ok(out)
}

pub(crate) const fn align_up(v: u64, align: u64) -> u64 {
    let rem = v % align;
    if rem == 0 { v } else { v + (align - rem) }
}

fn geometry_desc(mesh: &DxrTriangleMeshDesc) -> D3D12_RAYTRACING_GEOMETRY_DESC {
    let (idx_format, idx_count, idx_buf) = match mesh.index_buffer_gpu {
        Some((addr, count, fmt)) => (fmt, count, addr),
        None => (DXGI_FORMAT_UNKNOWN, 0, 0),
    };
    D3D12_RAYTRACING_GEOMETRY_DESC {
        Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
        Flags: if mesh.opaque {
            D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE
        } else {
            D3D12_RAYTRACING_GEOMETRY_FLAG_NONE
        },
        Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
            Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                Transform3x4: 0,
                IndexFormat: idx_format,
                VertexFormat: DXGI_FORMAT_R32G32B32_FLOAT,
                IndexCount: idx_count,
                VertexCount: mesh.vertex_count,
                IndexBuffer: idx_buf,
                VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                    StartAddress: mesh.vertex_buffer_gpu,
                    StrideInBytes: mesh.vertex_stride_bytes,
                },
            },
        },
    }
}

fn blas_build_flags(
    mesh: &DxrTriangleMeshDesc,
) -> D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS {
    let mut f = if mesh.prefer_fast_trace {
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE
    } else {
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_NONE
    };
    if mesh.allow_update {
        f |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
    }
    if mesh.allow_compaction {
        f |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION;
    }
    f
}

/// Convenience helper for index buffers typed as `DXGI_FORMAT_R32_UINT`.
pub const INDEX_FORMAT_U32: DXGI_FORMAT = DXGI_FORMAT_R32_UINT;
