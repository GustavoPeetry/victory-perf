//! Real-GPU D3D12 compute execution & OOB/null-UAV verification harness (`victory-gpu`).
//!
//! Exercises pipeline compilation, multi-group (`4x1x1` = 256 threads) compute
//! dispatch, readback parity, D3D12 null raw UAV bounds safety (`memory.rs`
//! 1024-slot descriptor table contract), and GPU breadcrumb recording on RTX 2060 SUPER.

use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use windows::Win32::Foundation::CloseHandle;
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::{
    D3D_ROOT_SIGNATURE_VERSION_1_0, D3D12_BUFFER_UAV, D3D12_BUFFER_UAV_FLAG_RAW,
    D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_DESC, D3D12_COMMAND_QUEUE_FLAG_NONE,
    D3D12_COMPUTE_PIPELINE_STATE_DESC, D3D12_DESCRIPTOR_HEAP_DESC,
    D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
    D3D12_DESCRIPTOR_RANGE, D3D12_DESCRIPTOR_RANGE_TYPE_UAV, D3D12_FENCE_FLAG_NONE,
    D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_TYPE_READBACK, D3D12_HEAP_TYPE_UPLOAD,
    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_NONE,
    D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE,
    D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    D3D12_ROOT_DESCRIPTOR, D3D12_ROOT_DESCRIPTOR_TABLE, D3D12_ROOT_PARAMETER,
    D3D12_ROOT_PARAMETER_0, D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
    D3D12_ROOT_PARAMETER_TYPE_SRV, D3D12_ROOT_PARAMETER_TYPE_UAV, D3D12_ROOT_SIGNATURE_DESC,
    D3D12_ROOT_SIGNATURE_FLAG_NONE, D3D12_SHADER_BYTECODE, D3D12_SHADER_VISIBILITY_ALL,
    D3D12_UAV_DIMENSION_BUFFER, D3D12_UNORDERED_ACCESS_VIEW_DESC,
    D3D12_UNORDERED_ACCESS_VIEW_DESC_0, D3D12SerializeRootSignature, ID3D12CommandAllocator,
    ID3D12CommandList, ID3D12CommandQueue, ID3D12DescriptorHeap, ID3D12Device, ID3D12Fence,
    ID3D12GraphicsCommandList, ID3D12PipelineState, ID3D12RootSignature,
};
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_TYPELESS;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::{Interface, s};

use crate::breadcrumbs::{GpuOpBreadcrumb, OpKind, QueueKind};
use crate::dxr_backend::{readback_vec, transition_barrier};
use crate::dxr_proto::{create_buffer, upload_slice};

static HARNESS_SUBMISSION_ID: AtomicU64 = AtomicU64::new(10_000);

const PARITY_HLSL: &str = r#"
ByteAddressBuffer g_in : register(t0);
RWByteAddressBuffer g_out : register(u0);

[numthreads(64, 1, 1)]
void cs_main(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    uint val = g_in.Load(idx * 4u);
    uint mixed = (val ^ (idx * 0x9E3779B1u)) + (idx & 63u);
    g_out.Store(idx * 4u, mixed);
}
"#;

const NULL_UAV_OOB_HLSL: &str = r#"
RWByteAddressBuffer g_out : register(u0);
RWByteAddressBuffer g_null : register(u1);

[numthreads(64, 1, 1)]
void cs_main(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    uint oob_read = g_null.Load(idx * 4u + 4096u);
    g_null.Store(idx * 4u + 4096u, 0xDEADBEEFu);
    g_out.Store(idx * 8u, oob_read);
    g_out.Store(idx * 8u + 4u, 0xA5A50000u | idx);
}
"#;

fn compile_cs(source: &str) -> Result<Vec<u8>, String> {
    let mut blob: Option<ID3DBlob> = None;
    let mut err: Option<ID3DBlob> = None;
    // SAFETY: static HLSL source string & entry point for FXC `cs_5_1`.
    let hr = unsafe {
        D3DCompile(
            source.as_ptr().cast::<c_void>(),
            source.len(),
            s!("shader_exec_harness.hlsl"),
            None,
            None,
            s!("cs_main"),
            s!("cs_5_1"),
            0,
            0,
            &mut blob,
            Some(&mut err),
        )
    };
    if let Err(e) = hr {
        let msg = err
            .and_then(|b| {
                // SAFETY: FXC error blob contains UTF-8 diagnostic text.
                let s = unsafe {
                    std::slice::from_raw_parts(b.GetBufferPointer().cast::<u8>(), b.GetBufferSize())
                };
                std::str::from_utf8(s).ok().map(str::to_owned)
            })
            .unwrap_or_default();
        return Err(format!("D3DCompile cs_5_1 failed ({e}): {msg}"));
    }
    let blob = blob.ok_or("missing FXC blob")?;
    // SAFETY: FXC output blob memory stays valid until `blob` drops.
    let bytes = unsafe {
        std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
    };
    Ok(bytes.to_vec())
}

fn make_compute_pso(
    device: &ID3D12Device,
    root: &ID3D12RootSignature,
    bytecode: &[u8],
) -> Result<ID3D12PipelineState, String> {
    let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        // SAFETY: borrowed COM pointer consumed synchronously during PSO creation.
        pRootSignature: unsafe { std::mem::transmute_copy(root) },
        CS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytecode.as_ptr().cast::<c_void>(),
            BytecodeLength: bytecode.len(),
        },
        ..Default::default()
    };
    // SAFETY: valid device & compute PSO desc.
    unsafe { device.CreateComputePipelineState(&desc) }.map_err(|e| format!("ComputePSO: {e}"))
}

fn run_queue_list(
    device: &ID3D12Device,
    pso: &ID3D12PipelineState,
    record: impl FnOnce(&ID3D12GraphicsCommandList, usize) -> Result<(), String>,
) -> Result<u64, String> {
    let started = Instant::now();
    // SAFETY: self-contained D3D12 command queue, allocator, list, fence, and event lifecycle.
    unsafe {
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
        let list: ID3D12GraphicsCommandList = device
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, pso)
            .map_err(|e| format!("CreateCommandList: {e}"))?;
        let list_ptr = list.as_raw() as usize;
        record(&list, list_ptr)?;
        list.Close().map_err(|e| format!("Close list: {e}"))?;

        let fence: ID3D12Fence = device
            .CreateFence(0, D3D12_FENCE_FLAG_NONE)
            .map_err(|e| format!("CreateFence: {e}"))?;
        let event =
            CreateEventW(None, false, false, None).map_err(|e| format!("CreateEventW: {e}"))?;

        let cmd: [Option<ID3D12CommandList>; 1] =
            [Some(list.cast().map_err(|e| format!("cast: {e}"))?)];
        let sub_id = HARNESS_SUBMISSION_ID.fetch_add(1, Ordering::Relaxed);
        crate::breadcrumbs::record_submit(QueueKind::Direct, sub_id, 1, 1, 0x5EA0_0310);
        crate::breadcrumbs::mark_submission_fence(&[list_ptr], sub_id, 1);

        queue.ExecuteCommandLists(&cmd);
        let sig = queue
            .Signal(&fence, 1)
            .and_then(|()| fence.SetEventOnCompletion(1, event));
        if let Err(e) = sig {
            let _ = CloseHandle(event);
            return Err(format!("Signal/SetEventOnCompletion: {e}"));
        }
        let wait_rc = WaitForSingleObject(event, 5000);
        let _ = CloseHandle(event);
        if wait_rc.0 != 0 {
            let reason = device.GetDeviceRemovedReason();
            return Err(format!("WaitForSingleObject={}: {reason:?}", wait_rc.0));
        }
        if let Err(reason) = device.GetDeviceRemovedReason() {
            return Err(format!("GetDeviceRemovedReason failed: {reason}"));
        }
    }
    Ok(started.elapsed().as_micros() as u64)
}

/// Runs a 256-thread (`4x1x1`) compute parity check on `device`, returning `(elapsed_us, output)`.
pub fn execute_256_thread_parity(
    device: &ID3D12Device,
    input: &[u32; 256],
) -> Result<(u64, Vec<u32>), String> {
    let cs = compile_cs(PARITY_HLSL)?;
    let params = [
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
    ];
    let rs_desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        NumStaticSamplers: 0,
        pStaticSamplers: std::ptr::null(),
        Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
    };
    let root: ID3D12RootSignature = unsafe {
        let mut blob = None;
        let mut err = None;
        D3D12SerializeRootSignature(
            &rs_desc,
            D3D_ROOT_SIGNATURE_VERSION_1_0,
            &mut blob,
            Some(&mut err),
        )
        .map_err(|e| format!("SerializeRS: {e}"))?;
        let blob = blob.ok_or("no rs blob")?;
        let bytes =
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
        device
            .CreateRootSignature(0, bytes)
            .map_err(|e| format!("CreateRootSignature: {e}"))?
    };
    let pso = make_compute_pso(device, &root, &cs)?;
    let byte_len = std::mem::size_of_val(input) as u64;
    let in_up = create_buffer(
        device,
        byte_len,
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    )?;
    upload_slice(&in_up, input)?;
    let in_gpu = create_buffer(
        device,
        byte_len,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    )?;
    let out_gpu = create_buffer(
        device,
        byte_len,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    )?;
    let out_rb = create_buffer(
        device,
        byte_len,
        D3D12_HEAP_TYPE_READBACK,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    )?;

    let elapsed_us = run_queue_list(device, &pso, |list, list_ptr| {
        // SAFETY: valid committed buffers & root signature bound on open compute list.
        unsafe {
            list.CopyBufferRegion(&in_gpu, 0, &in_up, 0, byte_len);
            list.ResourceBarrier(&[transition_barrier(
                &in_gpu,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )]);
            list.SetComputeRootSignature(&root);
            list.SetComputeRootShaderResourceView(0, in_gpu.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(1, out_gpu.GetGPUVirtualAddress());
            crate::breadcrumbs::record_op(GpuOpBreadcrumb {
                seq: 0,
                frame: 1,
                queue: QueueKind::Direct,
                cmd_list_ptr: list_ptr,
                submission_id: 0,
                fence_value: u64::MAX,
                op_kind: OpKind::Dispatch,
                pipeline_hash: 0x5EA0_0311,
                shader_hashes: [0x5EA0_0311_0000_0001, 0, 0],
                dispatch_dims: [4, 1, 1],
                root_sig_hash: 0x101,
                bound_resources_count: 2,
                resource_state_mask: 0,
            });
            list.Dispatch(4, 1, 1);
            list.ResourceBarrier(&[transition_barrier(
                &out_gpu,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            list.CopyBufferRegion(&out_rb, 0, &out_gpu, 0, byte_len);
        }
        Ok(())
    })?;

    let out = readback_vec::<u32>(&out_rb, 256)?;
    Ok((elapsed_us, out))
}

/// Verifies null raw UAV descriptors (`memory.rs` 1024-slot heap contract) return 0 on
/// OOB loads, drop OOB stores, and leave the D3D12 device healthy across 256 threads.
pub fn execute_null_uav_oob_probe(device: &ID3D12Device) -> Result<Vec<u32>, String> {
    let cs = compile_cs(NULL_UAV_OOB_HLSL)?;
    let range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
        NumDescriptors: 2,
        BaseShaderRegister: 0,
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: 0,
    };
    let param = D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                NumDescriptorRanges: 1,
                pDescriptorRanges: &range,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    };
    let rs_desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: 1,
        pParameters: &param,
        NumStaticSamplers: 0,
        pStaticSamplers: std::ptr::null(),
        Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
    };
    let root: ID3D12RootSignature = unsafe {
        let mut blob = None;
        let mut err = None;
        D3D12SerializeRootSignature(
            &rs_desc,
            D3D_ROOT_SIGNATURE_VERSION_1_0,
            &mut blob,
            Some(&mut err),
        )
        .map_err(|e| format!("SerializeRS oob: {e}"))?;
        let blob = blob.ok_or("no oob rs blob")?;
        let bytes =
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
        device
            .CreateRootSignature(0, bytes)
            .map_err(|e| format!("CreateRootSignature oob: {e}"))?
    };
    let pso = make_compute_pso(device, &root, &cs)?;
    let heap: ID3D12DescriptorHeap = unsafe {
        device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: 2,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            ..Default::default()
        })
    }
    .map_err(|e| format!("CreateDescriptorHeap: {e}"))?;

    let dwords = 256 * 2u32;
    let byte_len = u64::from(dwords) * 4;
    let out_gpu = create_buffer(
        device,
        byte_len,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    )?;
    let out_rb = create_buffer(
        device,
        byte_len,
        D3D12_HEAP_TYPE_READBACK,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    )?;

    unsafe {
        let inc = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
            as usize;
        let slot0 = heap.GetCPUDescriptorHandleForHeapStart();
        let mut slot1 = slot0;
        slot1.ptr += inc;

        let live_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    NumElements: dwords,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                    ..Default::default()
                },
            },
        };
        let null_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    NumElements: 1,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                    ..Default::default()
                },
            },
        };
        device.CreateUnorderedAccessView(&out_gpu, None, Some(&live_desc), slot0);
        device.CreateUnorderedAccessView(None, None, Some(&null_desc), slot1);
    }

    let _ = run_queue_list(device, &pso, |list, _ptr| {
        // SAFETY: shader-visible heap holds initialized `u0` (live) and `u1` (null raw UAV).
        unsafe {
            let gpu_start = heap.GetGPUDescriptorHandleForHeapStart();
            list.SetDescriptorHeaps(&[Some(heap.clone())]);
            list.SetComputeRootSignature(&root);
            list.SetComputeRootDescriptorTable(0, gpu_start);
            list.Dispatch(4, 1, 1);
            list.ResourceBarrier(&[transition_barrier(
                &out_gpu,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            list.CopyBufferRegion(&out_rb, 0, &out_gpu, 0, byte_len);
        }
        Ok(())
    })?;

    readback_vec::<u32>(&out_rb, dwords as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::Win32::Graphics::Direct3D12::D3D12CreateDevice;

    fn test_device() -> Option<ID3D12Device> {
        let mut dev: Option<ID3D12Device> = None;
        // SAFETY: queries default hardware D3D12 12_0 adapter for standalone GPU tests.
        unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut dev).ok()? };
        dev
    }

    #[test]
    fn gpu_harness_executes_256_threads_and_records_breadcrumbs() {
        let Some(device) = test_device() else {
            eprintln!("no D3D12 device; skipped");
            return;
        };
        let mut input = [0u32; 256];
        for (i, v) in input.iter_mut().enumerate() {
            *v = (i as u32).wrapping_mul(0x0101_0003).wrapping_add(17);
        }
        let (elapsed_us, got) =
            execute_256_thread_parity(&device, &input).expect("256-thread GPU parity");
        assert_eq!(got.len(), 256);
        for (idx, (&inp, &out)) in input.iter().zip(got.iter()).enumerate() {
            let i = idx as u32;
            let expected = (inp ^ i.wrapping_mul(0x9E37_79B1)).wrapping_add(i & 63);
            assert_eq!(out, expected, "parity mismatch at tid={idx}");
        }
        if crate::breadcrumbs::enabled() {
            let report = crate::breadcrumbs::format_removal_report(0);
            assert!(
                report.contains("dims=4x1x1"),
                "breadcrumb ring must capture compute dispatch dimensions: {report}"
            );
        }
        println!("SHADER_EXEC_256_US={elapsed_us}");
    }

    #[test]
    fn gpu_harness_null_uav_oob_reads_zero_and_drops_writes_without_device_removal() {
        let Some(device) = test_device() else {
            eprintln!("no D3D12 device; skipped");
            return;
        };
        let out = execute_null_uav_oob_probe(&device).expect("null UAV OOB probe");
        assert_eq!(out.len(), 512);
        for idx in 0..256 {
            assert_eq!(
                out[idx * 2],
                0,
                "null UAV OOB read at tid={idx} must return 0"
            );
            assert_eq!(
                out[idx * 2 + 1],
                0xA5A5_0000u32 | (idx as u32),
                "sentinel mismatch at tid={idx}"
            );
        }
    }
}
