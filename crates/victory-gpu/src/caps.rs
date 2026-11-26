//! Centralized desktop D3D12, DXGI, DXR, DirectML, Upscaler, and CPU capability probing.
//!
//! Probes real adapter/driver/OS/CPU capabilities at startup without assuming features from GPU
//! generation. Used by REA-003 render graph, shader recovery, DXR, and upscaling subsystems.

use std::ffi::c_void;
use std::sync::OnceLock;

use windows::Win32::Foundation::FreeLibrary;
use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
use windows::core::PCSTR;

/// Snapshot of real host adapter, D3D12, runtime library, and CPU capabilities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceCaps {
    pub adapter_name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub subsystem_id: u32,
    pub revision: u32,
    pub dedicated_vram_bytes: u64,
    pub dedicated_system_bytes: u64,
    pub shared_system_bytes: u64,
    pub dxgi_local_budget_bytes: u64,
    pub dxgi_local_usage_bytes: u64,
    pub max_feature_level_raw: i32,
    pub max_feature_level: String,
    pub max_shader_model_raw: i32,
    pub max_shader_model: String,
    pub raytracing_tier: i32,
    pub raytracing_tier_name: String,
    pub resource_binding_tier: i32,
    pub tiled_resources_tier: i32,
    pub resource_heap_tier: i32,
    pub conservative_rasterization_tier: i32,
    pub render_passes_tier: i32,
    pub mesh_shader_tier: i32,
    pub sampler_feedback_tier: i32,
    pub wave_ops: bool,
    pub wave_lane_count_min: u32,
    pub wave_lane_count_max: u32,
    pub total_lane_count: u32,
    pub enhanced_barriers_supported: bool,
    pub gpu_upload_heap_supported: bool,
    pub typed_uav_load_additional_formats: bool,
    pub double_precision_float_ops: bool,
    pub rovs_supported: bool,
    pub ps_specified_stencil_ref: bool,
    pub depth_bounds_test_supported: bool,
    pub int64_shader_ops: bool,
    pub atomic_int64_typed_resource: bool,
    pub atomic_int64_group_shared: bool,
    pub atomic_int64_descriptor_heap: bool,
    pub mesh_shader_pipeline_stats: bool,
    pub srv_only_tiled_resource_tier3: bool,
    pub directml_dll_present: bool,
    pub directml_create_device_present: bool,
    pub dlss_ngx_present: bool,
    pub dlss_ngx_path: Option<String>,
    pub fsr3_dll_present: bool,
    pub fsr3_dll_path: Option<String>,
    pub cpu_avx2: bool,
    pub cpu_erms: bool,
    pub cpu_fsrm: bool,
    pub cpu_waitpkg: bool,
}

static HOST_CAPS: OnceLock<DeviceCaps> = OnceLock::new();

/// Stores and caches the probed capabilities of the primary host device.
pub(crate) fn init_host_caps(
    device: &ID3D12Device,
    adapter: Option<&IDXGIAdapter3>,
) -> &'static DeviceCaps {
    HOST_CAPS.get_or_init(|| {
        let caps = probe_device_caps(device, adapter);
        tracing::info!(
            adapter = %caps.adapter_name,
            fl = %caps.max_feature_level,
            sm = %caps.max_shader_model,
            dxr = %caps.raytracing_tier_name,
            binding_tier = caps.resource_binding_tier,
            tiled_tier = caps.tiled_resources_tier,
            wave_ops = caps.wave_ops,
            wave_min = caps.wave_lane_count_min,
            wave_max = caps.wave_lane_count_max,
            mesh_tier = caps.mesh_shader_tier,
            enhanced_barriers = caps.enhanced_barriers_supported,
            gpu_upload_heap = caps.gpu_upload_heap_supported,
            typed_uav_load = caps.typed_uav_load_additional_formats,
            int64_ops = caps.int64_shader_ops,
            atomic64_typed = caps.atomic_int64_typed_resource,
            atomic64_lds = caps.atomic_int64_group_shared,
            atomic64_heap = caps.atomic_int64_descriptor_heap,
            directml = caps.directml_create_device_present,
            dlss_ngx = caps.dlss_ngx_present,
            fsr3_dll = caps.fsr3_dll_present,
            cpu_waitpkg = caps.cpu_waitpkg,
            "host desktop GPU & CPU capabilities probed"
        );
        caps
    })
}

/// Returns cached host capabilities if the device has been initialized.
pub fn host_caps() -> Option<&'static DeviceCaps> {
    HOST_CAPS.get()
}

fn feature_level_name(fl: i32) -> &'static str {
    match fl {
        x if x == D3D_FEATURE_LEVEL_12_2.0 => "12_2",
        x if x == D3D_FEATURE_LEVEL_12_1.0 => "12_1",
        x if x == D3D_FEATURE_LEVEL_12_0.0 => "12_0",
        x if x == D3D_FEATURE_LEVEL_11_1.0 => "11_1",
        x if x == D3D_FEATURE_LEVEL_11_0.0 => "11_0",
        _ => "unknown",
    }
}

fn shader_model_name(sm: i32) -> String {
    match sm {
        0x68 => "6_8".to_owned(),
        0x67 => "6_7".to_owned(),
        0x66 => "6_6".to_owned(),
        0x65 => "6_5".to_owned(),
        0x64 => "6_4".to_owned(),
        0x63 => "6_3".to_owned(),
        0x62 => "6_2".to_owned(),
        0x61 => "6_1".to_owned(),
        0x60 => "6_0".to_owned(),
        0x51 => "5_1".to_owned(),
        other => format!("{other:#x}"),
    }
}

fn raytracing_tier_name(tier: i32) -> &'static str {
    match tier {
        0 => "NotSupported",
        10 => "Tier1_0",
        11 => "Tier1_1",
        12 => "Tier1_2",
        _ => "Unknown",
    }
}

fn probe_dll(candidates: &[&str], symbol: Option<&[u8]>) -> (bool, bool, Option<String>) {
    for &candidate in candidates {
        let wide: Vec<u16> = candidate.encode_utf16().chain([0]).collect();
        // SAFETY: `wide` is NUL-terminated; loaded library is freed before returning.
        let module = unsafe { LoadLibraryW(windows::core::PCWSTR(wide.as_ptr())) };
        if let Ok(module) = module {
            let has_sym = match symbol {
                Some(sym) => unsafe { GetProcAddress(module, PCSTR(sym.as_ptr())) }.is_some(),
                None => true,
            };
            // SAFETY: module handle returned by LoadLibraryW.
            unsafe {
                let _ = FreeLibrary(module);
            }
            return (true, has_sym, Some(candidate.to_owned()));
        }
    }
    (false, false, None)
}

#[cfg(target_arch = "x86_64")]
fn probe_cpu_features() -> (bool, bool, bool, bool) {
    // CPUID leaf 0 and leaf 7 subleaf 0 are universally supported on x86-64.
    let max_leaf = std::arch::x86_64::__cpuid(0).eax;
    if max_leaf < 7 {
        return (false, false, false, false);
    }
    let leaf7 = std::arch::x86_64::__cpuid_count(7, 0);
    let avx2 = (leaf7.ebx & (1 << 5)) != 0;
    let erms = (leaf7.ebx & (1 << 9)) != 0;
    let waitpkg = (leaf7.ecx & (1 << 5)) != 0;
    let fsrm = (leaf7.edx & (1 << 4)) != 0;
    (avx2, erms, fsrm, waitpkg)
}

#[cfg(not(target_arch = "x86_64"))]
fn probe_cpu_features() -> (bool, bool, bool, bool) {
    (false, false, false, false)
}

/// Probes all relevant D3D12, DXGI, DXR, Wave, Barrier, Atomic, DirectML, Upscaler, and CPU features.
pub fn probe_device_caps(device: &ID3D12Device, adapter: Option<&IDXGIAdapter3>) -> DeviceCaps {
    let mut adapter_name = String::from("Unknown");
    let mut vendor_id = 0u32;
    let mut device_id = 0u32;
    let mut subsystem_id = 0u32;
    let mut revision = 0u32;
    let mut dedicated_vram_bytes = 0u64;
    let mut dedicated_system_bytes = 0u64;
    let mut shared_system_bytes = 0u64;
    let mut dxgi_local_budget_bytes = 0u64;
    let mut dxgi_local_usage_bytes = 0u64;

    let own_adapter;
    let adapter_ref = match adapter {
        Some(a) => Some(a),
        None => {
            // SAFETY: creates DXGI factory and enumerates the adapter matching `device`.
            own_adapter = unsafe {
                CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0)).and_then(|f| {
                    let luid = device.GetAdapterLuid();
                    f.EnumAdapterByLuid::<IDXGIAdapter3>(luid)
                })
            }
            .ok();
            own_adapter.as_ref()
        }
    };

    if let Some(a) = adapter_ref {
        // SAFETY: valid IDXGIAdapter3 interface.
        if let Ok(desc) = unsafe { a.GetDesc1() } {
            let len = desc
                .Description
                .iter()
                .position(|&c| c == 0)
                .unwrap_or(desc.Description.len());
            adapter_name = String::from_utf16_lossy(&desc.Description[..len]);
            vendor_id = desc.VendorId;
            device_id = desc.DeviceId;
            subsystem_id = desc.SubSysId;
            revision = desc.Revision;
            dedicated_vram_bytes = desc.DedicatedVideoMemory as u64;
            dedicated_system_bytes = desc.DedicatedSystemMemory as u64;
            shared_system_bytes = desc.SharedSystemMemory as u64;
        }
        let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: valid IDXGIAdapter3 and output struct.
        if unsafe { a.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info) }.is_ok()
        {
            dxgi_local_budget_bytes = info.Budget;
            dxgi_local_usage_bytes = info.CurrentUsage;
        }
    }

    // 1. Feature Levels
    let candidate_levels = [
        D3D_FEATURE_LEVEL_12_2,
        D3D_FEATURE_LEVEL_12_1,
        D3D_FEATURE_LEVEL_12_0,
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
    ];
    let mut fl_data = D3D12_FEATURE_DATA_FEATURE_LEVELS {
        NumFeatureLevels: candidate_levels.len() as u32,
        pFeatureLevelsRequested: candidate_levels.as_ptr(),
        MaxSupportedFeatureLevel: D3D_FEATURE_LEVEL_11_0,
    };
    let fl_ok = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_FEATURE_LEVELS,
            (&raw mut fl_data).cast::<c_void>(),
            size_of::<D3D12_FEATURE_DATA_FEATURE_LEVELS>() as u32,
        )
    }
    .is_ok();
    let max_feature_level_raw = if fl_ok {
        fl_data.MaxSupportedFeatureLevel.0
    } else {
        D3D_FEATURE_LEVEL_12_0.0
    };

    // 2. Shader Model (probe highest supported from 6.8 down to 6.0)
    let mut max_shader_model_raw = 0x51i32;
    for sm in [0x68i32, 0x67, 0x66, 0x65, 0x64, 0x63, 0x62, 0x61, 0x60] {
        let mut sm_data = D3D12_FEATURE_DATA_SHADER_MODEL {
            HighestShaderModel: D3D_SHADER_MODEL(sm),
        };
        if unsafe {
            device.CheckFeatureSupport(
                D3D12_FEATURE_SHADER_MODEL,
                (&raw mut sm_data).cast::<c_void>(),
                size_of::<D3D12_FEATURE_DATA_SHADER_MODEL>() as u32,
            )
        }
        .is_ok()
        {
            max_shader_model_raw = sm_data.HighestShaderModel.0;
            break;
        }
    }

    // 3. OPTIONS
    let mut opts = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    let _ = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS,
            (&raw mut opts).cast::<c_void>(),
            size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS>() as u32,
        )
    };

    // 4. OPTIONS1 (WaveOps, Int64)
    let mut opts1 = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
    let _ = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS1,
            (&raw mut opts1).cast::<c_void>(),
            size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS1>() as u32,
        )
    };

    // 5. OPTIONS2 (DepthBounds)
    let mut opts2 = D3D12_FEATURE_DATA_D3D12_OPTIONS2::default();
    let _ = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS2,
            (&raw mut opts2).cast::<c_void>(),
            size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS2>() as u32,
        )
    };

    // 6. OPTIONS5 (RaytracingTier, RenderPassesTier)
    let mut opts5 = D3D12_FEATURE_DATA_D3D12_OPTIONS5::default();
    let _ = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS5,
            (&raw mut opts5).cast::<c_void>(),
            size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS5>() as u32,
        )
    };

    // 7. OPTIONS7 (MeshShaderTier, SamplerFeedbackTier)
    let mut opts7 = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
    let _ = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS7,
            (&raw mut opts7).cast::<c_void>(),
            size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS7>() as u32,
        )
    };

    // 8. OPTIONS9 (AtomicInt64 on Typed/GroupShared, MeshShaderPipelineStats)
    let mut opts9 = D3D12_FEATURE_DATA_D3D12_OPTIONS9::default();
    let _ = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS9,
            (&raw mut opts9).cast::<c_void>(),
            size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS9>() as u32,
        )
    };

    // 9. OPTIONS11 (AtomicInt64 on DescriptorHeapResource)
    let mut opts11 = D3D12_FEATURE_DATA_D3D12_OPTIONS11::default();
    let _ = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS11,
            (&raw mut opts11).cast::<c_void>(),
            size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS11>() as u32,
        )
    };

    // 10. OPTIONS12 (EnhancedBarriers)
    let mut opts12 = D3D12_FEATURE_DATA_D3D12_OPTIONS12::default();
    let _ = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS12,
            (&raw mut opts12).cast::<c_void>(),
            size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS12>() as u32,
        )
    };

    // 11. OPTIONS16 (GPUUploadHeapSupported)
    let mut opts16 = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
    let _ = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS16,
            (&raw mut opts16).cast::<c_void>(),
            size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS16>() as u32,
        )
    };

    let (directml_dll_present, directml_create_device_present, _) =
        probe_dll(&["DirectML.dll"], Some(b"DMLCreateDevice\0"));

    let mut dlss_candidates: Vec<String> = Vec::new();
    if let Ok(env_path) = std::env::var("VICTORY_DLSS_SDK_DLL") {
        let trimmed = env_path.trim();
        if !trimmed.is_empty() {
            dlss_candidates.push(trimmed.to_owned());
        }
    }
    dlss_candidates.extend(
        ["nvngx_dlss.dll", "_nvngx.dll"]
            .into_iter()
            .map(str::to_owned),
    );
    let dlss_refs: Vec<&str> = dlss_candidates.iter().map(String::as_str).collect();
    let (dlss_ngx_present, _, dlss_ngx_path) = probe_dll(&dlss_refs, None);

    let mut fsr_candidates: Vec<String> = Vec::new();
    if let Ok(env_path) = std::env::var("VICTORY_FSR3_DLL") {
        let trimmed = env_path.trim();
        if !trimmed.is_empty() {
            fsr_candidates.push(trimmed.to_owned());
        }
    }
    fsr_candidates.extend(
        [
            "amd_fidelityfx_dx12.dll",
            "amd_fidelityfx_upscaler_dx12.dll",
            "ffx_fsr3upscaler_x64.dll",
        ]
        .into_iter()
        .map(str::to_owned),
    );
    let fsr_refs: Vec<&str> = fsr_candidates.iter().map(String::as_str).collect();
    let (fsr3_dll_present, _, fsr3_dll_path) = probe_dll(&fsr_refs, None);

    let (cpu_avx2, cpu_erms, cpu_fsrm, cpu_waitpkg) = probe_cpu_features();

    DeviceCaps {
        adapter_name,
        vendor_id,
        device_id,
        subsystem_id,
        revision,
        dedicated_vram_bytes,
        dedicated_system_bytes,
        shared_system_bytes,
        dxgi_local_budget_bytes,
        dxgi_local_usage_bytes,
        max_feature_level_raw,
        max_feature_level: feature_level_name(max_feature_level_raw).to_owned(),
        max_shader_model_raw,
        max_shader_model: shader_model_name(max_shader_model_raw),
        raytracing_tier: opts5.RaytracingTier.0,
        raytracing_tier_name: raytracing_tier_name(opts5.RaytracingTier.0).to_owned(),
        resource_binding_tier: opts.ResourceBindingTier.0,
        tiled_resources_tier: opts.TiledResourcesTier.0,
        resource_heap_tier: opts.ResourceHeapTier.0,
        conservative_rasterization_tier: opts.ConservativeRasterizationTier.0,
        render_passes_tier: opts5.RenderPassesTier.0,
        mesh_shader_tier: opts7.MeshShaderTier.0,
        sampler_feedback_tier: opts7.SamplerFeedbackTier.0,
        wave_ops: opts1.WaveOps.as_bool(),
        wave_lane_count_min: opts1.WaveLaneCountMin,
        wave_lane_count_max: opts1.WaveLaneCountMax,
        total_lane_count: opts1.TotalLaneCount,
        enhanced_barriers_supported: opts12.EnhancedBarriersSupported.as_bool(),
        gpu_upload_heap_supported: opts16.GPUUploadHeapSupported.as_bool(),
        typed_uav_load_additional_formats: opts.TypedUAVLoadAdditionalFormats.as_bool(),
        double_precision_float_ops: opts.DoublePrecisionFloatShaderOps.as_bool(),
        rovs_supported: opts.ROVsSupported.as_bool(),
        ps_specified_stencil_ref: opts.PSSpecifiedStencilRefSupported.as_bool(),
        depth_bounds_test_supported: opts2.DepthBoundsTestSupported.as_bool(),
        int64_shader_ops: opts1.Int64ShaderOps.as_bool(),
        atomic_int64_typed_resource: opts9.AtomicInt64OnTypedResourceSupported.as_bool(),
        atomic_int64_group_shared: opts9.AtomicInt64OnGroupSharedSupported.as_bool(),
        atomic_int64_descriptor_heap: opts11
            .AtomicInt64OnDescriptorHeapResourceSupported
            .as_bool(),
        mesh_shader_pipeline_stats: opts9.MeshShaderPipelineStatsSupported.as_bool(),
        srv_only_tiled_resource_tier3: opts5.SRVOnlyTiledResourceTier3.as_bool(),
        directml_dll_present,
        directml_create_device_present,
        dlss_ngx_present,
        dlss_ngx_path,
        fsr3_dll_present,
        fsr3_dll_path,
        cpu_avx2,
        cpu_erms,
        cpu_fsrm,
        cpu_waitpkg,
    }
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn json_opt_str(s: Option<&str>) -> String {
    match s {
        Some(v) => json_escape(v),
        None => "null".to_owned(),
    }
}

impl DeviceCaps {
    /// Serializes the probed capabilities into formatted JSON.
    pub fn to_json(&self) -> String {
        format!(
            concat!(
                "{{\n",
                "  \"adapter_name\": {},\n",
                "  \"vendor_id\": {},\n",
                "  \"device_id\": {},\n",
                "  \"subsystem_id\": {},\n",
                "  \"revision\": {},\n",
                "  \"dedicated_vram_bytes\": {},\n",
                "  \"dedicated_system_bytes\": {},\n",
                "  \"shared_system_bytes\": {},\n",
                "  \"dxgi_local_budget_bytes\": {},\n",
                "  \"dxgi_local_usage_bytes\": {},\n",
                "  \"max_feature_level_raw\": {},\n",
                "  \"max_feature_level\": {},\n",
                "  \"max_shader_model_raw\": {},\n",
                "  \"max_shader_model\": {},\n",
                "  \"raytracing_tier\": {},\n",
                "  \"raytracing_tier_name\": {},\n",
                "  \"resource_binding_tier\": {},\n",
                "  \"tiled_resources_tier\": {},\n",
                "  \"resource_heap_tier\": {},\n",
                "  \"conservative_rasterization_tier\": {},\n",
                "  \"render_passes_tier\": {},\n",
                "  \"mesh_shader_tier\": {},\n",
                "  \"sampler_feedback_tier\": {},\n",
                "  \"wave_ops\": {},\n",
                "  \"wave_lane_count_min\": {},\n",
                "  \"wave_lane_count_max\": {},\n",
                "  \"total_lane_count\": {},\n",
                "  \"enhanced_barriers_supported\": {},\n",
                "  \"gpu_upload_heap_supported\": {},\n",
                "  \"typed_uav_load_additional_formats\": {},\n",
                "  \"double_precision_float_ops\": {},\n",
                "  \"rovs_supported\": {},\n",
                "  \"ps_specified_stencil_ref\": {},\n",
                "  \"depth_bounds_test_supported\": {},\n",
                "  \"int64_shader_ops\": {},\n",
                "  \"atomic_int64_typed_resource\": {},\n",
                "  \"atomic_int64_group_shared\": {},\n",
                "  \"atomic_int64_descriptor_heap\": {},\n",
                "  \"mesh_shader_pipeline_stats\": {},\n",
                "  \"srv_only_tiled_resource_tier3\": {},\n",
                "  \"directml_dll_present\": {},\n",
                "  \"directml_create_device_present\": {},\n",
                "  \"dlss_ngx_present\": {},\n",
                "  \"dlss_ngx_path\": {},\n",
                "  \"fsr3_dll_present\": {},\n",
                "  \"fsr3_dll_path\": {},\n",
                "  \"cpu_avx2\": {},\n",
                "  \"cpu_erms\": {},\n",
                "  \"cpu_fsrm\": {},\n",
                "  \"cpu_waitpkg\": {}\n",
                "}}"
            ),
            json_escape(&self.adapter_name),
            self.vendor_id,
            self.device_id,
            self.subsystem_id,
            self.revision,
            self.dedicated_vram_bytes,
            self.dedicated_system_bytes,
            self.shared_system_bytes,
            self.dxgi_local_budget_bytes,
            self.dxgi_local_usage_bytes,
            self.max_feature_level_raw,
            json_escape(&self.max_feature_level),
            self.max_shader_model_raw,
            json_escape(&self.max_shader_model),
            self.raytracing_tier,
            json_escape(&self.raytracing_tier_name),
            self.resource_binding_tier,
            self.tiled_resources_tier,
            self.resource_heap_tier,
            self.conservative_rasterization_tier,
            self.render_passes_tier,
            self.mesh_shader_tier,
            self.sampler_feedback_tier,
            self.wave_ops,
            self.wave_lane_count_min,
            self.wave_lane_count_max,
            self.total_lane_count,
            self.enhanced_barriers_supported,
            self.gpu_upload_heap_supported,
            self.typed_uav_load_additional_formats,
            self.double_precision_float_ops,
            self.rovs_supported,
            self.ps_specified_stencil_ref,
            self.depth_bounds_test_supported,
            self.int64_shader_ops,
            self.atomic_int64_typed_resource,
            self.atomic_int64_group_shared,
            self.atomic_int64_descriptor_heap,
            self.mesh_shader_pipeline_stats,
            self.srv_only_tiled_resource_tier3,
            self.directml_dll_present,
            self.directml_create_device_present,
            self.dlss_ngx_present,
            json_opt_str(self.dlss_ngx_path.as_deref()),
            self.fsr3_dll_present,
            json_opt_str(self.fsr3_dll_path.as_deref()),
            self.cpu_avx2,
            self.cpu_erms,
            self.cpu_fsrm,
            self.cpu_waitpkg,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_real_adapter_capabilities() {
        let mut device: Option<ID3D12Device> = None;
        if unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device) }.is_err() {
            return;
        }
        let device = device.unwrap();
        let caps = probe_device_caps(&device, None);
        assert!(!caps.adapter_name.is_empty());
        assert!(caps.max_feature_level_raw >= D3D_FEATURE_LEVEL_12_0.0);
        println!("REA003_DEVICE_CAPS_JSON_START");
        println!("{}", caps.to_json());
        println!("REA003_DEVICE_CAPS_JSON_END");
    }
}
