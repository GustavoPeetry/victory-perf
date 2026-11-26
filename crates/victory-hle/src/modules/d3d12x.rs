//! `d3d12_xs.dll`: virtual memory setup, device creation, root signature
//! serialization. The device and everything it creates live in
//! `victory-gpu`.

use std::ffi::c_void;

use victory_gpu::device;
use victory_gpu::object::{HRESULT, Iid, Object};

use crate::{HleExport, HleModule};

fn register_rea_telemetry() {
    static REGISTERED: std::sync::Once = std::sync::Once::new();
    REGISTERED.call_once(|| {
        victory_gpu::rea_telemetry::register_cpu_stats_provider(|| {
            let s = victory_loader::console_insn_stats();
            victory_gpu::rea_telemetry::CpuTelemetry {
                fastpath_enabled: s.fastpath_enabled,
                patched_clzero_loops: s.patched_clzero_loops,
                patched_clzero_alt_loops: s.patched_clzero_alt_loops,
                patched_monitorx: s.patched_monitorx,
                patched_mwaitx: s.patched_mwaitx,
                fast_clzero_calls: s.fast_clzero_calls,
                fast_clzero_bytes: s.fast_clzero_bytes,
                fast_monitorx_calls: s.fast_monitorx_calls,
                fast_mwaitx_calls: s.fast_mwaitx_calls,
                wait_yields: s.wait_yields,
                wait_sleeps: s.wait_sleeps,
                wait_early_wakes: s.wait_early_wakes,
                emu_total: s.emu_total,
                emu_zeroed_bytes: s.emu_zeroed_bytes,
            }
        });
        victory_gpu::rea_telemetry::register_streaming_stats_provider(|| {
            let s = super::dstorage::streaming_stats();
            victory_gpu::rea_telemetry::StreamingTelemetry {
                enabled: s.enabled,
                scratch_reuses: s.scratch_reuses,
                scratch_bytes_saved: s.scratch_bytes_saved,
                low_ram_throttles: s.low_ram_throttles,
                page_dest_notifies: s.page_dest_notifies,
            }
        });
    });
}

/// Device creation; first hands the device the title's general allocator
/// (`XMemAlloc`, which honours the title's hooks) for resource memory.
extern "system" fn create_device(
    adapter: *const c_void,
    parameters: *const u32,
    iid: *const Iid,
    out: *mut *const Object,
) -> HRESULT {
    register_rea_telemetry();
    victory_gpu::resource::use_allocator(super::xmem::XMemAlloc, super::xmem::XMemFree);
    victory_gpu::resource::use_page_mapper(super::xmem::map_alias);
    #[cfg(windows)]
    victory_gpu::on_failure(host_gpu_failed);
    device::create(adapter, parameters, iid, out)
}

/// A host GPU failure ends the run like a crash: reported to the crash log,
/// then the process ends as the console ends a title.
#[cfg(windows)]
fn host_gpu_failed(what: &str) -> ! {
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, TerminateProcess};
    victory_loader::log_crash(&format!("VIctory: host GPU failure: {what}"));
    const DXGI_ERROR_DEVICE_REMOVED: u32 = 0x887A_0005;
    // SAFETY: ends our own process.
    unsafe { TerminateProcess(GetCurrentProcess(), DXGI_ERROR_DEVICE_REMOVED) };
    unreachable!("the process has ended")
}

#[cfg(windows)]
pub use victory_gpu::precompile;

/// Keeps the host's translations of the title's shaders in `dir` (a
/// folder of the cache) across runs, and its compiled pipelines beside it
/// (`pipelines`).
pub fn use_shader_cache(dir: std::path::PathBuf) {
    register_rea_telemetry();
    #[cfg(windows)]
    {
        victory_gpu::pipeline::use_pipeline_cache(dir.with_file_name("pipelines"));
        victory_gpu::shader_cache::use_shader_cache(dir);
    }
    #[cfg(not(windows))]
    let _ = dir;
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "d3d12x",
        dlls: &["d3d12_xs.dll"],
        exports: vec![
            HleExport {
                name: "D3D12ConfigureVirtualMemoryX",
                address: device::configure_virtual_memory as *const () as usize,
            },
            HleExport {
                name: "D3D12XboxCreateDevice",
                address: create_device as *const () as usize,
            },
            HleExport {
                name: "D3D12SerializeRootSignature",
                address: victory_gpu::root::serialize as *const () as usize,
            },
            HleExport {
                name: "D3D12SerializeVersionedRootSignature",
                address: victory_gpu::root::serialize_versioned as *const () as usize,
            },
            // The AMD upscaler's (Series S).
            HleExport {
                name: "D3D12CreateRootSignatureDeserializer",
                address: victory_gpu::root::create_deserializer as *const () as usize,
            },
            // A data import: the title reads the id at this address.
            HleExport {
                name: "IID_ID3D12Device",
                address: (&raw const device::IID) as usize,
            },
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HleResolver, HostPolicy};
    use victory_loader::ImportResolver;
    use victory_loader::Resolution;
    use victory_pe::ImportSymbol;

    #[test]
    fn the_interface_id_import_is_bound_to_data() {
        let symbol = ImportSymbol::Name {
            hint: 0,
            name: "IID_ID3D12Device".into(),
        };
        let r = HleResolver::new(HostPolicy::none());
        let Resolution::Found { address, .. } = r.resolve("d3d12_xs.dll", &symbol) else {
            panic!("not bound");
        };
        // SAFETY: the address is our static id.
        assert_eq!(unsafe { *(address as *const [u8; 16]) }, device::IID);
    }
}
