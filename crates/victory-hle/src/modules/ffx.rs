//! Opt-in FidelityFX / upscaler HLE surface (`VICTORY_REA003_UPSCALER=fsr3|dlss`).
//!
//! When `VICTORY_REA003_UPSCALER` is unset or `original` (the default), this module exposes
//! zero exports so default import resolution remains unchanged.
#![allow(non_snake_case)]

use std::sync::atomic::{AtomicU64, Ordering};

use crate::HleModule;

static FFX_CREATE_CALLS: AtomicU64 = AtomicU64::new(0);
static FFX_DISPATCH_CALLS: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfxHleMode {
    Original,
    Fsr3,
    Dlss,
}

impl FfxHleMode {
    pub fn from_env() -> Self {
        match std::env::var("VICTORY_REA003_UPSCALER").ok().as_deref() {
            Some(v) if v.trim().eq_ignore_ascii_case("fsr3") => Self::Fsr3,
            Some(v) if v.trim().eq_ignore_ascii_case("dlss") => Self::Dlss,
            _ => Self::Original,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Original => "original",
            Self::Fsr3 => "fsr3",
            Self::Dlss => "dlss",
        }
    }
}

/// Snapshot of opt-in FidelityFX HLE export mode and call counts.
pub fn telemetry_stats() -> (FfxHleMode, u64, u64) {
    (
        FfxHleMode::from_env(),
        FFX_CREATE_CALLS.load(Ordering::Relaxed),
        FFX_DISPATCH_CALLS.load(Ordering::Relaxed),
    )
}

/// Clean-room stub returning abstract REA-003 FFX interface revision when enabled.
pub extern "system" fn ffxGetInterfaceVersion() -> u32 {
    0x0003_0000
}

pub extern "system" fn ffxFsr3ContextCreate(_context: *mut usize, _desc: *const u8) -> i32 {
    FFX_CREATE_CALLS.fetch_add(1, Ordering::Relaxed);
    0
}

pub extern "system" fn ffxFsr3ContextDispatch(_context: *mut usize, desc: *const u8) -> i32 {
    FFX_DISPATCH_CALLS.fetch_add(1, Ordering::Relaxed);
    #[cfg(windows)]
    victory_gpu::dxr_proto::on_slot_0x218_call(
        FFX_DISPATCH_CALLS.load(Ordering::Relaxed) as u32,
        desc,
    );
    #[cfg(not(windows))]
    let _ = desc;
    0
}

pub extern "system" fn ffxFsr3ContextDestroy(_context: *mut usize) -> i32 {
    0
}

pub(crate) fn module_for_mode(mode: FfxHleMode) -> HleModule {
    let exports = match mode {
        FfxHleMode::Original => Vec::new(),
        FfxHleMode::Fsr3 | FfxHleMode::Dlss => vec![
            crate::hle_export!(ffxGetInterfaceVersion),
            crate::hle_export!(ffxFsr3ContextCreate),
            crate::hle_export!(ffxFsr3ContextDispatch),
            crate::hle_export!(ffxFsr3ContextDestroy),
        ],
    };
    HleModule {
        name: "ffx",
        dlls: &[
            "amd_fidelityfx_dx12.dll",
            "ffx_fsr2_api_dx12_x64.dll",
            "ffx_fsr3upscaler_x64.dll",
        ],
        exports,
    }
}

pub(crate) fn module() -> HleModule {
    module_for_mode(FfxHleMode::from_env())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffx_hle_module_is_inactive_in_original_mode_and_active_when_enabled() {
        assert_eq!(FfxHleMode::from_env(), FfxHleMode::Original);
        let orig = module_for_mode(FfxHleMode::Original);
        assert!(orig.exports.is_empty());

        let fsr3 = module_for_mode(FfxHleMode::Fsr3);
        assert_eq!(fsr3.exports.len(), 4);
        assert!(fsr3.serves("amd_fidelityfx_dx12.dll"));
        assert_eq!(ffxGetInterfaceVersion(), 0x0003_0000);
        assert_eq!(
            ffxFsr3ContextDispatch(std::ptr::null_mut(), std::ptr::null()),
            0
        );
    }
}
