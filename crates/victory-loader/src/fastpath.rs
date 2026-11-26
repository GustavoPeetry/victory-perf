//! Native x86-64 fast paths eliminating repeated `#UD` traps for AMD console-only
//! instructions (Front D).
//!
//! When `VICTORY_REA_NATIVE_CPU_FASTPATH=1` is set:
//! 1. At PE image load time (before any title thread runs), executable sections
//!    are scanned for verified multi-instruction `clzero` loop signatures (both the
//!    6 standard `ClzeroLoop` sites and the 4 alternate `ClzeroAltLoop` sites) and
//!    both `monitorx`/`mwaitx` synchronization pairs.
//! 2. Matched sites are patched with `E9 rel32` jumps into strict W^X nearby
//!    trampoline slabs: Page 0 holds generated machine code (`PAGE_RW -> PAGE_RX`
//!    + `FlushInstructionCache`) and Page 1 holds `PAGE_RW` telemetry counters
//!      updated via RIP-relative `lock add` / `lock inc`.
//! 3. `clzero` loops run natively via `rep stosb` (ERMS/FSRM) with 0 exceptions
//!    while preserving registers and `RFLAGS`; `monitorx`/`mwaitx` yield cleanly
//!    via Windows `WaitOnAddress` instead of busy spinning or trapping.

use crate::console_insn_decode as decode;

use std::cell::Cell;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READ,
    PAGE_EXECUTE_READWRITE, PAGE_GUARD, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
    VirtualProtect, VirtualQuery,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

const PAGE_SIZE: usize = 4096;

/// Master switch (`VICTORY_REA_NATIVE_CPU_FASTPATH=1`, OFF by default).
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("VICTORY_REA_NATIVE_CPU_FASTPATH")
            .ok()
            .is_some_and(|v| {
                let t = v.trim();
                t == "1" || t.eq_ignore_ascii_case("true") || t.eq_ignore_ascii_case("yes")
            });
        if on {
            tracing::info!("REA-002 native CPU instruction fast paths enabled");
        }
        on
    })
}

/// Global telemetry counters across both VEH fallback and native trampolines.
static EMU_CLZERO_SINGLE: AtomicU64 = AtomicU64::new(0);
static EMU_CLZERO_LOOPS: AtomicU64 = AtomicU64::new(0);
static EMU_CLZERO_ALT_LOOPS: AtomicU64 = AtomicU64::new(0);
static EMU_MONITORX: AtomicU64 = AtomicU64::new(0);
static EMU_MWAITX: AtomicU64 = AtomicU64::new(0);
static EMU_OTHER: AtomicU64 = AtomicU64::new(0);
static EMU_TOTAL: AtomicU64 = AtomicU64::new(0);
static EMU_ASKED: AtomicU64 = AtomicU64::new(0);
static EMU_ZEROED_BYTES: AtomicU64 = AtomicU64::new(0);

static PATCHED_CLZERO_LOOPS: AtomicU64 = AtomicU64::new(0);
static PATCHED_CLZERO_ALT_LOOPS: AtomicU64 = AtomicU64::new(0);
static PATCHED_MONITORX: AtomicU64 = AtomicU64::new(0);
static PATCHED_MWAITX: AtomicU64 = AtomicU64::new(0);
static FAST_MONITORX_CALLS: AtomicU64 = AtomicU64::new(0);
static FAST_MWAITX_CALLS: AtomicU64 = AtomicU64::new(0);
static WAIT_ON_ADDRESS_CALLS: AtomicU64 = AtomicU64::new(0);

/// Telemetry snapshot exported via [`console_insn_stats`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ConsoleInsnStats {
    pub fastpath_enabled: bool,
    pub emu_total: u64,
    pub emu_clzero_single: u64,
    pub emu_clzero_loops: u64,
    pub emu_clzero_alt_loops: u64,
    pub emu_monitorx: u64,
    pub emu_mwaitx: u64,
    pub emu_other: u64,
    pub emu_asked: u64,
    pub emu_zeroed_bytes: u64,
    pub patched_clzero_loops: u64,
    pub patched_clzero_alt_loops: u64,
    pub patched_monitorx: u64,
    pub patched_mwaitx: u64,
    pub fast_clzero_calls: u64,
    pub fast_clzero_bytes: u64,
    pub fast_monitorx_calls: u64,
    pub fast_mwaitx_calls: u64,
    pub wait_on_address_calls: u64,
}

#[repr(C)]
struct SlabData {
    fast_clzero_calls: AtomicU64,
    fast_clzero_bytes: AtomicU64,
    on_monitorx_fn: usize,
    on_mwaitx_fn: usize,
}

struct SlabRecord {
    code_page: usize,
    data_page: usize,
}

static SLABS: Mutex<Vec<SlabRecord>> = Mutex::new(Vec::new());

pub(crate) fn record_veh_asked(n: u64) {
    EMU_ASKED.fetch_add(n, Ordering::Relaxed);
}

pub(crate) fn record_veh_zeroed(bytes: u64) {
    EMU_ZEROED_BYTES.fetch_add(bytes, Ordering::Relaxed);
}

pub(crate) fn record_veh_insn(kind: &decode::ConsoleOnly) {
    EMU_TOTAL.fetch_add(1, Ordering::Relaxed);
    match kind {
        decode::ConsoleOnly::Clzero => EMU_CLZERO_SINGLE.fetch_add(1, Ordering::Relaxed),
        decode::ConsoleOnly::ClzeroLoop { .. } => EMU_CLZERO_LOOPS.fetch_add(1, Ordering::Relaxed),
        decode::ConsoleOnly::ClzeroAltLoop { .. } => {
            EMU_CLZERO_ALT_LOOPS.fetch_add(1, Ordering::Relaxed)
        }
        decode::ConsoleOnly::Monitorx => EMU_MONITORX.fetch_add(1, Ordering::Relaxed),
        decode::ConsoleOnly::Mwaitx => EMU_MWAITX.fetch_add(1, Ordering::Relaxed),
        _ => EMU_OTHER.fetch_add(1, Ordering::Relaxed),
    };
}

/// Snapshot all emulated `#UD` and native fast-path metrics.
pub fn console_insn_stats() -> ConsoleInsnStats {
    let mut fast_calls = 0u64;
    let mut fast_bytes = 0u64;
    if let Ok(slabs) = SLABS.try_lock() {
        for s in slabs.iter() {
            // SAFETY: `s.data_page` is a leaked `PAGE_READWRITE` page holding `SlabData`.
            let data = unsafe { &*(s.data_page as *const SlabData) };
            fast_calls += data.fast_clzero_calls.load(Ordering::Relaxed);
            fast_bytes += data.fast_clzero_bytes.load(Ordering::Relaxed);
        }
    }
    ConsoleInsnStats {
        fastpath_enabled: enabled(),
        emu_total: EMU_TOTAL.load(Ordering::Relaxed),
        emu_clzero_single: EMU_CLZERO_SINGLE.load(Ordering::Relaxed),
        emu_clzero_loops: EMU_CLZERO_LOOPS.load(Ordering::Relaxed),
        emu_clzero_alt_loops: EMU_CLZERO_ALT_LOOPS.load(Ordering::Relaxed),
        emu_monitorx: EMU_MONITORX.load(Ordering::Relaxed),
        emu_mwaitx: EMU_MWAITX.load(Ordering::Relaxed),
        emu_other: EMU_OTHER.load(Ordering::Relaxed),
        emu_asked: EMU_ASKED.load(Ordering::Relaxed),
        emu_zeroed_bytes: EMU_ZEROED_BYTES.load(Ordering::Relaxed),
        patched_clzero_loops: PATCHED_CLZERO_LOOPS.load(Ordering::Relaxed),
        patched_clzero_alt_loops: PATCHED_CLZERO_ALT_LOOPS.load(Ordering::Relaxed),
        patched_monitorx: PATCHED_MONITORX.load(Ordering::Relaxed),
        patched_mwaitx: PATCHED_MWAITX.load(Ordering::Relaxed),
        fast_clzero_calls: fast_calls,
        fast_clzero_bytes: fast_bytes,
        fast_monitorx_calls: FAST_MONITORX_CALLS.load(Ordering::Relaxed),
        fast_mwaitx_calls: FAST_MWAITX_CALLS.load(Ordering::Relaxed),
        wait_on_address_calls: WAIT_ON_ADDRESS_CALLS.load(Ordering::Relaxed),
    }
}

type WaitOnAddressFn = unsafe extern "system" fn(
    address: *const c_void,
    compare_address: *const c_void,
    address_size: usize,
    dw_milliseconds: u32,
) -> i32;

fn wait_on_address_fn() -> Option<WaitOnAddressFn> {
    static FN: OnceLock<Option<WaitOnAddressFn>> = OnceLock::new();
    *FN.get_or_init(|| unsafe {
        let kb = GetModuleHandleA(c"kernelbase.dll".as_ptr().cast());
        if kb.is_null() {
            return None;
        }
        let proc = GetProcAddress(kb, c"WaitOnAddress".as_ptr().cast())?;
        Some(std::mem::transmute::<
            unsafe extern "system" fn() -> isize,
            WaitOnAddressFn,
        >(proc))
    })
}

thread_local! {
    static MONITOR_STATE: Cell<(usize, u64)> = const { Cell::new((0, 0)) };
}

fn is_readable_u64(va: usize) -> bool {
    if !(0x1_0000..=0x0000_7FFF_FFFF_FFFF).contains(&va) {
        return false;
    }
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    let got = unsafe {
        VirtualQuery(
            va as *const c_void,
            &mut info,
            size_of::<MEMORY_BASIC_INFORMATION>(),
        )
    };
    if got == 0 || info.State != MEM_COMMIT {
        return false;
    }
    if (info.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0 {
        return false;
    }
    (info.BaseAddress as usize).saturating_add(info.RegionSize) >= va + 8
}

extern "C" fn on_monitorx(va: usize) {
    FAST_MONITORX_CALLS.fetch_add(1, Ordering::Relaxed);
    let val = if is_readable_u64(va) {
        // SAFETY: committed readable 8-byte slot verified immediately above.
        unsafe { (va as *const u64).read_unaligned() }
    } else {
        0
    };
    MONITOR_STATE.with(|s| s.set((va, val)));
}

extern "C" fn on_mwaitx() {
    FAST_MWAITX_CALLS.fetch_add(1, Ordering::Relaxed);
    let (va, snap) = MONITOR_STATE.with(Cell::get);
    if va != 0 && is_readable_u64(va) {
        // SAFETY: committed readable slot verified above.
        let current = unsafe { (va as *const u64).read_unaligned() };
        if current == snap
            && let Some(wait_fn) = wait_on_address_fn()
        {
            WAIT_ON_ADDRESS_CALLS.fetch_add(1, Ordering::Relaxed);
            // Bounded 1 ms wait avoids indefinite sleep on missed updates while
            // replacing 100% busy CPU spin loops on 4-core hardware.
            let _ = unsafe {
                wait_fn(
                    va as *const c_void,
                    (&raw const snap).cast::<c_void>(),
                    8,
                    1,
                )
            };
            return;
        }
    }
    std::thread::yield_now();
}

/// Expose synchronous wait helper directly for tests & diagnostics.
#[cfg(test)]
pub(crate) fn probe_wait_on_address_pair(va: *const u64) {
    on_monitorx(va as usize);
    on_mwaitx();
}

struct CodeEmitter {
    bytes: Vec<u8>,
    base_va: usize,
    data_va: usize,
}

impl CodeEmitter {
    fn new(base_va: usize, data_va: usize) -> Self {
        Self {
            bytes: Vec::new(),
            base_va,
            data_va,
        }
    }

    #[inline]
    fn cur_va(&self) -> usize {
        self.base_va + self.bytes.len()
    }

    #[inline]
    fn push(&mut self, b: u8) {
        self.bytes.push(b);
    }

    #[inline]
    fn extend(&mut self, slice: &[u8]) {
        self.bytes.extend_from_slice(slice);
    }

    fn emit_rel32_to(&mut self, target_va: usize) -> Result<i32, ()> {
        let next_ip = (self.cur_va() + 4) as i128;
        let delta = target_va as i128 - next_ip;
        i32::try_from(delta).map_err(|_| ())
    }

    fn emit_jmp(&mut self, target_va: usize) -> Result<(), ()> {
        self.push(0xE9);
        let rel = self.emit_rel32_to(target_va)?;
        self.extend(&rel.to_le_bytes());
        Ok(())
    }

    /// `mov r64(dst), r64(src)`: REX.W + 89 /r
    fn emit_mov_r64_r64(&mut self, dst: u8, src: u8) {
        let rex = 0x48 | (((src >> 3) & 1) << 2) | ((dst >> 3) & 1);
        let modrm = 0xC0 | ((src & 7) << 3) | (dst & 7);
        self.extend(&[rex, 0x89, modrm]);
    }

    /// `cmp r64(lhs), r64(rhs)`: REX.W + 39 /r
    fn emit_cmp_r64_r64(&mut self, lhs: u8, rhs: u8) {
        let rex = 0x48 | (((rhs >> 3) & 1) << 2) | ((lhs >> 3) & 1);
        let modrm = 0xC0 | ((rhs & 7) << 3) | (lhs & 7);
        self.extend(&[rex, 0x39, modrm]);
    }

    /// `lea rax, [r64(base) + disp8]` (must not use RSP/R12 base without SIB).
    fn emit_lea_rax_disp8(&mut self, base: u8, disp8: i8) {
        let b3 = (base >> 3) & 1;
        let rex = 0x48 | b3;
        if (base & 7) == 4 {
            // RSP / R12 requires SIB byte 0x24.
            self.extend(&[rex, 0x8D, 0x44, 0x24, disp8 as u8]);
        } else {
            self.extend(&[rex, 0x8D, 0x40 | (base & 7), disp8 as u8]);
        }
    }

    /// Load `reg_idx` (captured before pushes touched RSP/RCX/RDI) into `dst_reg`,
    /// given stack layout `[rsp] = orig_rdi, [rsp+8] = orig_rcx, [rsp+16] = orig_rflags`.
    fn emit_read_saved_gpr(&mut self, dst_reg: u8, src_reg: u8) {
        match src_reg & 15 {
            // RCX was pushed to [rsp + 8].
            1 => {
                let rex = 0x48 | (((dst_reg >> 3) & 1) << 2);
                let modrm = 0x44 | ((dst_reg & 7) << 3);
                self.extend(&[rex, 0x8B, modrm, 0x24, 0x08]);
            }
            // RDI was pushed to [rsp].
            7 => {
                let rex = 0x48 | (((dst_reg >> 3) & 1) << 2);
                let modrm = 0x04 | ((dst_reg & 7) << 3);
                self.extend(&[rex, 0x8B, modrm, 0x24]);
            }
            other => self.emit_mov_r64_r64(dst_reg, other),
        }
    }

    fn emit_lock_inc_data(&mut self, field_offset: usize) -> Result<(), ()> {
        let target = self.data_va + field_offset;
        self.extend(&[0xF0, 0x48, 0xFF, 0x05]);
        let disp = self.emit_rel32_to(target)?;
        self.extend(&disp.to_le_bytes());
        Ok(())
    }

    fn emit_lock_add_rcx_data(&mut self, field_offset: usize) -> Result<(), ()> {
        let target = self.data_va + field_offset;
        self.extend(&[0xF0, 0x48, 0x01, 0x0D]);
        let disp = self.emit_rel32_to(target)?;
        self.extend(&disp.to_le_bytes());
        Ok(())
    }

    /// Emit native stub for `ClzeroLoop { end_reg }`, returning to `site_va + 12`.
    fn emit_clzero_loop_stub(&mut self, end_reg: u8, return_va: usize) -> Result<usize, ()> {
        let entry = self.cur_va();
        // Save RFLAGS, RCX, RDI -> [rsp]=orig_rdi, [rsp+8]=orig_rcx, [rsp+16]=orig_rflags
        self.extend(&[0x9C, 0x51, 0x57]);
        // rdi = start (orig RAX), rcx = end (orig end_reg)
        self.emit_mov_r64_r64(7, 0);
        self.emit_read_saved_gpr(1, end_reg);
        // rcx = rcx - rdi (unsigned length)
        self.extend(&[0x48, 0x29, 0xF9]);
        // jbe skip_zero (len == 0 or end < start)
        self.extend(&[0x76, 0x00]);
        let jbe_disp_idx = self.bytes.len() - 1;
        // Guard len <= 0x7FFF_FFFF (`mov eax, 0x7FFFFFFF; cmp rcx, rax; ja skip`)
        self.extend(&[0xB8, 0xFF, 0xFF, 0xFF, 0x7F, 0x48, 0x39, 0xC1]);
        self.extend(&[0x77, 0x00]);
        let ja_disp_idx = self.bytes.len() - 1;

        // Telemetry: lock inc calls; lock add bytes, rcx (offsets 0 and 8 in SlabData)
        self.emit_lock_inc_data(0)?;
        self.emit_lock_add_rcx_data(8)?;

        // Fast ERMS zeroing: xor eax, eax; cld; rep stosb -> advances RDI to end!
        self.extend(&[0x31, 0xC0, 0xFC, 0xF3, 0xAA]);
        // RAX = end (RDI)
        self.emit_mov_r64_r64(0, 7);
        // jmp after_skip (3-byte `pop rdi; pop rcx; popfq` below starts right after skip!)
        self.extend(&[0xEB, 0x05]);

        let skip_pos = self.bytes.len();
        self.bytes[jbe_disp_idx] = (skip_pos - (jbe_disp_idx + 1)) as u8;
        self.bytes[ja_disp_idx] = (skip_pos - (ja_disp_idx + 1)) as u8;
        // On skip (e.g. `end == start`), match VEH ClzeroLoop behavior (`RAX = end`).
        self.emit_read_saved_gpr(0, end_reg);
        //Wait: read_saved_gpr only reads 3..5 bytes! Wait above `eb 05` might mismatch exact byte length of emit_read_saved_gpr!
        let after_skip_start = self.bytes.len();
        self.bytes[skip_pos - 1] = (after_skip_start - skip_pos) as u8;

        // Restore RDI, RCX, RFLAGS, then `cmp rax, end_reg` to leave loop-exit flags exact!
        self.extend(&[0x5F, 0x59, 0x9D]);
        self.emit_cmp_r64_r64(0, end_reg);
        self.emit_jmp(return_va)?;
        Ok(entry)
    }

    /// Emit native stub for `ClzeroAltLoop { cur_reg, end_reg }`, returning to `site_va + 15`.
    fn emit_clzero_alt_loop_stub(
        &mut self,
        cur_reg: u8,
        end_reg: u8,
        return_va: usize,
    ) -> Result<usize, ()> {
        let entry = self.cur_va();
        self.extend(&[0x9C, 0x51, 0x57]);
        // RAX = orig cur_reg; RCX = orig end_reg
        self.emit_read_saved_gpr(0, cur_reg);
        self.emit_read_saved_gpr(1, end_reg);
        // RDI = RAX; RCX = end - start
        self.emit_mov_r64_r64(7, 0);
        self.extend(&[0x48, 0x29, 0xF9]);
        self.extend(&[0x76, 0x00]);
        let jbe_disp_idx = self.bytes.len() - 1;
        self.extend(&[0xB8, 0xFF, 0xFF, 0xFF, 0x7F, 0x48, 0x39, 0xC1]);
        self.extend(&[0x77, 0x00]);
        let ja_disp_idx = self.bytes.len() - 1;

        self.emit_lock_inc_data(0)?;
        self.emit_lock_add_rcx_data(8)?;
        self.extend(&[0x31, 0xC0, 0xFC, 0xF3, 0xAA]);

        // Restore RDI, RCX, RFLAGS; then update exit registers:
        // RAX = end_reg - 0x40; cur_reg = end_reg; cmp cur_reg, end_reg.
        self.extend(&[0x5F, 0x59, 0x9D]);
        self.emit_lea_rax_disp8(end_reg, -0x40);
        self.emit_mov_r64_r64(cur_reg, end_reg);
        self.emit_cmp_r64_r64(cur_reg, end_reg);
        self.emit_jmp(return_va)?;

        let skip_pos = self.bytes.len();
        self.bytes[jbe_disp_idx] = (skip_pos - (jbe_disp_idx + 1)) as u8;
        self.bytes[ja_disp_idx] = (skip_pos - (ja_disp_idx + 1)) as u8;
        // Skip path: restore regs and fall through with `mov rax, cur_reg; add cur_reg, 0x40; cmp cur_reg, end_reg`.
        self.extend(&[0x5F, 0x59, 0x9D]);
        self.emit_mov_r64_r64(0, cur_reg);
        self.emit_cmp_r64_r64(cur_reg, end_reg);
        self.emit_jmp(return_va)?;
        Ok(entry)
    }

    /// Emit ABI-preserving call thunk (`save RFLAGS + all volatile GPRs + XMM0..3`)
    /// calling `SlabData[fn_field_off]` with `rcx = orig_rax`.
    fn emit_call_thunk(
        &mut self,
        fn_field_off: usize,
        post_mode: CallThunkPost,
        return_va: usize,
    ) -> Result<usize, ()> {
        let entry = self.cur_va();
        // Save RFLAGS and volatile GPRs:
        self.extend(&[
            0x9C, // pushfq
            0x50, 0x51, 0x52, // push rax, rcx, rdx
            0x41, 0x50, 0x41, 0x51, 0x41, 0x52, 0x41, 0x53, // push r8, r9, r10, r11
            // Put monitored addr (`orig rax`) into rcx before aligning rsp:
            0x48, 0x8B, 0x4C, 0x24, 0x30, // mov rcx, [rsp + 0x30]
            // Preserve RBP and align stack to 16 bytes + 32-byte Win64 shadow space:
            0x55, // push rbp
            0x48, 0x89, 0xE5, // mov rbp, rsp
            0x48, 0x83, 0xE4, 0xF0, // and rsp, -16
            0x48, 0x83, 0xEC, 0x40, // sub rsp, 64 (32B shadow + room for xmm save)
            0x0F, 0x11, 0x44, 0x24, 0x20, // movups [rsp + 0x20], xmm0
            0x0F, 0x11, 0x4C, 0x24, 0x30, // movups [rsp + 0x30], xmm1
        ]);
        // call qword ptr [rip + SlabData.fn_field_off] (`FF 15 disp32`)
        self.extend(&[0xFF, 0x15]);
        let disp = self.emit_rel32_to(self.data_va + fn_field_off)?;
        self.extend(&disp.to_le_bytes());

        self.extend(&[
            0x0F, 0x10, 0x44, 0x24, 0x20, // movups xmm0, [rsp + 0x20]
            0x0F, 0x10, 0x4C, 0x24, 0x30, // movups xmm1, [rsp + 0x30]
            0x48, 0x89, 0xEC, // mov rsp, rbp
            0x5D, // pop rbp
            0x41, 0x5B, 0x41, 0x5A, 0x41, 0x59, 0x41, 0x58, // pop r11..r8
            0x5A, 0x59, 0x58, // pop rdx, rcx, rax
            0x9D, // popfq
        ]);
        match post_mode {
            CallThunkPost::Monitorx => {
                // Reproduce overwritten `xor ecx, ecx; xor edx, edx`.
                self.extend(&[0x31, 0xC9, 0x31, 0xD2]);
            }
            CallThunkPost::Mwaitx => {
                // Reproduce overwritten `mov ecx, 2; xor eax, eax`.
                self.extend(&[0xB9, 0x02, 0x00, 0x00, 0x00, 0x31, 0xC0]);
            }
        }
        self.emit_jmp(return_va)?;
        Ok(entry)
    }
}

#[derive(Clone, Copy)]
enum CallThunkPost {
    Monitorx,
    Mwaitx,
}

#[derive(Clone, Copy, Debug)]
enum PlannedSite {
    ClzeroLoop { patch_va: usize, end_reg: u8 },
    ClzeroAltLoop { patch_va: usize, cur_reg: u8, end_reg: u8 },
    Monitorx { patch_va: usize },
    Mwaitx { patch_va: usize },
}

/// Summary of load-time binary fast-path patching.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PatchSummary {
    pub clzero_loops: usize,
    pub clzero_alt_loops: usize,
    pub monitorx_sites: usize,
    pub mwaitx_sites: usize,
    pub slabs_allocated: usize,
}

unsafe fn alloc_nearby_slab(near_va: usize) -> Option<SlabRecord> {
    // Probe within +/- 1.5 GiB in 64 KiB steps so `E9 rel32` always reaches cleanly.
    let step = 0x10_0000usize; // 1 MiB stride
    let max_offset = 0x5000_0000usize; // 1.25 GiB < 2 GiB signed range
    let mut attempts = 0usize;
    let mut mem = std::ptr::null_mut();

    for dir in [1isize, -1isize] {
        let mut off = 0x100_0000usize;
        while off <= max_offset && attempts < 1024 {
            attempts += 1;
            let cand = if dir > 0 {
                near_va.saturating_add(off)
            } else {
                near_va.saturating_sub(off)
            };
            if cand < 0x1000_0000 {
                break;
            }
            let ptr = unsafe {
                VirtualAlloc(
                    cand as *const c_void,
                    2 * PAGE_SIZE,
                    MEM_RESERVE | MEM_COMMIT,
                    PAGE_READWRITE,
                )
            };
            if !ptr.is_null() && (ptr as usize).abs_diff(near_va) < 0x7000_0000 {
                mem = ptr;
                break;
            }
            if !ptr.is_null() {
                unsafe {
                    let _ = windows_sys::Win32::System::Memory::VirtualFree(
                        ptr,
                        0,
                        windows_sys::Win32::System::Memory::MEM_RELEASE,
                    );
                }
            }
            off += step;
        }
        if !mem.is_null() {
            break;
        }
    }

    if mem.is_null() {
        return None;
    }

    let code_page = mem as usize;
    let data_page = code_page + PAGE_SIZE;
    // SAFETY: `data_page` is committed RW zeroed page sized >= `SlabData`.
    unsafe {
        std::ptr::write(
            data_page as *mut SlabData,
            SlabData {
                fast_clzero_calls: AtomicU64::new(0),
                fast_clzero_bytes: AtomicU64::new(0),
                on_monitorx_fn: on_monitorx as *const () as usize,
                on_mwaitx_fn: on_mwaitx as *const () as usize,
            },
        );
    }
    Some(SlabRecord {
        code_page,
        data_page,
    })
}

/// Find standard/alt `clzero` loops and `monitorx`/`mwaitx` windows in `code_bytes`
/// anchored at `section_va`. Pure read-only scanner.
fn find_sites_in_buffer(section_va: usize, sec: &[u8]) -> Vec<PlannedSite> {
    let mut sites = Vec::new();
    let mut i = 0usize;
    while i < sec.len() {
        // 1. Check 15-byte ClzeroAltLoop starting at `mov rax, cur_reg`.
        if i + 15 <= sec.len()
            && let Some((cur_reg, end_reg)) = decode::decode_alt_clzero_loop(&sec[i..i + 15])
        {
            sites.push(PlannedSite::ClzeroAltLoop {
                patch_va: section_va + i,
                cur_reg,
                end_reg,
            });
            i += 15;
            continue;
        }

        // 2. Check 12-byte standard ClzeroLoop starting at `0F 01 FC` (`clzero`).
        if i + 12 <= sec.len()
            && sec[i..i + 3] == [0x0F, 0x01, 0xFC]
            && !(i > 0 && sec[i - 1] == 0x67)
            && sec[i + 3..i + 7] == [0x48, 0x83, 0xC0, 0x40]
            && (sec[i + 7] == 0x48 || sec[i + 7] == 0x4C)
            && sec[i + 8] == 0x39
            && (sec[i + 9] >> 6) == 3
            && (sec[i + 9] & 7) == 0
            && (sec[i + 7] & 1) == 0
            && sec[i + 10..i + 12] == [0x72, 0xF4]
        {
            let modrm = sec[i + 9];
            let end_reg = ((modrm >> 3) & 7) | if sec[i + 7] == 0x4C { 8 } else { 0 };
            sites.push(PlannedSite::ClzeroLoop {
                patch_va: section_va + i,
                end_reg,
            });
            i += 12;
            continue;
        }

        // 3. Verified MONITORX pair pattern: `xor ecx,ecx; xor edx,edx; monitorx; mov ..., [...]`
        if i + 10 <= sec.len()
            && sec[i..i + 7] == [0x31, 0xC9, 0x31, 0xD2, 0x0F, 0x01, 0xFA]
            && sec[i + 7..i + 9] == [0x48, 0x8B]
        {
            sites.push(PlannedSite::Monitorx {
                patch_va: section_va + i,
            });
            i += 7;
            continue;
        }

        // 4. Verified MWAITX pair pattern: `mov ecx, 2; xor eax, eax; mwaitx; mov ..., [...]`
        if i + 13 <= sec.len()
            && sec[i..i + 10]
                == [0xB9, 0x02, 0x00, 0x00, 0x00, 0x31, 0xC0, 0x0F, 0x01, 0xFB]
            && sec[i + 10..i + 12] == [0x48, 0x8B]
        {
            sites.push(PlannedSite::Mwaitx {
                patch_va: section_va + i,
            });
            i += 10;
            continue;
        }

        i += 1;
    }
    sites
}

unsafe fn emit_and_patch_sites(near_va: usize, sites: &[PlannedSite]) -> PatchSummary {
    let mut summary = PatchSummary::default();
    if sites.is_empty() {
        return summary;
    }

    let Some(slab) = (unsafe { alloc_nearby_slab(near_va) }) else {
        tracing::warn!(near_va = format_args!("{near_va:#x}"), "failed to allocate nearby fastpath slab");
        return summary;
    };

    let mut emitter = CodeEmitter::new(slab.code_page, slab.data_page);
    let mut patches: Vec<(usize, usize, usize)> = Vec::new(); // (patch_va, patch_len, stub_va)

    for &site in sites {
        match site {
            PlannedSite::ClzeroLoop { patch_va, end_reg } => {
                if let Ok(stub) = emitter.emit_clzero_loop_stub(end_reg, patch_va + 12) {
                    patches.push((patch_va, 12, stub));
                    summary.clzero_loops += 1;
                }
            }
            PlannedSite::ClzeroAltLoop {
                patch_va,
                cur_reg,
                end_reg,
            } => {
                if let Ok(stub) =
                    emitter.emit_clzero_alt_loop_stub(cur_reg, end_reg, patch_va + 15)
                {
                    patches.push((patch_va, 15, stub));
                    summary.clzero_alt_loops += 1;
                }
            }
            PlannedSite::Monitorx { patch_va } => {
                if let Ok(stub) = emitter.emit_call_thunk(
                    std::mem::offset_of!(SlabData, on_monitorx_fn),
                    CallThunkPost::Monitorx,
                    patch_va + 7,
                ) {
                    patches.push((patch_va, 7, stub));
                    summary.monitorx_sites += 1;
                }
            }
            PlannedSite::Mwaitx { patch_va } => {
                if let Ok(stub) = emitter.emit_call_thunk(
                    std::mem::offset_of!(SlabData, on_mwaitx_fn),
                    CallThunkPost::Mwaitx,
                    patch_va + 10,
                ) {
                    patches.push((patch_va, 10, stub));
                    summary.mwaitx_sites += 1;
                }
            }
        }
    }

    if emitter.bytes.len() > PAGE_SIZE {
        tracing::warn!(bytes = emitter.bytes.len(), "fastpath stubs exceeded 4 KiB slab page");
        return PatchSummary::default();
    }

    // Strict W^X: copy stub bytes while RW, flush, and transition slab code page -> RX!
    unsafe {
        std::ptr::copy_nonoverlapping(
            emitter.bytes.as_ptr(),
            slab.code_page as *mut u8,
            emitter.bytes.len(),
        );
        let mut old = 0u32;
        if VirtualProtect(
            slab.code_page as *const c_void,
            PAGE_SIZE,
            PAGE_EXECUTE_READ,
            &mut old,
        ) == 0
        {
            return PatchSummary::default();
        }
        FlushInstructionCache(
            GetCurrentProcess(),
            slab.code_page as *const c_void,
            emitter.bytes.len(),
        );
    }

    // Now overwrite each matched site cleanly (`E9 rel32` + NOP pad) and restore RX.
    let proc_handle = unsafe { GetCurrentProcess() };
    for (patch_va, patch_len, stub_va) in patches {
        let disp = stub_va as i128 - (patch_va + 5) as i128;
        let Ok(rel32) = i32::try_from(disp) else {
            continue;
        };
        let mut buf = vec![0x90u8; patch_len];
        buf[0] = 0xE9;
        buf[1..5].copy_from_slice(&rel32.to_le_bytes());

        let mut old = 0u32;
        unsafe {
            if VirtualProtect(
                patch_va as *const c_void,
                patch_len,
                PAGE_EXECUTE_READWRITE,
                &mut old,
            ) != 0
            {
                std::ptr::copy_nonoverlapping(buf.as_ptr(), patch_va as *mut u8, patch_len);
                let _ = VirtualProtect(patch_va as *const c_void, patch_len, old, &mut old);
                FlushInstructionCache(proc_handle, patch_va as *const c_void, patch_len);
            }
        }
    }

    let mut slabs = SLABS.lock().unwrap_or_else(|e| e.into_inner());
    slabs.push(SlabRecord {
        code_page: slab.code_page,
        data_page: slab.data_page,
    });
    summary.slabs_allocated = 1;

    PATCHED_CLZERO_LOOPS.fetch_add(summary.clzero_loops as u64, Ordering::Relaxed);
    PATCHED_CLZERO_ALT_LOOPS.fetch_add(summary.clzero_alt_loops as u64, Ordering::Relaxed);
    PATCHED_MONITORX.fetch_add(summary.monitorx_sites as u64, Ordering::Relaxed);
    PATCHED_MWAITX.fetch_add(summary.mwaitx_sites as u64, Ordering::Relaxed);

    summary
}

/// Scan executable PE sections of mapped image `base` before any title code runs
/// and apply native W^X fast-path patches when `VICTORY_REA_NATIVE_CPU_FASTPATH=1`.
///
/// # Safety
/// `base` must point to a mapped PE32+ image whose `.text` / executable sections
/// are not yet concurrently executing on title threads.
pub(crate) unsafe fn patch_image(base: usize) -> PatchSummary {
    if !enabled() || base == 0 {
        return PatchSummary::default();
    }
    let exec_ranges = unsafe { pe_executable_sections(base) };
    let mut all_sites = Vec::new();
    for (sec_va, sec_len) in exec_ranges {
        // SAFETY: section mapped inside loaded PE image before execution starts.
        let bytes = unsafe { std::slice::from_raw_parts(sec_va as *const u8, sec_len) };
        all_sites.extend(find_sites_in_buffer(sec_va, bytes));
    }
    if all_sites.is_empty() {
        return PatchSummary::default();
    }
    let summary = unsafe { emit_and_patch_sites(base, &all_sites) };
    tracing::info!(
        clzero_loops = summary.clzero_loops,
        clzero_alt_loops = summary.clzero_alt_loops,
        monitorx = summary.monitorx_sites,
        mwaitx = summary.mwaitx_sites,
        slabs = summary.slabs_allocated,
        "REA-002 load-time native CPU fastpath patches installed"
    );
    summary
}

unsafe fn pe_executable_sections(base: usize) -> Vec<(usize, usize)> {
    const IMAGE_SCN_MEM_EXECUTE: u32 = 0x2000_0000;
    let mut out = Vec::new();
    unsafe {
        if (base as *const u16).read_unaligned() != 0x5A4D {
            return out;
        }
        let lfanew = (base as *const u8).add(0x3C).cast::<u32>().read_unaligned() as usize;
        let nt = base + lfanew;
        if (nt as *const u32).read_unaligned() != 0x0000_4550 {
            return out;
        }
        let num_sections = (nt + 6) as *const u16;
        let opt_size = (nt + 20) as *const u16;
        let nsec = num_sections.read_unaligned() as usize;
        let first_sec = nt + 24 + opt_size.read_unaligned() as usize;
        for i in 0..nsec {
            let sh = first_sec + i * 40;
            let vsize = (sh + 8) as *const u32;
            let rva = (sh + 12) as *const u32;
            let raw_size = (sh + 16) as *const u32;
            let chars = (sh + 36) as *const u32;
            if (chars.read_unaligned() & IMAGE_SCN_MEM_EXECUTE) != 0 {
                let len = vsize.read_unaligned().min(raw_size.read_unaligned()) as usize;
                let va = base + rva.read_unaligned() as usize;
                if len > 0 {
                    out.push((va, len));
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_flag_is_off() {
        assert!(!enabled());
    }

    #[test]
    fn executable_native_clzero_and_alt_loop_trampolines_match_semantics() {
        // Synthetic executable code region in W^X memory verifying BOTH:
        // 1) standard ClzeroLoop (`clzero; add rax, 0x40; cmp rax, r10; jb -12`),
        // 2) alt ClzeroAltLoop (`mov rax, r10; add r10, 0x40; clzero; cmp r10, r11; jb -15`),
        // executed natively as real machine code without any #UD exceptions.
        let code_buf = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                PAGE_SIZE,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_EXECUTE_READWRITE,
            )
        };
        assert!(!code_buf.is_null());
        let code_va = code_buf as usize;

        // Function 1 (`offset 0`): fn(start: rcx, end: rdx) -> end_rax in rax
        //   mov rax, rcx        (48 89 C8)
        //   mov r10, rdx        (49 89 D2)
        //   clzero              (0F 01 FC)  <-- offset +6 (12 bytes)
        //   add rax, 0x40       (48 83 C0 40)
        //   cmp rax, r10        (4C 39 D0)
        //   jb -12              (72 F4)
        //   ret                 (C3)
        let fn1: [u8; 19] = [
            0x48, 0x89, 0xC8, 0x49, 0x89, 0xD2, 0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x4C,
            0x39, 0xD0, 0x72, 0xF4, 0xC3,
        ];

        // Function 2 (`offset 0x40`): fn(start: rcx, end: rdx) -> new_cur_in_rax
        //   mov r10, rcx        (49 89 CA)
        //   mov r11, rdx        (49 89 D3)
        //   mov rax, r10        (4C 89 D0)  <-- offset +6 (15 bytes)
        //   add r10, 0x40       (49 83 C2 40)
        //   clzero              (0F 01 FC)
        //   cmp r10, r11        (4D 39 DA)
        //   jb -15              (72 F1)
        //   mov rax, r10        (4C 89 D0)  <-- check cur_reg == end!
        //   ret                 (C3)
        let fn2: [u8; 25] = [
            0x49, 0x89, 0xCA, 0x49, 0x89, 0xD3, 0x4C, 0x89, 0xD0, 0x49, 0x83, 0xC2, 0x40, 0x0F,
            0x01, 0xFC, 0x4D, 0x39, 0xDA, 0x72, 0xF1, 0x4C, 0x89, 0xD0, 0xC3,
        ];

        unsafe {
            std::ptr::copy_nonoverlapping(fn1.as_ptr(), code_buf.cast::<u8>(), fn1.len());
            std::ptr::copy_nonoverlapping(fn2.as_ptr(), code_buf.cast::<u8>().add(0x40), fn2.len());
        }

        let sec_slice = unsafe { std::slice::from_raw_parts(code_buf.cast::<u8>(), 0x100) };
        let sites = find_sites_in_buffer(code_va, sec_slice);
        assert_eq!(sites.len(), 2, "must discover both standard and alt clzero loop sites");

        let summary = unsafe { emit_and_patch_sites(code_va, &sites) };
        assert_eq!(summary.clzero_loops, 1);
        assert_eq!(summary.clzero_alt_loops, 1);

        // Lock code page RX just like production code does.
        let mut old = 0u32;
        unsafe {
            assert_ne!(
                VirtualProtect(code_buf, PAGE_SIZE, PAGE_EXECUTE_READ, &mut old),
                0
            );
            FlushInstructionCache(GetCurrentProcess(), code_buf, PAGE_SIZE);
        }

        let exec1: unsafe extern "C" fn(usize, usize) -> usize =
            unsafe { std::mem::transmute(code_va) };
        let exec2: unsafe extern "C" fn(usize, usize) -> usize =
            unsafe { std::mem::transmute(code_va + 0x40) };

        let before_stats = console_insn_stats();
        let mut expected_bytes = 0u64;
        for &size in &[64usize, 128, 4096, 65536] {
            let mut mem1 = vec![0xAAu8; size + 128];
            let p1 = mem1.as_mut_ptr() as usize + 64;
            let ret1 = unsafe { exec1(p1, p1 + size) };
            assert_eq!(ret1, p1 + size);
            assert!(mem1[..64].iter().all(|&b| b == 0xAA));
            assert!(mem1[64..64 + size].iter().all(|&b| b == 0));
            assert!(mem1[64 + size..].iter().all(|&b| b == 0xAA));

            let mut mem2 = vec![0x55u8; size + 128];
            let p2 = mem2.as_mut_ptr() as usize + 64;
            let ret2 = unsafe { exec2(p2, p2 + size) };
            assert_eq!(ret2, p2 + size);
            assert!(mem2[..64].iter().all(|&b| b == 0x55));
            assert!(mem2[64..64 + size].iter().all(|&b| b == 0));
            assert!(mem2[64 + size..].iter().all(|&b| b == 0x55));

            expected_bytes += (size * 2) as u64;
        }
        // Edge case size = 0 (end == start): must touch zero memory.
        let mut canary = [0x77u8; 64];
        let pc = canary.as_mut_ptr() as usize;
        assert_eq!(unsafe { exec1(pc, pc) }, pc);
        assert!(canary.iter().all(|&b| b == 0x77));

        let after_stats = console_insn_stats();
        assert_eq!(after_stats.emu_total, before_stats.emu_total, "zero #UD exceptions");
        assert!(after_stats.fast_clzero_bytes - before_stats.fast_clzero_bytes >= expected_bytes);
        assert!(after_stats.fast_clzero_calls - before_stats.fast_clzero_calls >= 8);
    }

    #[test]
    fn monitorx_mwaitx_wait_on_address_wakes_on_cross_thread_store() {
        use std::sync::atomic::AtomicU64;
        let slot = std::sync::Arc::new(AtomicU64::new(0x1111_2222_3333_4444));
        let slot2 = slot.clone();
        let handle = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(2));
            slot2.store(0xDEAD_BEEF_CAFE_BABE, Ordering::Release);
            unsafe {
                if let Some(kb) = GetModuleHandleA(c"kernelbase.dll".as_ptr().cast()).as_ref() {
                    let h = kb as *const _ as _;
                    if let Some(wake) = GetProcAddress(h, c"WakeByAddressAll".as_ptr().cast()) {
                        let f: unsafe extern "system" fn(*const c_void) = std::mem::transmute(wake);
                        f(slot2.as_ptr().cast());
                    }
                }
            }
        });

        let before = console_insn_stats();
        probe_wait_on_address_pair(slot.as_ptr());
        handle.join().unwrap();
        let after = console_insn_stats();
        assert!(after.fast_monitorx_calls > before.fast_monitorx_calls);
        assert!(after.fast_mwaitx_calls > before.fast_mwaitx_calls);
        assert!(after.wait_on_address_calls > before.wait_on_address_calls);
    }

    #[test]
    fn executable_native_monitorx_mwaitx_thunks_preserve_registers() {
        let code_buf = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                PAGE_SIZE,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_EXECUTE_READWRITE,
            )
        };
        assert!(!code_buf.is_null());
        let code_va = code_buf as usize;

        // Synthesize `extern "C" fn(watched_ptr: rcx) -> u64`:
        //   push rdi              (57)
        //   mov rdi, rcx          (48 89 CF)
        //   mov rax, rdi          (48 89 F8)
        //   xor ecx, ecx          (31 C9)    <-- MONITORX 7B window + post `48 8B 17`
        //   xor edx, edx          (31 D2)
        //   monitorx              (0F 01 FA)
        //   mov rdx, [rdi]        (48 8B 17)
        //   mov ecx, 2            (B9 02 00 00 00) <-- MWAITX 10B window + post `48 8B 07`
        //   xor eax, eax          (31 C0)
        //   mwaitx                (0F 01 FB)
        //   mov rax, [rdi]        (48 8B 07)
        //   pop rdi               (5F)
        //   ret                   (C3)
        let func_bytes: [u8; 32] = [
            0x57, 0x48, 0x89, 0xCF, 0x48, 0x89, 0xF8,
            0x31, 0xC9, 0x31, 0xD2, 0x0F, 0x01, 0xFA,
            0x48, 0x8B, 0x17,
            0xB9, 0x02, 0x00, 0x00, 0x00, 0x31, 0xC0, 0x0F, 0x01, 0xFB,
            0x48, 0x8B, 0x07,
            0x5F, 0xC3,
        ];
        unsafe {
            std::ptr::copy_nonoverlapping(func_bytes.as_ptr(), code_buf.cast::<u8>(), func_bytes.len());
        }
        let sec_slice = unsafe { std::slice::from_raw_parts(code_buf.cast::<u8>(), func_bytes.len()) };
        let sites = find_sites_in_buffer(code_va, sec_slice);
        assert_eq!(sites.len(), 2, "must discover monitorx and mwaitx patch windows");
        let summary = unsafe { emit_and_patch_sites(code_va, &sites) };
        assert_eq!(summary.monitorx_sites, 1);
        assert_eq!(summary.mwaitx_sites, 1);

        let mut old = 0u32;
        unsafe {
            assert_ne!(VirtualProtect(code_buf, PAGE_SIZE, PAGE_EXECUTE_READ, &mut old), 0);
            FlushInstructionCache(GetCurrentProcess(), code_buf, PAGE_SIZE);
        }

        let exec: unsafe extern "C" fn(*const u64) -> u64 = unsafe { std::mem::transmute(code_va) };
        let watched = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0xABCD_1234_5678_90EF));
        let before = console_insn_stats();
        let read = unsafe { exec(watched.as_ptr()) };
        assert_eq!(read, 0xABCD_1234_5678_90EF);
        let after = console_insn_stats();
        assert_eq!(after.emu_total, before.emu_total, "zero #UD exceptions on monitorx/mwaitx");
        assert!(after.fast_monitorx_calls > before.fast_monitorx_calls);
        assert!(after.fast_mwaitx_calls > before.fast_mwaitx_calls);
    }
}
