//! Instructions the console's CPU (Zen 2) has and a PC's may lack – a title raises
//! STATUS_ILLEGAL_INSTRUCTION on them on an Intel CPU before Ice Lake / Alder Lake (CLWB,
//! SHA) or on any Intel CPU (CLZERO, RDPRU, MONITORX / MWAITX and SSE4a's EXTRQ / INSERTQ,
//! which are AMD's). The exception handler (`eh`) emulates the ones decoded here and steps over
//! them.

/// A console-only instruction at the faulting address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsoleOnly {
    /// `clwb m8`: write a cache line back – nothing to do, the PC's memory is coherent
    Clwb,
    /// `clzero`: zero the 64-byte cache line at rAX
    Clzero,
    /// `clzero` loop: `clzero; add rax, 0x40; cmp rax, <reg>; jb -12` – zero `[rax..reg)`
    ClzeroLoop { end_reg: u8 },
    /// Alternate `clzero` loop (observed at `0x144e8ab4c/ab6b/ab8a/abad` in build `1.0.2.1`):
    /// `mov rax, <cur>; add <cur>, 0x40; clzero; cmp <cur>, <end>; jb -15` (15 bytes total,
    /// faulting at `clzero` +7 bytes into the loop).
    ClzeroAltLoop { cur_reg: u8, end_reg: u8 },
    /// `rdpru`: ecx = 0 MPERF / 1 APERF → edx:eax (a counter)
    Rdpru,
    /// `monitorx` / `mwaitx`: a wait the emulation skips
    Monitorx,
    Mwaitx,
    /// `rdpid r64`: TSC_AUX (the processor number) into the register (CONTEXT order: rax, rcx,
    /// rdx, rbx, rsp, rbp, rsi, rdi, r8..r15)
    Rdpid(u8),
    /// SSE4a `extrq xmm, imm8 (length), imm8 (index)` / `extrq xmm1, xmm2` (`src` Some: the
    /// length and index in xmm2's bits 5:0 and 13:8): a bit field of the low quadword, moved
    /// down and zero-extended (the high quadword zeroed). Registers are XMM numbers 0..15.
    Extrq { dst: u8, src: Option<u8>, len: u8, idx: u8 },
    /// SSE4a `insertq xmm1, xmm2, imm8 (length), imm8 (index)` / `insertq xmm1, xmm2` (`reg`:
    /// the length and index in xmm2's bits 69:64 and 77:72): the low bits of xmm2 put into a
    /// bit field of xmm1's low quadword.
    Insertq { dst: u8, src: u8, reg: bool, len: u8, idx: u8 },
    /// `ud2`: the title's own trap (not emulated)
    Ud2,
}

/// A field length of SSE4a's six bits: 0 is 64.
fn field_len(len: u8) -> u32 {
    match len & 63 {
        0 => 64,
        n => n as u32,
    }
}

/// EXTRQ on a low quadword: `len` bits from bit `idx`, zero-extended (a field past bit 63 is
/// undefined on the console; here it is what a shift gives).
pub fn extrq(v: u64, len: u8, idx: u8) -> u64 {
    let (len, idx) = (field_len(len), (idx & 63) as u32);
    let field = v.checked_shr(idx).unwrap_or(0);
    if len >= 64 {
        field
    } else {
        field & ((1u64 << len) - 1)
    }
}

/// INSERTQ on a low quadword: the low `len` bits of `src` put at bit `idx` of `dst`.
pub fn insertq(dst: u64, src: u64, len: u8, idx: u8) -> u64 {
    let (len, idx) = (field_len(len), (idx & 63) as u32);
    let ones = if len >= 64 { u64::MAX } else { (1u64 << len) - 1 };
    let mask = ones.checked_shl(idx).unwrap_or(0);
    (dst & !mask) | ((src & ones).checked_shl(idx).unwrap_or(0) & mask)
}

/// The length of a ModRM-encoded operand (ModRM, SIB and displacement) at `b[0]`, None when the
/// bytes run out.
fn modrm_len(b: &[u8]) -> Option<usize> {
    let modrm = *b.first()?;
    let (md, rm) = (modrm >> 6, modrm & 7);
    if md == 3 {
        return Some(1);
    }
    let mut len = 1;
    let mut base = rm;
    if rm == 4 {
        base = *b.get(1)? & 7;
        len += 1;
    }
    len += match md {
        0 if base == 5 => 4, // disp32 (rip-relative, or a SIB without a base)
        0 => 0,
        1 => 1,
        _ => 4,
    };
    (b.len() >= len).then_some(len)
}

/// The console-only instruction at the start of `code` and its length.
pub fn decode(code: &[u8]) -> Option<(ConsoleOnly, usize)> {
    let mut i = 0;
    // `rep`: the last of F2 / F3, which decides between them (and over 66) for SSE4a
    let (mut p66, mut pf3, mut rep, mut rex) = (false, false, 0u8, 0u8);
    while let Some(&b) = code.get(i) {
        match b {
            0x66 => p66 = true,
            0xF2 => rep = 0xF2,
            0xF3 => {
                pf3 = true;
                rep = 0xF3;
            }
            0x2E | 0x3E | 0x26 | 0x36 | 0x64 | 0x65 | 0x67 => {}
            // REX counts only right before the opcode: one before a legacy prefix is ignored
            0x40..=0x4F => {
                rex = b;
                i += 1;
                continue;
            }
            _ => break,
        }
        rex = 0;
        i += 1;
    }
    if *code.get(i)? != 0x0F {
        return None;
    }
    match (*code.get(i + 1)?, code.get(i + 2).copied()) {
        (0x01, Some(0xFC)) => {
            // clzero (0F 01 FC); add rax, 0x40 (48 83 C0 40); cmp rax, <reg> ([48|4C] 39 modrm); jb -12 (72 F4)
            // (not after an 0x67: that CLZERO is at EAX, and the jb goes back to the 0F, past the
            // prefix – it is stepped alone, the loop is taken from the next one)
            if code.len() >= i + 12
                && !code[..i].contains(&0x67)
                && code[i + 3..i + 7] == [0x48, 0x83, 0xC0, 0x40]
                && (code[i + 7] == 0x48 || code[i + 7] == 0x4C)
                && code[i + 8] == 0x39
                && code[i + 10..i + 12] == [0x72, 0xF4]
            {
                let rex_r = code[i + 7] == 0x4C;
                let modrm = code[i + 9];
                let reg_idx = ((modrm >> 3) & 7) | if rex_r { 8 } else { 0 };
                Some((ConsoleOnly::ClzeroLoop { end_reg: reg_idx }, i + 12))
            } else {
                Some((ConsoleOnly::Clzero, i + 3))
            }
        }
        (0x01, Some(0xFD)) => Some((ConsoleOnly::Rdpru, i + 3)),
        (0x01, Some(0xFA)) => Some((ConsoleOnly::Monitorx, i + 3)),
        (0x01, Some(0xFB)) => Some((ConsoleOnly::Mwaitx, i + 3)),
        (0x0B, _) => Some((ConsoleOnly::Ud2, i + 2)),
        // SSE4a: register operands only (ModRM mod 3); REX.R extends reg, REX.B rm
        // (the last of F2 / F3 decides: F2 INSERTQ, F3 #UD; neither with 66: EXTRQ)
        (0x78 | 0x79, Some(m)) if (rep == 0xF2 || (rep == 0 && p66)) && m >> 6 == 3 => {
            let reg = ((m >> 3) & 7) | (rex & 4) << 1;
            let rm = (m & 7) | (rex & 1) << 3;
            let imm = |k: usize| code.get(i + 3 + k).copied();
            match (code[i + 1], rep == 0xF2) {
                // extrq xmm(rm), imm8, imm8 – the ModRM reg field is /0
                (0x78, false) if (m >> 3) & 7 == 0 => Some((ConsoleOnly::Extrq { dst: rm, src: None, len: imm(0)?, idx: imm(1)? }, i + 5)),
                (0x79, false) => Some((ConsoleOnly::Extrq { dst: reg, src: Some(rm), len: 0, idx: 0 }, i + 3)),
                (0x78, true) => Some((ConsoleOnly::Insertq { dst: reg, src: rm, reg: false, len: imm(0)?, idx: imm(1)? }, i + 5)),
                (0x79, true) => Some((ConsoleOnly::Insertq { dst: reg, src: rm, reg: true, len: 0, idx: 0 }, i + 3)),
                _ => None,
            }
        }
        // 66 0F AE /6 with a memory operand (without 66 it is xsaveopt)
        (0xAE, Some(m)) if p66 && (m >> 3) & 7 == 6 && m >> 6 != 3 => Some((ConsoleOnly::Clwb, i + 2 + modrm_len(&code[i + 2..])?)),
        // F3 0F C7 /7 with a register operand
        (0xC7, Some(m)) if pf3 && (m >> 3) & 7 == 7 && m >> 6 == 3 => Some((ConsoleOnly::Rdpid((m & 7) | (rex & 1) << 3), i + 3)),
        _ => None,
    }
}

/// Match a 15-byte alternate `clzero` loop window starting at `mov rax, <cur_reg>`:
/// `mov rax, cur` (3B) + `add cur, 0x40` (4B) + `clzero` (3B) + `cmp cur, end` (3B) + `jb -15` (2B).
/// Returns `(cur_reg, end_reg)` on exact match.
pub fn decode_alt_clzero_loop(window: &[u8]) -> Option<(u8, u8)> {
    if window.len() < 15 {
        return None;
    }
    // 1. `mov rax, cur_reg`: `[48|4C] 89 <modrm>` with mod=3, rm=0 (`rax`).
    let rex_mov = window[0];
    if (rex_mov != 0x48 && rex_mov != 0x4C) || window[1] != 0x89 {
        return None;
    }
    let m_mov = window[2];
    if (m_mov >> 6) != 3 || (m_mov & 7) != 0 {
        return None;
    }
    let cur_reg = ((m_mov >> 3) & 7) | if (rex_mov & 4) != 0 { 8 } else { 0 };

    // 2. `add cur_reg, 0x40`: `[48|49] 83 <C0+cur&7> 40`.
    let expect_rex_add = if cur_reg >= 8 { 0x49 } else { 0x48 };
    if window[3] != expect_rex_add
        || window[4] != 0x83
        || window[5] != (0xC0 | (cur_reg & 7))
        || window[6] != 0x40
    {
        return None;
    }

    // 3. `clzero`: `0F 01 FC`.
    if window[7..10] != [0x0F, 0x01, 0xFC] {
        return None;
    }

    // 4. `cmp cur_reg, end_reg`: `48..4F 39 <modrm>` with W=1, X=0, rm=cur_reg.
    let rex_cmp = window[10];
    if (rex_cmp & 0xFA) != 0x48 {
        return None;
    }
    if ((rex_cmp & 1) != 0) != (cur_reg >= 8) {
        return None;
    }
    if window[11] != 0x39 {
        return None;
    }
    let m_cmp = window[12];
    if (m_cmp >> 6) != 3 || (m_cmp & 7) != (cur_reg & 7) {
        return None;
    }
    let end_reg = ((m_cmp >> 3) & 7) | if (rex_cmp & 4) != 0 { 8 } else { 0 };
    if cur_reg == end_reg {
        return None;
    }

    // 5. `jb -15`: `72 F1`.
    if window[13..15] != [0x72, 0xF1] {
        return None;
    }
    Some((cur_reg, end_reg))
}


#[cfg(test)]
mod tests {
    use super::*;

    /// EXTRQ / INSERTQ as AMD's manual gives them: a length of 0 is 64.
    #[test]
    fn sse4a_bit_fields() {
        assert_eq!(extrq(0x1234_5678_9ABC_DEF0, 8, 16), 0xBC);
        assert_eq!(extrq(0x1234_5678_9ABC_DEF0, 0, 0), 0x1234_5678_9ABC_DEF0);
        assert_eq!(extrq(0x8000_0000_0000_0000, 1, 63), 1);
        assert_eq!(insertq(0xFFFF_FFFF_FFFF_FFFF, 0xAB, 8, 8), 0xFFFF_FFFF_FFFF_ABFF);
        assert_eq!(insertq(0, 0x1FF, 4, 60), 0xF000_0000_0000_0000);
        assert_eq!(insertq(0x1111, 0xDEAD_BEEF_0000_0000, 0, 0), 0xDEAD_BEEF_0000_0000);
        for (v, len, idx) in [(0x0123_4567_89AB_CDEFu64, 12u8, 20u8), (u64::MAX, 63, 1), (0x55, 3, 0)] {
            assert_eq!(extrq(insertq(0, v, len, idx), len, idx), extrq(v, len, 0));
        }
    }

    #[test]
    fn decodes_the_console_only_instructions() {
        assert_eq!(decode(&[0x0F, 0x01, 0xFC]), Some((ConsoleOnly::Clzero, 3)));
        // clzero loop against r10 (4C 39 D0 = cmp rax, r10)
        assert_eq!(
            decode(&[0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x4C, 0x39, 0xD0, 0x72, 0xF4]),
            Some((ConsoleOnly::ClzeroLoop { end_reg: 10 }, 12))
        );
        // clzero loop against rdx (48 39 D0 = cmp rax, rdx)
        assert_eq!(
            decode(&[0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x48, 0x39, 0xD0, 0x72, 0xF4]),
            Some((ConsoleOnly::ClzeroLoop { end_reg: 2 }, 12))
        );
        // the same loop after an address-size prefix: that one CLZERO alone
        assert_eq!(decode(&[0x67, 0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x48, 0x39, 0xD0, 0x72, 0xF4]), Some((ConsoleOnly::Clzero, 4)));
        assert_eq!(decode(&[0x0F, 0x01, 0xFD]), Some((ConsoleOnly::Rdpru, 3)));
        assert_eq!(decode(&[0x0F, 0x01, 0xFA]), Some((ConsoleOnly::Monitorx, 3)));
        assert_eq!(decode(&[0x0F, 0x01, 0xFB]), Some((ConsoleOnly::Mwaitx, 3)));
        assert_eq!(decode(&[0x0F, 0x0B]), Some((ConsoleOnly::Ud2, 2)));
        // clwb [rax]; clwb [rcx+0x40]; clwb [rsp+8]; clwb [rip+disp32]; clwb [r8]
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x30]), Some((ConsoleOnly::Clwb, 4)));
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x71, 0x40]), Some((ConsoleOnly::Clwb, 5)));
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x74, 0x24, 0x08]), Some((ConsoleOnly::Clwb, 6)));
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x35, 1, 2, 3, 4]), Some((ConsoleOnly::Clwb, 8)));
        assert_eq!(decode(&[0x66, 0x41, 0x0F, 0xAE, 0x30]), Some((ConsoleOnly::Clwb, 5)));
        // rdpid rax; rdpid r9
        assert_eq!(decode(&[0xF3, 0x0F, 0xC7, 0xF8]), Some((ConsoleOnly::Rdpid(0), 4)));
        assert_eq!(decode(&[0xF3, 0x41, 0x0F, 0xC7, 0xF9]), Some((ConsoleOnly::Rdpid(9), 5)));
        // not console-only: clflushopt (/7), xsaveopt (no 66), a plain mov, truncated bytes
        // SSE4a, as a retail game has it (#216: insertq xmm0, xmm1 with length / index bytes)
        assert_eq!(decode(&[0xF2, 0x0F, 0x78, 0xC1, 0x08, 0x10]), Some((ConsoleOnly::Insertq { dst: 0, src: 1, reg: false, len: 8, idx: 16 }, 6)));
        assert_eq!(decode(&[0xF2, 0x45, 0x0F, 0x79, 0xC1]), Some((ConsoleOnly::Insertq { dst: 8, src: 9, reg: true, len: 0, idx: 0 }, 5)));
        assert_eq!(decode(&[0x66, 0x0F, 0x78, 0xC2, 0x04, 0x08]), Some((ConsoleOnly::Extrq { dst: 2, src: None, len: 4, idx: 8 }, 6)));
        assert_eq!(decode(&[0x66, 0x0F, 0x79, 0xCA]), Some((ConsoleOnly::Extrq { dst: 1, src: Some(2), len: 0, idx: 0 }, 4)));
        assert_eq!(decode(&[0xF2, 0x0F, 0x78, 0xC1, 0x08]), None, "an immediate missing");
        assert_eq!(decode(&[0x0F, 0x78, 0xC1, 0x08, 0x10]), None, "VMREAD without a prefix is not SSE4a");
        // prefixes as a Zen 5 runs them (Kalmarv on #209): the last of F2 / F3 decides, 66 next
        // to them does not; 0x67 and a REX before a legacy prefix are ignored
        let insertq = Some((ConsoleOnly::Insertq { dst: 0, src: 1, reg: false, len: 16, idx: 16 }, 7));
        for code in [[0x66, 0xF2, 0x0F, 0x78, 0xC1, 0x10, 0x10], [0xF2, 0x66, 0x0F, 0x78, 0xC1, 0x10, 0x10], [0xF3, 0xF2, 0x0F, 0x78, 0xC1, 0x10, 0x10], [0x67, 0xF2, 0x0F, 0x78, 0xC1, 0x10, 0x10], [0x41, 0xF2, 0x0F, 0x78, 0xC1, 0x10, 0x10]] {
            assert_eq!(decode(&code), insertq, "{code:02x?}");
        }
        for code in [[0xF2, 0xF3, 0x0F, 0x78, 0xC1, 0x10, 0x10], [0xF3, 0x66, 0x0F, 0x78, 0xC1, 0x10, 0x10], [0x66, 0xF3, 0x0F, 0x78, 0xC0, 0x10, 0x10]] {
            assert_eq!(decode(&code), None, "#UD on the CPU: {code:02x?}");
        }
        assert_eq!(decode(&[0x67, 0x0F, 0x01, 0xFC]), Some((ConsoleOnly::Clzero, 4)));
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x38]), None);
        assert_eq!(decode(&[0x0F, 0xAE, 0x30]), None);
        assert_eq!(decode(&[0x48, 0x8B, 0x01]), None);
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x74, 0x24]), None);
        assert_eq!(decode(&[0x0F]), None);
    }

    #[test]
    fn decodes_alt_clzero_loops_across_all_verified_register_pairs() {
        // Real 15-byte loop structures verified from build 1.0.2.1 (`0x144e8ab4c/ab6b/ab8a/abad`).
        let cases: &[([u8; 15], u8, u8)] = &[
            ([0x4C, 0x89, 0xD0, 0x49, 0x83, 0xC2, 0x40, 0x0F, 0x01, 0xFC, 0x4D, 0x39, 0xDA, 0x72, 0xF1], 10, 11), // r10, r11
            ([0x4C, 0x89, 0xC8, 0x49, 0x83, 0xC1, 0x40, 0x0F, 0x01, 0xFC, 0x4D, 0x39, 0xD1, 0x72, 0xF1], 9, 10),  // r9, r10
            ([0x4C, 0x89, 0xC0, 0x49, 0x83, 0xC0, 0x40, 0x0F, 0x01, 0xFC, 0x4D, 0x39, 0xC8, 0x72, 0xF1], 8, 9),   // r8, r9
            ([0x48, 0x89, 0xD0, 0x48, 0x83, 0xC2, 0x40, 0x0F, 0x01, 0xFC, 0x4C, 0x39, 0xC2, 0x72, 0xF1], 2, 8),   // rdx, r8
        ];
        for &(bytes, cur, end) in cases {
            assert_eq!(decode_alt_clzero_loop(&bytes), Some((cur, end)), "{bytes:02x?}");
        }
        let mut bad = cases[0].0;
        bad[14] = 0xF4; // wrong jb displacement
        assert_eq!(decode_alt_clzero_loop(&bad), None);
    }
}
