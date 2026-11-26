//! REA-003 Frente B shader semantic recovery helpers (`VICTORY_REA003_SHADER_RECOVERY=1`).
//!
//! Kept as a submodule of [`super`] so helper logic stays isolated from the
//! core emitter and keeps `lower.rs` well below the asset-policy file size cap.

use std::collections::{HashMap, HashSet};

use super::super::bitcode::{Attr, Bin, V};
use super::{Lower, Val, Why};
use crate::gfx10::{Fields, NULL, Operand};
use crate::ir::{Inst, Program, Reg, Value};
use crate::native::{ResourceType, RootType};
use crate::resource::{Address, Site, Source};
use crate::structure::{LaneExit, Structured};

/// Whether `i` is a `v_cndmask_b32` reading an untracked constant/scalar lane mask
/// under `RecoveryFamily::ConstMask`.
pub(super) fn is_const_mask_cndmask(st: &Structured, i: &Inst) -> bool {
    super::super::recover_family_allowed(super::super::RecoveryFamily::ConstMask)
        && i.name() == "v_cndmask_b32"
        && matches!(
            &i.fields,
            Fields::Valu(v)
                if matches!(
                    v.src.get(2),
                    Some(&Operand::Scalar { code, .. })
                        if code != NULL
                            && i.read(Reg::S(code))
                                .is_some_and(|m| !st.masks.contains_key(&m))
                )
        )
}

/// Checks whether `root` is structurally uniform across active console waves
/// (built only from scalar inputs, constants, uniform-predecessor `Phi`s, uniform
/// unguarded `ds_read_b32` without divergent LDS writes, and non-lane-conditional
/// VALU instructions in `all`-lane blocks).
pub(in super::super) fn is_wave_uniform_val(
    p: &Program,
    st: &Structured,
    all: Option<&[bool]>,
    root: Value,
) -> bool {
    let is_all = |b: usize| all.is_none_or(|a| a.get(b).copied().unwrap_or(false));
    let has_divergent_lds_write = p.blocks.iter().enumerate().any(|(b, block)| {
        !is_all(b)
            && block.insts.iter().any(|i| {
                let n = i.name();
                matches!(i.fields, Fields::Ds { .. })
                    && !n.starts_with("ds_read")
                    && !n.starts_with("ds_swizzle")
                    && !n.contains("permute")
            })
    });
    let mut def_inst: HashMap<Value, (usize, usize)> = HashMap::new();
    let mut def_phi: HashMap<Value, (usize, &crate::ir::Phi)> = HashMap::new();
    for (b, block) in p.blocks.iter().enumerate() {
        for phi in &block.phis {
            def_phi.insert(phi.value, (b, phi));
        }
        for (i, inst) in block.insts.iter().enumerate() {
            for &(_, w) in &inst.writes {
                def_inst.insert(w, (b, i));
            }
        }
    }
    let mut visited: HashSet<Value> = HashSet::new();
    let mut work: Vec<Value> = vec![root];
    while let Some(v) = work.pop() {
        if !visited.insert(v) {
            continue;
        }
        if visited.len() > 96 {
            return false;
        }
        if st.masks.contains_key(&v) || st.highs.contains_key(&v) {
            return false;
        }
        if let Some(&(r, _)) = p.inputs.iter().find(|x| x.1 == v) {
            match r {
                Reg::S(_) | Reg::Scc => continue,
                Reg::V(_) => return false,
            }
        }
        if let Some(&(phi_b, phi)) = def_phi.get(&v) {
            let preds = &p.blocks[phi_b].preds;
            if preds.is_empty() || !preds.iter().all(|&q| is_all(q)) {
                return false;
            }
            work.extend(phi.args.iter().copied());
            continue;
        }
        let Some(&(b, idx)) = def_inst.get(&v) else {
            return false;
        };
        let inst = &p.blocks[b].insts[idx];
        let name = inst.name();
        match &inst.fields {
            Fields::Sop { .. } => {
                for &(_, rval) in &inst.reads {
                    work.push(rval);
                }
            }
            Fields::Smem { .. } if !name.contains("store") => {
                for &(r, rval) in &inst.reads {
                    if !matches!(r, Reg::Scc) {
                        work.push(rval);
                    }
                }
            }
            Fields::Valu(vu)
                if is_all(b)
                    && vu.dpp.is_none()
                    && vu.sdwa.is_none()
                    && !name.starts_with("v_cndmask")
                    && !name.starts_with("v_add_co")
                    && !name.starts_with("v_sub_co")
                    && !name.starts_with("v_subrev_co")
                    && !name.starts_with("v_cmp")
                    && !name.starts_with("v_readlane")
                    && !name.starts_with("v_writelane")
                    && !name.starts_with("v_permlane")
                    && !name.starts_with("v_mbcnt")
                    && (!name.contains("lane") || name == "v_readfirstlane_b32") =>
            {
                for op in &vu.src {
                    match *op {
                        Operand::Vector { reg, .. } => {
                            let Some(rval) = inst.read(Reg::V(reg)) else {
                                return false;
                            };
                            work.push(rval);
                        }
                        Operand::Const(253) => {
                            let Some(rval) = inst.read(Reg::Scc) else {
                                return false;
                            };
                            work.push(rval);
                        }
                        Operand::Scalar { code: NULL, .. }
                        | Operand::Const(_)
                        | Operand::Literal(_) => {}
                        Operand::Scalar { code, .. } => {
                            let Some(rval) = inst.read(Reg::S(code)) else {
                                return false;
                            };
                            work.push(rval);
                        }
                        Operand::Attr { .. } => return false,
                    }
                }
            }
            Fields::Ds { addr, .. }
                if name == "ds_read_b32" && is_all(b) && !has_divergent_lds_write =>
            {
                let Some(Operand::Vector { reg, .. }) = *addr else {
                    return false;
                };
                let Some(rval) = inst.read(Reg::V(reg)) else {
                    return false;
                };
                work.push(rval);
            }
            _ => return false,
        }
    }
    true
}

/// Computes `lds_meetings` when uniform (`all[b]`) blocks contain explicit
/// `s_barrier` instructions, strictly rejecting any shader that places
/// `s_barrier` inside a divergent (`!all[b]`) block.
pub(in super::super) fn lds_meetings_with_barriers(
    p: &Program,
    st: &Structured,
    all: &[bool],
) -> Option<Vec<(bool, bool)>> {
    let n = st.exits.len();
    if (0..n).any(|b| !all[b] && p.blocks[b].insts.iter().any(|i| i.name() == "s_barrier")) {
        return None;
    }
    let succ = |b: usize| -> Vec<usize> {
        match st.exits[b] {
            LaneExit::Jump(t) => vec![t],
            LaneExit::Branch {
                taken, not_taken, ..
            } => vec![taken, not_taken],
            LaneExit::End | LaneExit::Indirect => Vec::new(),
        }
    };
    let mut entry_node = vec![0usize; n];
    let mut exit_node = vec![0usize; n];
    let mut phase_count = 0usize;
    for b in 0..n {
        if all[b] {
            continue;
        }
        entry_node[b] = phase_count;
        exit_node[b] = phase_count;
        phase_count += 1;
    }
    let mut region: Vec<usize> = (0..phase_count).collect();
    fn find_p(region: &mut [usize], mut x: usize) -> usize {
        while region[x] != x {
            region[x] = region[region[x]];
            x = region[x];
        }
        x
    }
    for b in 0..n {
        if all[b] {
            continue;
        }
        for t in succ(b) {
            if !all[t] {
                let (x, y) = (
                    find_p(&mut region, exit_node[b]),
                    find_p(&mut region, entry_node[t]),
                );
                region[x] = y;
            }
        }
    }
    let mut reads = vec![false; phase_count];
    let mut writes = vec![false; phase_count];
    for (b, block) in p.blocks.iter().enumerate().take(n) {
        if all[b] {
            continue;
        }
        let r = find_p(&mut region, entry_node[b]);
        for i in &block.insts {
            let name = i.name();
            if !matches!(i.fields, Fields::Ds { .. })
                || name.starts_with("ds_swizzle")
                || name.contains("permute")
            {
                continue;
            }
            if name.starts_with("ds_read") {
                reads[r] = true;
            } else if name.starts_with("ds_write") {
                writes[r] = true;
            } else {
                reads[r] = true;
                writes[r] = true;
            }
        }
    }
    if (0..phase_count).any(|r| reads[r] && writes[r]) {
        return None;
    }
    let mut meet = vec![(false, false); n];
    for b in 0..n {
        for t in succ(b) {
            if all[b] && !all[t] {
                let r = find_p(&mut region, entry_node[t]);
                meet[b].1 |= reads[r] || writes[r];
            } else if !all[b] && all[t] {
                let r = find_p(&mut region, exit_node[b]);
                meet[t].0 |= reads[r] || writes[r];
            }
        }
    }
    Some(meet)
}

impl<'a> Lower<'a> {
    /// Queries DXIL `getDimensions` for an image handle `h` and packs RDNA2
    /// `SQ_IMG_RSRC` dimension/mip words `(1, d1), (2, d2), (3, d3), (4, depth)`.
    pub(super) fn pack_image_rsrc_words(&mut self, h: V, kind: u32, uav: bool) -> Vec<(u32, V)> {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
        let mip = if uav {
            self.s.m.undef(i32t)
        } else {
            self.s.m.i32(0)
        };
        let d = self
            .s
            .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, mip]);
        let field = |l: &mut Self, k: u32| l.s.m.extract(c, d, k);
        let one = self.s.m.i32(1);
        let w = field(self, 0);
        let (height, depth) = match kind {
            1 => (one, one),
            2 => (field(self, 1), one),
            6 => (one, field(self, 1)),
            _ => (field(self, 1), field(self, 2)),
        };
        let levels = if uav { one } else { field(self, 3) };
        let less = |l: &mut Self, x: V| l.s.m.bin(c, Bin::Sub, x, one);
        let (w, height, depth, levels) = (
            less(self, w),
            less(self, height),
            less(self, depth),
            less(self, levels),
        );
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        let (k30, k2, k14, k16, mask) = (
            k(self, 30),
            k(self, 2),
            k(self, 14),
            k(self, 16),
            k(self, 0x3FFF),
        );
        let d1 = self.s.m.bin(c, Bin::Shl, w, k30);
        let hi = self.s.m.bin(c, Bin::LShr, w, k2);
        let hi = self.s.m.bin(c, Bin::And, hi, mask);
        let ht = self.s.m.bin(c, Bin::Shl, height, k14);
        let d2 = self.s.m.bin(c, Bin::Or, hi, ht);
        let ty = [0, 8, 9, 0, 10, 0, 12, 13][kind as usize];
        let base = k(self, 0xFAC | ty << 28);
        let last = self.s.m.bin(c, Bin::Shl, levels, k16);
        let d3 = self.s.m.bin(c, Bin::Or, base, last);
        vec![(1, d1), (2, d2), (3, d3), (4, depth)]
    }

    /// Queries DXIL `getDimensions` for a buffer handle `h` and converts byte
    /// size to element count when bound as a raw view with nonzero stride.
    pub(super) fn buffer_record_count(&mut self, h: V, props: (u32, u32), stride: u32) -> V {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
        let mip = self.s.m.undef(i32t);
        let d = self
            .s
            .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, mip]);
        let n = self.s.m.extract(c, d, 0);
        if props.0 & 0xFF == 11 && stride > 1 {
            let k = self.s.m.i32(stride);
            self.s.m.bin(c, Bin::UDiv, n, k)
        } else {
            n
        }
    }

    /// Recovers a 1..=4 dword buffer descriptor load whose `k == 2` (`num_records`)
    /// word or address is read as scalar data without any `buffer_*` consumer.
    pub(super) fn recover_unread_buffer_descriptor(
        &mut self,
        inst: &Inst,
        site: Site,
        code: u8,
        count: u32,
        other: Why,
    ) -> bool {
        if count > 4 {
            return false;
        }
        let addr = self.view_address(site);
        let records = if super::super::recover_family_allowed(super::super::RecoveryFamily::BufMeta)
        {
            let src = Source::Loaded {
                load: site,
                first: 0,
            };
            let used = self.used_kind(&src);
            if used.is_some_and(|u| matches!(u.kind, ResourceType::Srv | ResourceType::Uav)) {
                let uav = used.is_some_and(|u| u.kind == ResourceType::Uav);
                let u = if uav { 0x1000 } else { 0 };
                let stride = used.map_or(0, |u| u.stride);
                let props = if stride >= 4 && stride.is_multiple_of(4) {
                    (12 | u, stride)
                } else {
                    (11 | u, 0)
                };
                self.handle(site, &src, false, props)
                    .ok()
                    .map(|h| self.buffer_record_count(h, props, stride))
            } else {
                None
            }
        } else {
            None
        };
        if addr.is_none() && records.is_none() {
            return false;
        }
        for k in 0..count {
            let reg = Reg::S(code + k as u8);
            match (k, addr, records) {
                (0, Some((lo, _)), _) => self.write(inst, reg, lo, None),
                (1, Some((_, hi)), _) => self.write(inst, reg, hi, None),
                (2, _, Some(n)) => self.write(inst, reg, n, None),
                _ => {
                    if let Some(v) = inst.write(reg) {
                        self.vals.insert(v, Val::Missing(other));
                        self.stale.insert(v);
                    }
                }
            }
        }
        true
    }

    /// Recovers a bound 8-dword image descriptor read solely for `SQ_IMG_RSRC`
    /// dimension/mip fields when explicit table usage confirms an image SRV/UAV.
    pub(super) fn recover_unread_image_descriptor(
        &mut self,
        inst: &Inst,
        site: Site,
        code: u8,
        count: u32,
        address: Option<Address>,
        other: Why,
    ) -> Result<bool, Why> {
        let Some(a) = address else {
            return Ok(false);
        };
        if count != 8
            || !super::super::recover_family_allowed(super::super::RecoveryFamily::ImgMeta)
            || !matches!(
                self.x.native.layout.get(a.param).map(|p| p.root),
                Some(RootType::Table)
            )
        {
            return Ok(false);
        }
        let src = Source::Loaded {
            load: site,
            first: 0,
        };
        let Some(u) = self.used_kind(&src) else {
            return Ok(false);
        };
        if u.dwords != 8 || !matches!(u.kind, ResourceType::Srv | ResourceType::Uav) {
            return Ok(false);
        }
        let uav = u.kind == ResourceType::Uav;
        let u = if uav { 0x1000 } else { 0 };
        let h = self.handle(site, &src, false, (2 | u, 0x409))?;
        let words = self.pack_image_rsrc_words(h, 2, uav);
        for k in 0..count {
            let reg = Reg::S(code + k as u8);
            match words.iter().find(|w| w.0 == k) {
                Some(&(_, x)) => self.write(inst, reg, x, None),
                None => {
                    if let Some(v) = inst.write(reg) {
                        self.vals.insert(v, Val::Missing(other));
                        self.stale.insert(v);
                    }
                }
            }
        }
        Ok(true)
    }
}
