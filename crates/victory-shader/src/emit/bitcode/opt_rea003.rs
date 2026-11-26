//! Opt-in REA-003 DXIL bitcode optimizer passes (`VICTORY_REA003_SHADER_OPT=1`):
//!
//! - [`BITFIELD`]: algebraic simplification of nested masks, shift chains, and
//!   complementary `(x & m1) | (x & m2)` reassembly over integers.
//! - [`LICM`]: sound Loop-Invariant Code Motion from single-header natural loops
//!   with a unique outside unconditional preheader for pure non-trapping ops.

use super::opt::{Effect, Tree, map_operands, mask, resolve};
use super::{Bin, Module, Op, Term, V, for_each_operand};

/// Bitfield algebraic simplification pass flag.
pub const BITFIELD: u32 = 64;
/// Loop-invariant code motion pass flag.
pub const LICM: u32 = 128;

impl Module {
    /// Applies algebraic bitfield & shift-chain simplifications over reachable blocks.
    pub(super) fn bitfield_simplify(&mut self) -> bool {
        if self.blocks.is_empty() {
            return false;
        }
        let tree = self.dominator_tree();
        let mut repl: Vec<Option<V>> = vec![None; self.insts.len()];
        let mut any_changed = false;

        for _ in 0..2 {
            let mut round_changed = false;
            for &b in &tree.order {
                let body = self.blocks[b].body.clone();
                for i in body {
                    if repl[i as usize].is_some() {
                        continue;
                    }
                    let ty = self.insts[i as usize].ty;
                    let Some(bits) = self.int_bits(ty) else {
                        continue;
                    };
                    let m = mask(bits);
                    let op = self.insts[i as usize].op.clone();
                    let Op::Bin(bin, raw_a, raw_b) = op else {
                        continue;
                    };
                    let a = resolve(&repl, raw_a);
                    let b_op = resolve(&repl, raw_b);

                    if let Some(new_val) = self.fold_bitfield_to_value(bin, a, b_op, ty, bits, m) {
                        repl[i as usize] = Some(new_val);
                        round_changed = true;
                        continue;
                    }
                    if let Some(rewritten) = self.rewrite_bitfield_op(bin, a, b_op, ty, bits, m) {
                        self.insts[i as usize].op = rewritten;
                        round_changed = true;
                    }
                }
            }
            if !round_changed {
                break;
            }
            any_changed = true;
            for &b in &tree.order {
                let phis = self.blocks[b].phis.clone();
                for p in phis {
                    map_operands(&mut self.insts[p as usize].op, |v| resolve(&repl, v));
                }
                let body = self.blocks[b].body.clone();
                for i in body {
                    map_operands(&mut self.insts[i as usize].op, |v| resolve(&repl, v));
                }
                if let Some(Term::CondBr(c, t, f)) = self.blocks[b].term {
                    self.blocks[b].term = Some(Term::CondBr(resolve(&repl, c), t, f));
                }
            }
        }
        any_changed
    }

    fn as_bin_const_pair(&self, v: V) -> Option<(Bin, V, u64)> {
        let V::Inst(i) = v else { return None };
        let Op::Bin(op, x, y) = self.insts[i as usize].op else {
            return None;
        };
        if let Some((_, k)) = self.int_k(y) {
            return Some((op, x, k));
        }
        if matches!(op, Bin::And | Bin::Or | Bin::Xor)
            && let Some((_, k)) = self.int_k(x)
        {
            return Some((op, y, k));
        }
        None
    }

    fn fold_bitfield_to_value(
        &mut self,
        bin: Bin,
        a: V,
        b: V,
        ty: super::Ty,
        bits: u32,
        m: u64,
    ) -> Option<V> {
        match bin {
            Bin::And => {
                // `(x & c1) & c2`: zero, identical to inner, etc.
                for (lhs, rhs) in [(a, b), (b, a)] {
                    if let Some((_, c2)) = self.int_k(rhs)
                        && let Some((Bin::And, _, c1)) = self.as_bin_const_pair(lhs)
                    {
                        let comb = c1 & c2 & m;
                        if comb == 0 {
                            return Some(self.k(ty, bits, 0));
                        }
                        if comb == (c1 & m) {
                            return Some(lhs);
                        }
                    }
                    // `(x | c1) & c2`: if `(c1 & c2) == (c2 & m)` -> `c2`.
                    if let Some((_, c2)) = self.int_k(rhs)
                        && let Some((Bin::Or, _, c1)) = self.as_bin_const_pair(lhs)
                        && ((c1 & c2) & m) == (c2 & m)
                    {
                        return Some(self.k(ty, bits, c2 & m));
                    }
                }
                None
            }
            Bin::Or => {
                for (lhs, rhs) in [(a, b), (b, a)] {
                    if let Some((_, c2)) = self.int_k(rhs)
                        && let Some((Bin::Or, _, c1)) = self.as_bin_const_pair(lhs)
                    {
                        let comb = (c1 | c2) & m;
                        if comb == m {
                            return Some(self.k(ty, bits, m));
                        }
                        if comb == (c1 & m) {
                            return Some(lhs);
                        }
                    }
                }
                // `(x & m1) | (x & m2)` == `x & (m1 | m2)` == `x` when `(m1 | m2) & m == m`.
                if let (Some((Bin::And, x1, m1)), Some((Bin::And, x2, m2))) =
                    (self.as_bin_const_pair(a), self.as_bin_const_pair(b))
                    && x1 == x2
                    && ((m1 | m2) & m) == m
                {
                    return Some(x1);
                }
                None
            }
            Bin::Shl => {
                let (_, s) = self.int_k(b)?;
                if s < u64::from(bits)
                    && let Some((Bin::And, _, c)) = self.as_bin_const_pair(a)
                    && ((c << s) & m) == 0
                {
                    return Some(self.k(ty, bits, 0));
                }
                None
            }
            Bin::LShr => {
                let (_, s) = self.int_k(b)?;
                if s < u64::from(bits)
                    && let Some((Bin::And, _, c)) = self.as_bin_const_pair(a)
                    && ((c & m) >> s) == 0
                {
                    return Some(self.k(ty, bits, 0));
                }
                None
            }
            _ => None,
        }
    }

    fn rewrite_bitfield_op(
        &mut self,
        bin: Bin,
        a: V,
        b: V,
        ty: super::Ty,
        bits: u32,
        m: u64,
    ) -> Option<Op> {
        match bin {
            Bin::And => {
                for (lhs, rhs) in [(a, b), (b, a)] {
                    if let Some((_, c2)) = self.int_k(rhs)
                        && let Some((Bin::And, x, c1)) = self.as_bin_const_pair(lhs)
                    {
                        let comb = c1 & c2 & m;
                        if comb != (c1 & m) && comb != 0 {
                            let kc = self.k(ty, bits, comb);
                            return Some(Op::Bin(Bin::And, x, kc));
                        }
                    }
                }
                None
            }
            Bin::Or => {
                for (lhs, rhs) in [(a, b), (b, a)] {
                    if let Some((_, c2)) = self.int_k(rhs)
                        && let Some((Bin::Or, x, c1)) = self.as_bin_const_pair(lhs)
                    {
                        let comb = (c1 | c2) & m;
                        if comb != (c1 & m) && comb != m {
                            let kc = self.k(ty, bits, comb);
                            return Some(Op::Bin(Bin::Or, x, kc));
                        }
                    }
                }
                None
            }
            Bin::Shl | Bin::LShr => {
                let (_, s2) = self.int_k(b)?;
                let V::Inst(inner) = a else { return None };
                let Op::Bin(inner_op, x, k1) = self.insts[inner as usize].op else {
                    return None;
                };
                if inner_op == bin
                    && let Some((_, s1)) = self.int_k(k1)
                    && s1 < u64::from(bits)
                    && s2 < u64::from(bits)
                    && s1 + s2 < u64::from(bits)
                {
                    let total = self.k(ty, bits, s1 + s2);
                    return Some(Op::Bin(bin, x, total));
                }
                None
            }
            _ => None,
        }
    }

    /// Sound Loop-Invariant Code Motion for single-header natural loops whose header has
    /// a unique outside unconditional preheader (`Term::Br(header)`). Only hoists pure,
    /// non-trapping instructions out of `header.body` whose operands dominate `preheader`.
    pub(super) fn licm_hoist(&mut self) -> bool {
        if self.blocks.len() < 2 {
            return false;
        }
        let effects = self.effects();
        let tree = self.dominator_tree();
        let n_blocks = self.blocks.len();
        let (tin, tout) = dom_dfs_times(&tree, n_blocks);
        let dominates = |a: usize, b: usize| tin[a] != 0 && tin[a] <= tin[b] && tout[b] <= tout[a];

        let mut def_block = vec![usize::MAX; self.insts.len()];
        for (b, blk) in self.blocks.iter().enumerate() {
            for &i in blk.phis.iter().chain(&blk.body) {
                def_block[i as usize] = b;
            }
        }

        let mut changed = false;
        for &header in &tree.order {
            if header == 0 {
                continue;
            }
            // Collect backedges `tail -> header` where `header` dominates `tail`.
            let backedges: Vec<usize> = tree.preds[header]
                .iter()
                .copied()
                .filter(|&p| dominates(header, p))
                .collect();
            if backedges.is_empty() {
                continue;
            }

            // Build natural loop node set; bail if any predecessor isn't dominated by `header`.
            let mut in_loop = vec![false; n_blocks];
            in_loop[header] = true;
            let mut work = Vec::new();
            for tail in backedges {
                if !in_loop[tail] {
                    in_loop[tail] = true;
                    work.push(tail);
                }
            }
            let mut reducible = true;
            while let Some(node) = work.pop() {
                for &p in &tree.preds[node] {
                    if !dominates(header, p) {
                        reducible = false;
                        break;
                    }
                    if !in_loop[p] {
                        in_loop[p] = true;
                        work.push(p);
                    }
                }
                if !reducible {
                    break;
                }
            }
            if !reducible {
                continue;
            }

            // Require a single outside predecessor ending unconditionally with `Br(header)`.
            let outside: Vec<usize> = tree.preds[header]
                .iter()
                .copied()
                .filter(|&p| !in_loop[p])
                .collect();
            let [preheader] = outside[..] else {
                continue;
            };
            if preheader == header
                || in_loop[preheader]
                || tin[preheader] == 0
                || !matches!(self.blocks[preheader].term, Some(Term::Br(h)) if h == header)
            {
                continue;
            }

            // Hoist safe pure non-trapping instructions out of `header.body` in topological order.
            let old_body = std::mem::take(&mut self.blocks[header].body);
            let mut kept = Vec::with_capacity(old_body.len());
            for i in old_body {
                if self
                    .can_hoist_from_header(&effects, i, preheader, &in_loop, &def_block, &dominates)
                {
                    self.blocks[preheader].body.push(i);
                    def_block[i as usize] = preheader;
                    changed = true;
                } else {
                    kept.push(i);
                }
            }
            self.blocks[header].body = kept;
        }
        changed
    }

    fn can_hoist_from_header(
        &self,
        effects: &[Effect],
        i: u32,
        preheader: usize,
        in_loop: &[bool],
        def_block: &[usize],
        dominates: &impl Fn(usize, usize) -> bool,
    ) -> bool {
        let op = &self.insts[i as usize].op;
        let pure_safe = match op {
            Op::Bin(bin, _, _) => !matches!(bin, Bin::UDiv | Bin::SDiv | Bin::URem),
            Op::Cast(..) | Op::Cmp(..) | Op::Select(..) | Op::Extract(..) | Op::Gep(..) => true,
            Op::Call(f, _) => effects[*f as usize] == Effect::Pure,
            Op::Phi(..) | Op::Load(..) | Op::Store(..) | Op::AtomicRmw(..) | Op::CmpXchg(..) => {
                false
            }
        };
        if !pure_safe {
            return false;
        }
        let mut ok = true;
        for_each_operand(op, |v| {
            if let V::Inst(d) = v {
                let db = def_block[d as usize];
                if db == usize::MAX || in_loop[db] || !dominates(db, preheader) {
                    ok = false;
                }
            }
        });
        ok
    }
}

fn dom_dfs_times(tree: &Tree, n: usize) -> (Vec<u32>, Vec<u32>) {
    let mut tin = vec![0u32; n];
    let mut tout = vec![0u32; n];
    let mut tick = 0u32;
    let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
    while let Some((b, child_idx)) = stack.pop() {
        if child_idx == 0 {
            tick += 1;
            tin[b] = tick;
        }
        if let Some(&c) = tree.children[b].get(child_idx) {
            stack.push((b, child_idx + 1));
            stack.push((c, 0));
        } else {
            tick += 1;
            tout[b] = tick;
        }
    }
    (tin, tout)
}

#[cfg(test)]
mod tests {
    use super::super::opt::{ALL, ALL_REA003, DEAD};
    use super::super::{Attr, Bin, Module, Term, V};
    use super::{BITFIELD, LICM};

    trait VIndex {
        fn u32_id(self) -> u32;
        fn usize_id(self) -> usize;
    }
    impl VIndex for V {
        fn u32_id(self) -> u32 {
            match self {
                V::Inst(i) => i,
                _ => panic!("expected V::Inst"),
            }
        }
        fn usize_id(self) -> usize {
            self.u32_id() as usize
        }
    }

    fn pure_fn_kit() -> (Module, usize, super::V, super::V) {
        let mut m = Module::default();
        let i32t = m.int(32);
        let void = m.void();
        let fty = m.func_ty(i32t, vec![i32t, i32t]);
        m.ptr(fty);
        let reads = m.function("dx.op.bufferLoad.i32", fty, Some(Attr::ReadOnly), false);
        let writes = m.function("dx.op.bufferStore.i32", fty, Some(Attr::NoUnwind), false);
        let main_ty = m.func_ty(void, vec![]);
        m.ptr(main_ty);
        m.function("main", main_ty, None, true);
        let b = m.new_block();
        (m, b, reads, writes)
    }

    #[test]
    fn bitfield_folds_nested_masks_shifts_and_complements() {
        let (mut m, b, reads, writes) = pure_fn_kit();
        let zero = m.i32(0);
        let x = m.call(b, reads, vec![zero, zero]);
        let y = m.call(b, reads, vec![zero, zero]);

        // ((x & 0xFF) & 0x0F) -> (x & 0x0F)
        let m_ff = m.i32(0xFF);
        let m_0f = m.i32(0x0F);
        let a1 = m.bin(b, Bin::And, x, m_ff);
        let a2 = m.bin(b, Bin::And, a1, m_0f);

        // (y << 3) << 4 -> (y << 7)
        let s3 = m.i32(3);
        let s4 = m.i32(4);
        let sh1 = m.bin(b, Bin::Shl, y, s3);
        let sh2 = m.bin(b, Bin::Shl, sh1, s4);

        // (a2 & 0xFFFF) | (a2 & 0xFFFF_0000) -> a2
        let lo = m.i32(0x0000_FFFF);
        let hi = m.i32(0xFFFF_0000);
        let p_lo = m.bin(b, Bin::And, a2, lo);
        let p_hi = m.bin(b, Bin::And, a2, hi);
        let merged = m.bin(b, Bin::Or, p_lo, p_hi);

        let sum = m.bin(b, Bin::Add, merged, sh2);
        m.call(b, writes, vec![zero, sum]);
        m.terminate(b, Term::Ret);

        let before = m.blocks[b].body.len();
        m.optimise(BITFIELD | DEAD);
        let after = m.blocks[b].body.len();
        assert!(after < before, "expected fewer insts: {before} -> {after}");
        assert!(
            matches!(m.insts[sum.usize_id()].op, super::Op::Bin(Bin::Add, _, _)),
            "final add remains live"
        );
        m.write();
    }

    #[test]
    fn licm_hoists_pure_invariant_and_keeps_trapping_div_in_loop() {
        let (mut m, entry, reads, writes) = pure_fn_kit();
        let i32t = m.int(32);
        let zero = m.i32(0);
        let one = m.i32(1);
        let k7 = m.i32(7);
        let k3 = m.i32(3);
        let limit = m.i32(10);

        let ext = m.call(entry, reads, vec![zero, zero]);
        let div_divisor = m.call(entry, reads, vec![one, zero]);
        let pre = m.new_block();
        let header = m.new_block();
        let latch = m.new_block();
        let exit = m.new_block();

        m.terminate(entry, Term::Br(pre));
        m.terminate(pre, Term::Br(header));

        let iv = m.phi(header, i32t);
        let next_iv = m.bin(header, Bin::Add, iv, one);
        // Pure loop-invariant op inside `header` should hoist to `pre`.
        let inv_mul = m.bin(header, Bin::Mul, ext, k7);
        let inv_add = m.bin(header, Bin::Add, inv_mul, k3);
        // Trapping UDiv MUST stay in `header`, never hoisted to `pre`.
        let guarded_div = m.bin(header, Bin::UDiv, ext, div_divisor);
        let comb = m.bin(header, Bin::Xor, inv_add, guarded_div);
        m.call(header, writes, vec![iv, comb]);
        let cond = m.cmp(header, super::super::pred::IULT, next_iv, limit);
        m.terminate(header, Term::CondBr(cond, latch, exit));
        m.terminate(latch, Term::Br(header));
        m.terminate(exit, Term::Ret);
        m.set_phi(iv, vec![(zero, pre), (next_iv, latch)]);

        m.optimise(LICM);
        assert!(
            m.blocks[pre].body.contains(&inv_mul.u32_id()),
            "inv_mul should hoist to preheader"
        );
        assert!(
            m.blocks[pre].body.contains(&inv_add.u32_id()),
            "inv_add chain should hoist to preheader"
        );
        assert!(
            m.blocks[header].body.contains(&guarded_div.u32_id()),
            "trapping UDiv must NOT hoist out of loop"
        );
        assert_eq!(ALL, 63);
        assert_eq!(ALL_REA003, 63 | 64 | 128);
        m.write();
    }
}
