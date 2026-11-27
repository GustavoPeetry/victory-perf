## Summary

<!-- Describe what this PR changes and why. -->

## Scope & Provenance

- **Category:** <!-- perf / bugfix / shader / loader / docs / ci -->
- **Provenance (if ported or adapted):** <!-- Source repo + commit SHA / PR, GPL-3.0-or-later compatibility confirmed, or "Original work" -->
- **Files touched:** <!-- Brief summary; confirm no proprietary assets or title names -->

## Testing & Verification

- [ ] `cargo xtask policy` and `cargo xtask policy --staged` passed
- [ ] `cargo clippy --workspace --all-targets --locked -- -D warnings` passed
- [ ] `cargo clippy --workspace --all-targets --all-features --locked -- -D warnings` passed
- [ ] `cargo test --workspace --locked` passed
- [ ] `git diff --check` passed (LF line endings preserved per `.gitattributes`)
- [ ] Tested on live hardware (describe GPU, VRAM, RAM, driver, console profile, cache state, switches, and session stability duration below) **OR** compile / unit tests / docs only (no live title run performed)

### Hardware / Runtime Notes (if tested live)

- **Hardware & OS:** <!-- CPU / GPU + VRAM / RAM + pagefile / Windows build / GPU driver -->
- **Console profile & switches:** <!-- e.g. --console series-s, VICTORY_* overrides -->
- **Cache state:** <!-- Cold / DXIL-warm (victory precompile) / PSO-warm (2nd+ run) -->
- **Observed results & stability:** <!-- Matched-scene metrics, session duration, visual checks -->

## Limitations

<!-- Note any unverified configurations (e.g., AMD vs. NVIDIA, 8 GB vs. 12 GB+ VRAM, 16 GB RAM), known trade-offs, or AI assistance if applicable. -->
