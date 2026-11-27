# Contributing to `victory-perf`

Thank you for helping improve `victory-perf`. Keep pull requests focused,
verifiable against the current source tree, and strictly compliant with
[docs/asset-policy.md](docs/asset-policy.md).

---

## 1. Asset policy (non-negotiable)

Never commit or paste proprietary game material into commits, tests, docs,
issues, or pull requests:

- **Forbidden:** title executables, DLLs, archives, textures, shaders, audio,
  extracted strings/tables, shader dumps, frame captures, screenshots, or
  commercial game title names.
- **Allowed:** original code, clean-room format support, BLAKE3 build hashes,
  per-build hook offsets under neutral names, synthetic unit tests, and text
  logs containing no proprietary binary/hex payloads.
- Install the pre-commit hook before committing:

```powershell
cargo xtask install-hooks
```

Run `cargo xtask policy` and `cargo xtask policy --staged` to verify that every
file passes the allowlist in [xtask/src/policy.rs](xtask/src/policy.rs).

---

## 2. Provenance and license

This repository is licensed under **GPL-3.0-or-later** (see [LICENSE](LICENSE)).

- **Do not copy GPL-2.0-only code** (including from projects referenced solely
  for behavioral comparison).
- When porting or cherry-picking changes from another **GPL-3.0-or-later**
  fork (e.g. `Marvisak/VIctory`, `fxceless/victory-fork`,
  `VIctory-Belobug-AMD`, `ZigZagBoi1/ScarlettPort`), state the source repository,
  commit SHA or PR number, and any local adaptations in your commit message and
  PR description.

---

## 3. Required local verification

Match [the CI workflow](.github/workflows/ci.yml) before opening or updating a
pull request:

```powershell
cargo xtask policy
cargo xtask policy --staged
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --locked
```

Also verify formatting and whitespace (`git diff --check`) and confirm that any
local `victory.toml` or test data sits strictly outside the repository.

---

## 4. Benchmarking, hardware testing, and documentation discipline

- **Disclose test scope accurately:** Clearly separate compile/unit-test
  verification from live Windows GPU testing (noting GPU, VRAM, RAM, driver,
  `--console` mode, cache state, and active `VICTORY_*` overrides). Never claim
  unrun tests or fabricated FPS numbers.
- **Follow the measurement protocol:** When submitting performance changes or
  regression triage, use the matched-scene and cache-control rules in
  [docs/diagnostics-and-benchmarking.md](docs/diagnostics-and-benchmarking.md)
  and record text-only run notes using [docs/test-log.md](docs/test-log.md).
- **Avoid unsolicited `README.md` benchmark edits:** Leave self-reported
  benchmark tables and status numbers in [README.md](README.md) to the
  maintainer's verified hardware runs.
- **Shader cache & translator changes:** If a change modifies shader lifting
  inside `crates/victory-shader/src/` or adds/reads a new `VICTORY_*` switch
  there, ensure `crates/victory-shader/src/lib.rs` (`sources()`) and
  `crates/victory-gpu/src/shader_cache.rs` (`build()`) account for it (see
  [docs/shader-pipeline-architecture.md](docs/shader-pipeline-architecture.md)).
