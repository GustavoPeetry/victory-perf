# Diagnostics, Benchmarking, and Operational Switches

This guide documents `victory-perf`'s built-in telemetry, reproducible
benchmarking and bisection procedures, crash/device-removal triage, and the
operational `VICTORY_*` environment variables verified on `main` at `52a60947`.

---

## 1. Built-in telemetry (`--release` vs. `diag` build)

`victory` writes all runtime logs via `tracing` to `stderr`
(`crates/victory-core/src/logging.rs`) and appends unhandled faults to
`<data_dir>\<TITLE_ID>\crashes.log` (`apps/victory/src/run.rs:109-110`).
There is no separate telemetry daemon or CSV/JSON exporter; capture full logs in
PowerShell by teeing both streams outside the repository:

```powershell
# Example: capture full console output to a log file outside the repo
.\target\release\victory.exe run --console series-s --execute "<path-to-title-exe>" *>&1 |
    Tee-Object -FilePath "D:\victory-data\victory-run.log"
```

Filtering uses `VICTORY_LOG` (falling back to `RUST_LOG`). Standard `--release`
builds hard-cap log verbosity at `INFO` (`CAP = LevelFilter::INFO` in
`crates/victory-core/src/logging.rs:14-18`). To unlock `DEBUG`/`TRACE` logs and
deep profiling instrumentation, build the dedicated `diag` profile defined in
`Cargo.toml` (compiled with release optimizations into `target\diag\` so it does
not overwrite `target\release\victory.exe`):

```powershell
cargo build --profile diag -p victory --features diagnostics
```

### What standard `--release` builds log automatically

Even without `diagnostics`, `Meter::frame()` (`crates/victory-gpu/src/host.rs:6595-6822`),
`memcopy::log()` (`crates/victory-gpu/src/memcopy.rs:102-127`), and
`queue::present()` (`crates/victory-gpu/src/queue.rs:795-808`) emit:

- **Every ~1 second:**
  - Window caption suffix: `| VIctory <fps> fps <ms> ms/frame <draws> draws/frame`.
  - `the card's clock per frame` (`card_clock_line()`): estimates GPU idle time
    (`nothing of ours queued`), GPU busy %, idle time while the submitting thread
    was preparing (`the card idle behind us`), and submitting-thread wait time.
  - `submissions sent ahead per frame` (`late_line()`): flight settle counts
    and wait reasons (`VICTORY_LATE_WAIT` / `VICTORY_RELAY`).
  - `indirect mesh dispatches`, `indirect calls joined per frame` (`batch_line()`),
    `shader writes looked for on the CPU per frame` (`back_line()`), and card
    rehash/compare pass counts.
- **Every ~5 seconds:**
  - Main frame breakdown: `<fps> fps, <ms> ms/frame (<cpu> preparing: copies ... ms ... MB, textures ... ms ... MB; <gpu> on the GPU; <back_ms> copying back <back_mb> MB), <draws> draws/frame, dropped <dropped> without a host pipeline (<making> being made) and <indirect> indirect`, plus heap buffer passes, texture slot queue counters, and depth-twin counters.
  - `memcopy per frame` and `waits for every earlier shader write per frame` (`barrier_line()`).
- **Power-of-two presented frames (`1`, `2`, `4`, ..., `256`, `512`, ...):**
  - `frame presented`: cumulative `draws`, `drawn`, `dispatches`, `dispatched`.
  - From frame `256` onward: `shaders from the cache and translated so far` (`hits`, `misses`).

### Additional instrumentation in `--profile diag --features diagnostics`

| Feature / Env Var | Source | Output & caveats |
|---|---|---|
| `copystat` periodic breakdown (automatic in `diag`) | `crates/victory-gpu/src/copystat.rs:1055-1318` | Logs every ~5 s: per-frame `RecordCalls`/`RecordNs` across recording threads, `PaceWaits`/`PaceUs`, `BackScanUs`/`BackDiffer`/`BackSyncUs`, card wait histogram (`<2 ms`, `2–8 ms`, `8–32 ms`, `>32 ms`), top-12 slowest wait workloads, submitter user/kernel/lock wait times, page faults, and RAM/VRAM breakdown. Every 6th memory report walks virtual memory via `VirtualQuery` on `victory-memory-map` (disable the walk with `VICTORY_MEMORY_MAP=0`) |
| `VICTORY_GPU_TIMING=1` | `crates/victory-gpu/src/host.rs:1376-1380, 1962-1964`, `copystat.rs:1216-1250` | Inserts GPU timestamps around noted commands and logs the top-40 slowest commands/pipelines in ms/frame. **Caveat:** setting `VICTORY_GPU_TIMING` forces `late_stage() == 0` (`host.rs:1962-1964`), disabling late wait and relay; never use GPU-timing runs for overlap FPS comparisons |
| `VICTORY_CAPTURE=<dir>` (+ `VICTORY_CAPTURE_FRAMES=<list>`, `VICTORY_CAPTURE_WORLD=1`, `VICTORY_CAPTURE_WORLD_ALSO=<list>`, `VICTORY_DUMP_TARGETS=1`) | `crates/victory-gpu/src/capture.rs` | Saves half-resolution `frame-<n>.png` captures on powers of two / listed / world-entry frames plus `frame-<n>-ops.txt` operation logs and target dumps (`VICTORY_DUMP_TARGETS=1`). Keep `<dir>` outside the repo. **Caveat:** open `vitutu1/victory-perf` Issue #8 notes Series S two-plane HUD overlays can be omitted from captured PNGs |
| `VICTORY_D3D12_DRED=1` | `crates/victory-gpu/src/host.rs:652-660, 1359-1365, 4879` | Enables D3D12 DRED auto-breadcrumbs and page-fault reporting on device removal, correlating faulting GPU VAs against named host allocations |
| `VICTORY_D3D12_DEBUG=1` (+ `VICTORY_D3D12_GBV=1`) | `crates/victory-gpu/src/host.rs:624-649` | Enables the D3D12 debug layer (and GPU-based validation without state tracking when `VICTORY_D3D12_GBV` is also set). Significantly alters frame timing |
| `VICTORY_D3D12_WARP=1` | `crates/victory-gpu/src/host.rs:665-668` | Runs on the Microsoft Basic Render Driver (WARP) so shader wild accesses surface as CPU access violations instead of GPU resets |
| `VICTORY_CARD_ACCOUNTING=1` | `crates/victory-gpu/src/draw.rs:1891-1900`, `resource.rs:1444-1470` | Available in all builds; periodically audits resident vs. unmapped/evicted bindless textures on a background thread |
| `VICTORY_STACKS=<s>[,<s>...]` | `crates/victory-loader/src/stacks.rs` | Prints `module+rva` call stacks of every thread to `stderr` at the specified seconds after startup |
| `VICTORY_WATCH=<rva>[,<rva>...]` | `apps/victory/src/run.rs:160-186` | Polls `u32` values at `title_base + rva` (hex) every 5 ms and logs changes |
| `VICTORY_CRASH_SAVE=<rva>:<len>,...` | `apps/victory/src/run.rs:141-158` | Dumps named title memory slices to `<cache_dir>\crash-<rva>.bin` on crash |
| `VICTORY_AUDIO_CAPTURE=<file.wav>` | `crates/victory-hle/src/modules/audio/capture.rs` | Records title audio output to a `.wav` file outside the repo |
| `VICTORY_SKIP_PIPELINES=<hash,...>` | `crates/victory-gpu/src/pipeline.rs:446-465` | Drops draws/dispatches using the listed shader hashes (`diagnostics` build only) for pass isolation |

---

## 2. Reproducible benchmarking and bisection workflow

### Mandatory controls before comparing two runs

1. **Pin hardware, OS, and title inputs:** Keep the exact same GPU, driver
   version, RAM/pagefile configuration, title files, save checkpoint, `--console`
   profile, and `VICTORY_RENDER_SCALE`.
2. **Control cache state explicitly:** Distinguish three states and never mix
   them when evaluating FPS:
   - *Cold:* empty `<cache_dir>\shaders`, empty `<cache_dir>\pipelines`, freshly cleared GPU driver cache.
   - *DXIL-warm:* populated `<cache_dir>\shaders` via `victory precompile` (or `--pipelines` for compute PSOs), before a live run has filled `host-pipelines.bin` with graphics PSOs.
   - *PSO-warm:* second or later live run on the same GPU/driver showing `host pipeline library loaded` with `loaded` dominating `made` (verify per [docs/shader-pipeline-architecture.md](shader-pipeline-architecture.md#5-verifying-both-cache-layers-in-logs)).
3. **Separate phases and record stability:** Do not blend loading screens or
   menus with 3D world checkpoints. Record both steady-state frame metrics (after
   pipeline workers drain and `dropped without a host pipeline` settles near `0`)
   and **session duration until stop/crash** (plus `available physical RAM`).
4. **Match scene and camera:** Never compare frame-time numbers from different
   checkpoints, camera angles, or draw counts (`draws/frame` must match).

### Controlled A/B comparisons

Change **one variable or commit at a time** against the same checkpoint:

| Comparison | Command / Switch delta | What it isolates |
|---|---|---|
| Synchronous baseline vs. current defaults | Set `VICTORY_RELAY=0` and leave `VICTORY_LATE_WAIT` unset vs. defaults (`RELAY=2`, `LATE_WAIT=3`) | Measures net overlap gain vs. CPU/GPU pipelining hazards |
| Individual indirect/barrier optimizations | Flip one of `VICTORY_HOIST_CLAMPS=0`, `VICTORY_CLAMP_SCOPED=0`, `VICTORY_CLAMP_DEDUP=0`, `VICTORY_BARRIER_MERGE=0`, `VICTORY_INDIRECT_WINDOW=0`, `VICTORY_INDIRECT_MERGE=0` | Verifies whether a pass reduces GPU/CPU cost without mesh flicker or dropped draws |
| Memory & VRAM tier pressure | Compare `VICTORY_CARD_BUFFERS=0` vs. `2048` vs. `512–1024`, or `VICTORY_LIBRARY_TRIM=0` vs. `1` | Separates PCIe copy/paging overhead from VRAM oversubscription and 16 GB RAM working-set pressure |
| Driver vs. runtime stall | `VICTORY_SKIP_PIPELINES=<hash>` (in `diag`) or `VICTORY_RT_RESOLVE=0` / `VICTORY_ACTIVATE_RAYS=0` | Isolates expensive shaders/passes without changing engine scene flow |

### Crash, hang, and device-removal triage

1. Check `<data_dir>\<TITLE_ID>\crashes.log` for the faulting `module+rva` and
   thread stack trace.
2. If the crash occurs ~30 s into a long load/compile stall on `1.0.2.1`, test
   `VICTORY_NO_WATCHDOG=1` (`crates/victory-loader/src/hooks.rs`) to rule out
   the title's own watchdog fault (`mov dword ptr [0], imm32`).
3. On `DXGI_ERROR_DEVICE_REMOVED` / `DEVICE_HUNG`, leave `VICTORY_TRAIL` enabled
   (default on; `=0` disables the 96-entry trail logged by `log_removal_context()`)
   and run a `diag` build with `VICTORY_D3D12_DRED=1` to capture the faulting VA,
   last host list notes, and `pipeline::library::removal_note()` page status.
4. On multi-second hangs or deadlocks, use `VICTORY_STACKS=15,30,60` (`diag`
   build) to dump all thread call stacks before termination.

---

## 3. Audited operational `VICTORY_*` reference (`52a60947`)

Only operational, tuning, and diagnostic variables directly verified at
`52a60947` are listed below (unit-test child-process variables are omitted).

### Paths and runtime configuration

| Variable | Default | Source | Notes |
|---|---|---|---|
| `VICTORY_CONFIG` | `.\victory.toml` | `crates/victory-core/src/config.rs` | Override config TOML path |
| `VICTORY_GAME_DIR` | From `victory.toml` | `crates/victory-core/src/config.rs` | Title dump path (must sit outside repo) |
| `VICTORY_CACHE_DIR` | `%LOCALAPPDATA%\victory\cache` | `crates/victory-core/src/config.rs` | Holds `images\`, `shaders\`, and `pipelines\` (outside repo) |
| `VICTORY_DATA_DIR` | `%LOCALAPPDATA%\victory\data` | `crates/victory-core/src/config.rs` | Holds `<TitleID>\storage`, `temp`, and `crashes.log` (outside repo) |
| `VICTORY_LOG` | `info` (fallback `RUST_LOG`) | `crates/victory-core/src/logging.rs` | Trace/env filter; `--release` caps at `INFO`, `--features diagnostics` allows `TRACE` |

### Memory, VRAM, and copy-back tiers

| Variable | Default at `52a60947` | Source | Notes |
|---|---|---|---|
| `VICTORY_CARD_BUFFERS` | On (`!= "0"`) | `crates/victory-gpu/src/shadow.rs:180` | Mirrors eligible buffers onto VRAM and diffs via GPU `CARD_COMPARE` |
| `VICTORY_CARD_BUFFERS_MIN_KB` | `256` (KiB, clamped `>= 4 KiB`) | `crates/victory-gpu/src/shadow.rs:205` | Minimum buffer size for card mirroring |
| `VICTORY_CARD_BUFFERS_MIN_MB` | Unset (overrides `_MIN_KB`) | `crates/victory-gpu/src/shadow.rs:200` | Sets minimum in MiB (`1` omits `< 1 MiB` buffers from VRAM) |
| `VICTORY_CARD_BUFFERS_MB` | `2048` (MiB) | `crates/victory-gpu/src/shadow.rs:211` | Cap for VRAM card buffer copies (excluding hash twins) |
| `VICTORY_CARD_WINDOWS_MB` | `1536` (MiB; `0` disables) | `crates/victory-gpu/src/shadow.rs:314` | Cap for VRAM mirrors of read-mostly title memory windows |
| `VICTORY_CARD_VRAM_PERCENT` | `92` (clamped `50..=95`) | `crates/victory-gpu/src/shadow.rs:250` | Max % of DXGI local VRAM budget usable by card buffers/windows |
| `VICTORY_TEXTURE_ROOM_PERCENT` | `82` in code (clamped `50..=97`) | `crates/victory-gpu/src/resource.rs:1398` | Max % of DXGI budget for bindless texture copies after 2 GiB floor. *Note:* doc comment still says `75`, code uses `82` |
| `VICTORY_TEXTURE_BUDGET_MB` | Unset (uses `%` of DXGI budget) | `crates/victory-gpu/src/resource.rs:1375` | Hard MiB cap for filled bindless texture copies |
| `VICTORY_TEXTURE_EVICT` | On (`!= "0"`) | `crates/victory-gpu/src/resource.rs:1617` | Frees host texture copies unused for `120` frames (`EVICT_IDLE_FRAMES`) |
| `VICTORY_COPY_BACK_ONLY_READBACK` | Off (`== "1"` enables) | `crates/victory-gpu/src/readback.rs:30` | Skips CPU copy-back on non-readback shader buffers (`README` preset sets `0`) |
| `VICTORY_REBAR` | `0` (off; `1` windows, `2` + small buffers) | `crates/victory-gpu/src/rebar.rs:127` | Uses `D3D12_HEAP_TYPE_GPU_UPLOAD` when hardware/OS ReBAR (`OPTIONS16`) exists; never places CPU-readback buffers |
| `VICTORY_REBAR_MB` | 25% of budget (max `4 GiB`) | `crates/victory-gpu/src/rebar.rs:141, 178` | Caps ReBAR placement bytes |
| `VICTORY_INPLACE` | `0` (off; `1` private allocs, `2` + small buffers) | `crates/victory-gpu/src/inplace.rs:88` | Pins title pages via `OpenExistingHeapFromAddress*` up to `VICTORY_INPLACE_MB` (`4096`). **Do not combine level `2` with `VICTORY_LATE_WAIT`** |
| `VICTORY_KEEP_SNAPSHOTS` | Off (`== "1"` enables) | `crates/victory-gpu/src/shadow.rs:1526` | Keeps persistent CPU snapshots up to `VICTORY_KEEP_SNAPSHOTS_MB` (`256`); **disables `VICTORY_RELAY`** |

### Submission, synchronization, indirect draws, and barriers

| Variable | Default at `52a60947` | Source | Notes |
|---|---|---|---|
| `VICTORY_RELAY` | `2` (on; `0` off, `1` coarse, `2` exact) | `crates/victory-gpu/src/shadow.rs:556`, `host.rs:2065` | Asynchronous `relay_worker` landing/readback; disabled under `VICTORY_GPU_TIMING` or `VICTORY_KEEP_SNAPSHOTS=1` |
| `VICTORY_LATE_WAIT` | `3` (via relay; `0` off; `1/2/3`) | `crates/victory-gpu/src/host.rs:1959` | Defers submission waits (`3` also decouples present copies); forced to `0` by `VICTORY_GPU_TIMING` |
| `VICTORY_LATE_WAIT_LISTS` | `256` (min `1`) | `crates/victory-gpu/src/host.rs:2168` | Max in-flight host lists before settling |
| `VICTORY_WIDE_BACK` | On (`!= "0"`) | `crates/victory-gpu/src/shadow.rs:1048` | Multi-threaded CPU snapshots and diffs (`(cores / 4).clamp(2, 6)` threads) |
| `VICTORY_WIDE_MERGE` | Off (`== "1"` enables) | `crates/victory-gpu/src/host.rs:1780` | Multi-threaded CPU merge of fetched card-compare pieces |
| `VICTORY_CARD_INLINE_FETCH` | On (`!= "0"`) | `crates/victory-gpu/src/host.rs:1585` | Copies differing 64 KiB pieces inline in `CARD_COMPARE` into 128 MiB L0 ring |
| `VICTORY_CARD_REHASH_RUNS` | On (`!= "0"`) | `crates/victory-gpu/src/host.rs:1750` | Coalesces post-upload card rehash dispatches |
| `VICTORY_CARD_REHASH_GAP` | `4` (pieces) | `crates/victory-gpu/src/host.rs:1758` | Max gap bridged between rehash runs (`0` = touching pieces only) |
| `VICTORY_HOIST_CLAMPS` | `7` (`0` off; `1..=7`) | `crates/victory-gpu/src/indirect.rs:609` | Hoists indirect clamps before draws; mode `7` serializes clamp budget access cleanly |
| `VICTORY_CLAMP_SCOPED` | On (`!= "0"`) | `crates/victory-gpu/src/indirect.rs:270` | Scopes clamp UAV barriers to argument/count buffers and drops post-clamp global barrier |
| `VICTORY_CLAMP_DEDUP` | On (`!= "0"`) | `crates/victory-gpu/src/host.rs:5166` | Deduplicates scoped UAV barriers per host list |
| `VICTORY_BARRIER_MERGE` | On (`!= "0"`) | `crates/victory-gpu/src/host.rs:292` | Skips redundant global UAV barriers when `uav_pending` is clear |
| `VICTORY_CLAMP_LIGHT_RESTORE` | On (`!= "0"`) | `crates/victory-gpu/src/draw.rs:158` | Restores only PSO after a clamp dispatch instead of dirtying graphics root state |
| `VICTORY_INDIRECT_WINDOW` | On (`!= "0"`) | `crates/victory-gpu/src/draw.rs:203` | Clamps contiguous indirect argument windows up to `VICTORY_INDIRECT_WINDOW_MOST` (`4096`, `16..=65536`) |
| `VICTORY_INDIRECT_MERGE` | On (`!= "0"`) | `crates/victory-gpu/src/draw.rs:215` | Batches matching sequential indirect draws into a single `ExecuteIndirect` (requires `VICTORY_INDIRECT_WINDOW=1`) |

### Textures and copy path

| Variable | Default at `52a60947` | Source | Notes |
|---|---|---|---|
| `VICTORY_KNOWN_WRITES` | On (`!= "0"`) | `crates/victory-gpu/src/memcopy.rs:35` | Keeps write watches across `CopyMemoryToMemoryX` |
| `VICTORY_TEXTURE_OVERLAP` | On (`!= "0"`) | `crates/victory-gpu/src/memcopy.rs:41` | Refreshes textures on worker threads alongside buffer copies |
| `VICTORY_FAST_HASH` | On (`!= "0"`) | `crates/victory-gpu/src/memcopy.rs:47` | Uses `lane_hash` instead of BLAKE3 for guarded texture diff checks |
| `VICTORY_TEXTURE_HASH_EVERY` | `8` (`0` disables) | `crates/victory-gpu/src/memcopy.rs:53` | Re-hash cadence in frames for guarded unchanged textures |
| `VICTORY_TEXTURE_STAGE_MB` | `48` (MiB) | `crates/victory-gpu/src/draw.rs:1984` | Per-frame upload byte budget for bindless `heap_textures` |
| `VICTORY_TEXTURE_STAGE_MS` | `10` (ms) | `crates/victory-gpu/src/draw.rs:2051` | Per-frame time cap for bindless `heap_textures` uploads |
| `VICTORY_TEXTURE_NO_WAIT` | Off (`== "1"` enables) | `crates/victory-gpu/src/texture.rs:317` | Reads through kernel without stalling on memory remaps |
| `VICTORY_HEAP_WARMUP_BUCKETS` | `8` (clamped `1..=64`) | `crates/victory-gpu/src/draw.rs:1639` | Progressive `heap_buffers` bucket scan rate (`64` scans all on pass 0) |

### Shaders, pipelines, lighting stand-ins, and loader hooks

| Variable | Default at `52a60947` | Source | Notes |
|---|---|---|---|
| `VICTORY_PIPELINE_WAIT_MS` | `12000` (ms) | `crates/victory-gpu/src/pipeline/make.rs:184` | Per-frame, per-kind wait budget; lower (e.g. `500`) favors frame pacing over completeness |
| `VICTORY_PIPELINE_WAIT_SHARED` | `Window` (`reserve` / `0`) | `crates/victory-gpu/src/pipeline/make.rs:317` | How concurrent recording threads share the frame's wait budget |
| `VICTORY_PIPELINE_GATE_S` | `120` (s; `0` disables) | `crates/victory-gpu/src/pipeline/make.rs:230` | Cumulative extra loading wait allowance when `VICTORY_PIPELINE_WAIT_MS` is lower than `4 s`/`12 s` |
| `VICTORY_LIBRARY_MAP` / `_HANDLE` / `_TRIM` | All on (`!= "0"`) | `crates/victory-gpu/src/pipeline/library.rs:81, 90, 406` | Maps `host-pipelines.bin` via `MapViewOfFile`, keeps handle open across saves, and unlocks pages when idle (`2–10 s` by free RAM) |
| `VICTORY_LIBRARY_LOCK` | Off (`== "1"` enables) | `crates/victory-gpu/src/pipeline/library.rs:473` | Restores global single mutex around library calls (diagnostic only) |
| `VICTORY_RT_RESOLVE` / `VICTORY_ACTIVATE_RAYS` | Both compiled by default | `crates/victory-gpu/src/pipeline/make.rs:77, 94` | Set `=0` to omit `CS_RaytracingResolve` (when `VICTORY_BVH_BUILDER` is off) or `CS_ScreenProbes_ActivateRays` |
| `VICTORY_ACCC` | Compiled by default | `crates/victory-gpu/src/pipeline.rs:587` | Set `=0` to refuse DXIL-less `ACCC` compute pipelines |
| `VICTORY_BVH_BUILDER` / `VICTORY_BVH_QUARANTINE` | Both off (`== "1"` enables) | `crates/victory-gpu/src/pipeline.rs:280, 304` | Experimental RT BVH builder and write quarantine (`docs/fork-ray-tracing.md`) |
| `VICTORY_AMBIENT_STUB` / `VICTORY_SHADOWMASK_STUB` / `VICTORY_EXPOSURE_STUB` | Off (`== "1"` enables) | `crates/victory-gpu/src/shader_cache.rs:138, 182` | Lighting/exposure stand-ins (defaults `0.1`, `1.0`, `0.18`; override via `*_VALUE` / `*_HASHES`) |
| `VICTORY_WAVE_WIDE` / `VICTORY_WAVE_DEEP_SKIPS` / `VICTORY_WAVE_OWN_BITS` / `VICTORY_F16_NEAREST` / `VICTORY_GROUP_SYNC` / `VICTORY_SHADER_OPT` / `VICTORY_SHADER_LOOP_LIMIT` | Keyed in `shader_cache::build()` | `crates/victory-gpu/src/shader_cache.rs:69-95` | Included in DXIL shader cache key (`VICTORY_SHADER_LOOP_LIMIT` applies only in `diagnostics` builds) |
| `VICTORY_WAVE_LOCAL` / `VICTORY_FIRST_HALF` / `VICTORY_FIRSTLANE_OWN` / `VICTORY_MESH_OUTPUT_STRICT` / `VICTORY_STRICT_JOINS` | Read in `victory-shader`, **not** keyed in `build()` | `crates/victory-shader/src/emit/` | Clear `<cache_dir>\shaders` or use a separate `VICTORY_CACHE_DIR` when changing these five switches |
| `VICTORY_OWN_UPSCALER` / `VICTORY_RENDER_SCALE` / `VICTORY_NO_WATCHDOG` | Off | `crates/victory-loader/src/hooks.rs` | Build `1.0.2.1` hooks (`b99f876b74d4a77f`). Note Issue #2 when `VICTORY_OWN_UPSCALER=1` + scale `0.75`/`1.0` |

### Hardware-class starting points (unbenchmarked tuning guides)

- **12 GB+ VRAM / 32–64 GB RAM:** Start with the `README.md` `--console series-s`
  profile (`VICTORY_CARD_BUFFERS_MIN_MB=1`, `VICTORY_CARD_BUFFERS_MB=2048`,
  `VICTORY_RELAY=2`, `VICTORY_LATE_WAIT=3`, `VICTORY_WIDE_BACK=1`,
  `VICTORY_LIBRARY_TRIM=0` or `1`, `VICTORY_REBAR=0`, `VICTORY_COPY_BACK_ONLY_READBACK=0`,
  `VICTORY_AMBIENT_STUB=1`, `VICTORY_SHADOWMASK_STUB=1`, `VICTORY_OWN_UPSCALER=1`,
  `VICTORY_RENDER_SCALE=0.5`), preceded by `victory precompile --pipelines`.
- **8 GB VRAM or 16 GB RAM (informed by `vitutu1/victory-perf` Issue #3 & `rebar.rs`):**
  - Keep `VICTORY_LIBRARY_TRIM=1` (default) so mapped `host-pipelines.bin` pages
    unlock quickly (`2 s` idle below 2 GiB free RAM) instead of pinning working set.
  - Constrain VRAM mirrors (`VICTORY_CARD_BUFFERS_MB=768` or `1024`,
    `VICTORY_CARD_WINDOWS_MB=256` or `512`, `VICTORY_TEXTURE_ROOM_PERCENT=72`,
    `VICTORY_CARD_VRAM_PERCENT=86`) and check `VICTORY_CARD_ACCOUNTING=1` if shared
    system memory grows.
  - Keep `VICTORY_REBAR=0` unless `host.rs` / `rebar.rs` confirms ReBAR is
    supported and A/B testing improves frame time without PCIe readback stalls.
  - After a first warm run populates `host-pipelines.bin`, evaluate lowering
    `VICTORY_PIPELINE_WAIT_MS` (e.g. `500`–`2000`) with `VICTORY_NO_WATCHDOG=1`
    if 6–24 s paging spikes appear on 16 GB RAM.
