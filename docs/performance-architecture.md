# Performance Architecture and Cross-Fork Hypotheses

This review documents `victory-perf` (`vitutu1/victory-perf`) at `52a60947`
across CPU–GPU submission, memory mirroring, indirect-draw clamping, pipeline
preparation, and shader translation, and frames testable hypotheses against
sibling forks (`fxceless/victory-fork`, `Belobug/VIctory-Belobug-AMD`,
`ka1qux/BRVIctory-ka1qux`, `uiliannn/VItoriaBr`, and `Zettrox/ztrx`).

**Evidence rubric.** *Source-confirmed* findings describe the checked-out
implementation at `52a60947`. *Repository-reported* measurements come from
`README.md`, source comments, or upstream GitHub issues/PRs.
*Literature-supported* points refer to public hardware and D3D12 contracts.
*Open PRs / cross-fork hypotheses* mark unmerged experiments requiring matched
hardware validation. This document contributes no new gameplay measurements.

---

## 1. Execution model and `victory-perf` CPU–GPU pipeline

```text
Xbox offline    HLSL → compiler IR → RDNA 2 microcode → unified GDDR6 GPU
Windows D3D12   HLSL → DXIL → host UMD JIT → native GPU ISA (discrete VRAM)
victory-perf    console DXBC/GFX10 → decode → SSA → control-flow & bindings
                → SM 6.6 DXIL (<cache_dir>\shaders) → host D3D12 PSO (<cache_dir>\pipelines)
                → mapped x86-64 title code + HLE → host D3D12 direct queue
```

### Submission, synchronization, and relay defaults (`host.rs`, `shadow.rs`, `memcopy.rs`)

**Source-confirmed.** Title queues share a host direct queue (`crates/victory-gpu/src/queue.rs`,
`crates/victory-gpu/src/host.rs`) and split lists at guest memory-value waits
(`VICTORY_QUEUE_WAIT=1`, `VICTORY_SPLIT_WAITS=1`, `VICTORY_CUT_WAITS=1`,
`VICTORY_WAIT_FORCE=1`). Unlike source comments describing earlier synchronous
baselines, current defaults in `victory-perf` (`52a60947`) enable:

| Path | Default in `52a60947` | Source & behavioral notes |
|---|---|---|
| `VICTORY_RELAY` | Stage `2` (on except `cfg!(test)` or `VICTORY_KEEP_SNAPSHOTS=1`) | `crates/victory-gpu/src/shadow.rs:556`, `crates/victory-gpu/src/host.rs:2049-2066`. Moves flight readback, comparison, and completion onto `relay_worker` without holding the submission lock; stage `2` restricts dependency settles to declared overlapping ranges and `VICTORY_LATE_WAIT_LISTS` (default `256`) |
| `VICTORY_LATE_WAIT` | Stage `3` when `relay()` is on (forced to `0` under `VICTORY_GPU_TIMING`) | `crates/victory-gpu/src/host.rs:1959-2015`. Defers GPU waits across submissions and executes presents as independent flights (`late_free_present()`). Use `VICTORY_RELAY=0` with `VICTORY_LATE_WAIT` unset as the synchronous control |
| `VICTORY_WIDE_BACK` | On (`!= "0"`) | `crates/victory-gpu/src/shadow.rs:1048`. Parallelizes CPU snapshot capture (`take_snapshots`) and comparisons (`find_all`) across 1 MiB chunks on `(cores / 4).clamp(2, 6)` worker threads |
| `VICTORY_KNOWN_WRITES`, `VICTORY_TEXTURE_OVERLAP`, `VICTORY_FAST_HASH`, `VICTORY_TEXTURE_HASH_EVERY` | All on; check interval `8` | `crates/victory-gpu/src/memcopy.rs:32-58`. Uses `CopyMemoryToMemoryX` destination ranges to preserve write watches, overlaps texture refreshes with buffer preparation, uses `lane_hash` instead of BLAKE3, and checks guarded unchanged textures every 8 frames |

### Original `victory-perf` GPU and CPU optimization stack

**Source-confirmed.** On top of `fxceless/victory-fork` (`310c6eb`),
`victory-perf` enables the following optimizations by default at `52a60947`:

1. **Coalesced `CARD_COMPARE` + L0 inline readback ring (`crates/victory-gpu/src/host.rs:1575-1763`):**
   - `CARD_COMPARE` dispatches 64 coalesced lanes per 64 KiB piece (`1024`-byte
     stride) with LDS hash reduction and writes changed 64 KiB pieces inline into
     a `CARD_INLINE_SLOTS = 2048` (~128 MiB L0 system RAM) shared ring buffer
     (`VICTORY_CARD_INLINE_FETCH` on by default), avoiding per-landing
     `CopyBufferRegion` + `WaitForSingleObject` stalls.
   - Commit `a7b2fb8` enables gap-coalesced card rehash passes
     (`VICTORY_CARD_REHASH_RUNS` on by default, bridging gaps up to
     `VICTORY_CARD_REHASH_GAP = 4` pieces).
2. **Scoped, deduplicated barriers and hoisted indirect clamps (`crates/victory-gpu/src/indirect.rs`, `draw.rs`, `host.rs`):**
   - `VICTORY_CLAMP_SCOPED` (on by default) scopes indirect-clamp UAV barriers to
     the source argument and count buffers rather than draining all GPU UAVs, and
     drops the redundant post-clamp global barrier (`INDIRECT_ARGUMENT` transition
     orders the scratch output).
   - `VICTORY_CLAMP_DEDUP` (on by default, `ffed22d`) skips redundant per-list
     clamp UAV waits via `HostList::take_uav_barrier_for()`.
   - `VICTORY_BARRIER_MERGE` (on by default) tracks `uav_pending` (including
     image stores) per `HostList` and omits redundant global UAV barriers.
   - `VICTORY_INDIRECT_WINDOW` (on by default, `VICTORY_INDIRECT_WINDOW_MOST = 4096`),
     `VICTORY_CLAMP_LIGHT_RESTORE` (on by default), and `VICTORY_INDIRECT_MERGE`
     (on by default) batch sequential indirect calls and avoid full graphics
     state invalidation after compute clamps.
   - Commits `ec5a20f` and `ffa37e7` add hoisted indirect clamping (`VICTORY_HOIST_CLAMPS`,
     default mode `7`, ported from `VIctory-Belobug-AMD`), moving indirect clamps
     into a pre-pass list before draws and scoping the inter-clamp UAV barrier to
     the shared budget buffer.
3. **CPU stall elimination (`crates/victory-gpu/src/draw.rs`, `descriptor/host.rs`, `resource.rs`):**
   - `State::heap_buffers` warms up 64 descriptor buckets progressively
     (`VICTORY_HEAP_WARMUP_BUCKETS = 8` per pass, `a7b2fb8`) instead of scanning
     all buffer slots in a single frame-0 spike.
   - `HostHeap` caps failed full-ring dynamic descriptor searches at `WALKS = 8`
     per frame (`crates/victory-gpu/src/descriptor/host.rs:922-929`, `ec5a20f`).
   - Idle host copies of filled textures are eligible for eviction after
     `EVICT_IDLE_FRAMES = 120` (`VICTORY_TEXTURE_EVICT` on by default,
     `crates/victory-gpu/src/resource.rs:1613-1748`).

### Memory and transfer tiers (`shadow.rs`, `resource.rs`, `rebar.rs`, `inplace.rs`)

**Literature-supported and source-confirmed.** Xbox Series X and S share
physical GDDR6 across CPU and GPU, whereas desktop GPUs split system RAM and
VRAM over PCIe. `victory-perf` manages four distinct tiers:

- **Card buffers (`VICTORY_CARD_BUFFERS`, on by default):** Eligible shader
  buffers `>= 256 KiB` (`VICTORY_CARD_BUFFERS_MIN_KB = 256`, overridden by
  `VICTORY_CARD_BUFFERS_MIN_MB`) up to `2048 MiB` (`VICTORY_CARD_BUFFERS_MB`)
  keep a VRAM copy + 16-byte-per-64 KiB hash twin under `VICTORY_CARD_VRAM_PERCENT = 92%`
  (`crates/victory-gpu/src/shadow.rs:180-258`).
- **Card windows (`VICTORY_CARD_WINDOWS_MB`, default `1536` MiB):** Moves
  read-mostly windows outside buffers onto device-local VRAM up to the same budget.
- **Bindless texture budget (`crates/victory-gpu/src/resource.rs:1370-1406`):**
  Once placed/filled texture bytes exceed `FLOOR = 2 GiB`, `room()` gates new
  uploads against `budget * room_percent() / 100` unless `VICTORY_TEXTURE_BUDGET_MB`
  overrides it. *Note:* `room_percent()` defaults to `82` in code (`resource.rs:1404`),
  even though its doc comment still reads `75`.
- **Opt-in copy bypasses (`rebar.rs`, `readback.rs`, `inplace.rs`):**
  - `VICTORY_REBAR` (`0` off by default; `1` windows only, `2` also small
    buffers; capped by `VICTORY_REBAR_MB`): requires `D3D12_FEATURE_D3D12_OPTIONS16`
    (`GPUUploadHeapSupported`) and explicitly excludes CPU readback buffers
    (`rebar.rs:25-28`, `ffa37e7`) to avoid uncached PCIe reads.
  - `VICTORY_COPY_BACK_ONLY_READBACK=1` (off by default) skips CPU copy-back on
    non-readback shader buffers.
  - `VICTORY_INPLACE` (`0` off; `1` private whole allocations, `2` also small
    buffers, capped by `VICTORY_INPLACE_MB = 4096`): uses `OpenExistingHeapFromAddress*`,
    locks pages into RAM, and **must not** combine level `2` with `VICTORY_LATE_WAIT`
    (`inplace.rs:77-81`).

---

## 2. Lighting stand-ins, Series S overlay, loader hooks, and RT boundary

**Source-confirmed and repository-reported.**

- **Lighting stand-ins (`crates/victory-gpu/src/shader_cache.rs:124-209`, `crates/victory-shader/src/emit/stand_in.rs`, `exposure.rs`):**
  - `VICTORY_AMBIENT_STUB=1` replaces diffuse GI temporal resolve hashes with a
    constant floor (`VICTORY_AMBIENT_STUB_VALUE`, default `0.1`).
  - `VICTORY_SHADOWMASK_STUB=1` replaces RT distant shadow passes (`VICTORY_SHADOWMASK_STUB_VALUE`,
    default `1.0`).
  - `VICTORY_EXPOSURE_STUB=1` replaces untranslatable auto-exposure shader
    `e622b718b6d9b057` (`VICTORY_EXPOSURE_STUB_VALUE`, default `0.18`).
- **Ray-tracing status (`docs/fork-ray-tracing.md`, `crates/victory-gpu/src/pipeline.rs:230-315`):**
  `VICTORY_BVH_BUILDER=1` and `VICTORY_BVH_QUARANTINE=1` remain off by default;
  while off, lighting stand-ins compensate for missing bounce/shadow light
  rather than reproducing console ray tracing.
- **Series S overlay & build `1.0.2.1` hooks (`crates/victory-gpu/src/over.rs`, `crates/victory-loader/src/hooks.rs`):**
  - `over.rs` composites the title's 4K interface plane over the 1080p game
    plane on Series S two-plane presents (note open `vitutu1/victory-perf`
    Issue #8 reporting that diagnostic `capture.rs` PNGs omit the HUD plane).
  - `hooks.rs` gates `VICTORY_NO_WATCHDOG=1` (disarms the ~30 s watchdog),
    `VICTORY_OWN_UPSCALER=1` (forces the title's own upscaler instead of FSR),
    and `VICTORY_RENDER_SCALE=<0.2..=1.0>` exclusively on `BUILD_1_0_2_1`
    (`b99f876b74d4a77f`). Open `vitutu1/victory-perf` Issue #2 reports startup
    exits (`0x80000003`) on RTX 5090 D v2 when `VICTORY_OWN_UPSCALER=1` is paired
    with `VICTORY_RENDER_SCALE=0.75` or `1.0`, whereas `0.5` reaches gameplay.

**Repository-reported baseline (`README.md`):** On Ryzen 7 9800X3D + RTX 4070
SUPER 12 GB (ReBAR enabled) + 64 GB RAM under `--console series-s` with
`VICTORY_OWN_UPSCALER=1`, `VICTORY_RENDER_SCALE=0.5`, and lighting stand-ins,
`README.md` self-reports `14 fps (71 ms)` in a prison corridor, `11 fps (95 ms)`
in a heist interior, `9 fps (115 ms)` in a cell, and `4.5–6.5 fps` in heavy
~10,000-draw scenes.

---

## 3. Comparative performance hypotheses across sibling forks

| Project / target | Baseline & focus | What differentiates it for comparison |
|---|---|---|
| `vitutu1/victory-perf` (`52a60947`) | Target fork. Coalesced `CARD_COMPARE` + inline ring, hoisted indirect clamps (mode 7), scoped/deduped barriers, `victory precompile --pipelines`, `DEFAULT_WAIT_MS = 12_000` | Already integrates Belobug quad reads (`caa4e44`), hoist mode 7 + walk cap (`ec5a20f`), and progressive `heap_buffers` warmup (`a7b2fb8`); open PRs #10, #11, #12 and Issues #2, #3, #8 guide next checks |
| `fxceless/victory-fork` | Direct ancestor (`310c6eb` in ancestry) + newer upstream work | Newer upstream commits add dedicated submit threads, copy queues (`VICTORY_QUEUE2`), and indirect gather experiments; earlier README notes withdrew a `6.0 -> 8.9 fps` comparison because scenes differed |
| `Belobug/VIctory-Belobug-AMD` | AMD RDNA-focused shader & pipeline fixes | Source of ported quad reads (`caa4e44`) and hoisted clamps (`ec5a20f`); also experiments with screen-pass pipeline worker separation |
| `ka1qux/BRVIctory-ka1qux` | Offline PSO prebuild & AMD/VRAM diagnostics | Separates `--used-pipelines` ($P_{\text{obs}}$ via `early-pipelines.txt`) from full pack prebuilding so graphics PSOs can also be replayed offline |
| `uiliannn/VItoriaBr` | Series S / cross-vendor stability & documentation | Incorporates `fxceless/victory-fork` async submit / copy queue sync and tracks RX 6500 XT 8 GB / 16 GB RAM regression triage |
| `Zettrox/ztrx` | Mixed NVIDIA/AMD tuning & VRAM admission experiments | Differs in bindless texture admission (`usage` vs. `budget` without buffer/RT reserves) and pipeline-wait experiments |

### Hypothesis 1 — First-use wait budgets vs. 16 GB RAM paging & graphics PSO coverage

- **Observed:** `vitutu1/victory-perf` Issue #3 (`97c3aa2` on 16 GB RAM + RTX 5070 12 GB)
  measured `2,226–6,373 ms/frame` medians with `7–14` frames lasting `6–24 s`
  while minimum available RAM dropped to `0.29–0.31 GiB` (even with `VICTORY_LIBRARY_TRIM=0`
  and `VICTORY_CARD_BUFFERS_MB=2048 -> 512`), because `DEFAULT_WAIT_MS = 12_000`
  keeps frames waiting up to `12 s` per kind when driver compilation pages heavily.
- **Target delta:** `victory precompile --pipelines` warms compute PSOs offline
  ($S_{\text{perm}}$ compute), whereas graphics/mesh PSOs still compile on first
  encounter or via `early-pipelines.txt` (`EARLY_MAX = 8192`).
- **Hypothesis & falsifier:** On 16 GB RAM systems, enabling `VICTORY_LIBRARY_TRIM=1`
  (adaptive `2 s` unlock below 2 GiB free RAM) plus lowering `VICTORY_PIPELINE_WAIT_MS`
  after the first run (or after an initial warm pass fills `host-pipelines.bin`)
  should eliminate multi-second frame spikes without losing one-off loading passes.
  **Falsifier:** If `host-pipelines.bin` hits (`loaded` high, `made` ~0) still show
  6–24 s frames and low available RAM, the stall is WDDM/working-set paging
  outside `pipeline/make.rs`. Extending `victory precompile` to record and replay
  a normalized graphics/mesh PSO manifest (the $P_{\text{obs}}$ pattern) would
  directly remove first-run graphics compilation bursts.

### Hypothesis 2 — Descriptor mirror contention and hybrid-core scheduling

- **Observed:** Recording threads contend on `HostHeap::mirror` (`made_lock` /
  `kept`) and `HostHeap::buffer_slots` when thousands of draws resolve
  descriptors each frame.
- **Target delta:** `victory-perf` (`52a60947`) already caps failed ring walks
  (`WALKS = 8`) and warms `heap_buffers` progressively (`8` buckets/pass). Two
  open upstream PRs target the remaining CPU overhead:
  - **PR #11 (`HostHeap::mirror` sharding):** replaces `made_lock` and the global
    `kept` mutex with 256 cache-line-aligned `SlotStripe` mutexes and 64-bucket
    `buffers` mutex sharding.
  - **PR #10 (`affinity.rs`):** uses `GetSystemCpuSetInformation` /
    `SetThreadSelectedCpuSets` to steer critical GPU frame threads onto P-cores
    on hybrid CPUs and optionally move DirectStorage workers onto E-cores
    (`VICTORY_DISK_CORES=slow`).
- **Hypothesis & falsifier:** On multi-threaded scenes (~10,000 draws/frame),
  sharded descriptor creation (PR #11) should cut `RecordNs` / `RecordCalls`
  wall time across the title's recording threads, and P-core pinning (PR #10)
  should reduce scheduling jitter on Intel/hybrid CPUs. **Falsifier:** Compare
  `copystat` recording time (`list calls by thread`) and frame-time P95/P99 on
  the exact same warm save; if `RecordNs` stays flat or E-core throttling slows
  asset streaming, contention lies elsewhere (submission lock or GPU-bound work).

### Hypothesis 3 — Direct-queue relay (`VICTORY_RELAY=2`) vs. async submit & copy queues

- **Observed:** `fxceless/victory-fork` and `uiliannn/VItoriaBr` later added
  opt-in dedicated submission threads (`VICTORY_ASYNC_SUBMIT`) and secondary
  copy queues (`VICTORY_QUEUE2`). `victory-perf` (`52a60947`) does not include
  those paths; instead it runs `VICTORY_RELAY=2` + `VICTORY_LATE_WAIT=3` +
  inline L0 `CARD_COMPARE` fetch directly on `self.queue`.
- **Target delta:** `host.rs` `card_clock_line()` directly separates GPU idle
  time (`nothing of ours queued`) into time while the submitting thread was
  preparing vs. time before the title submitted.
- **Hypothesis & falsifier:** Measure `card_clock_line()` and `submission_begins()` /
  `submission_ends()` lock times in a `--profile diag` run before importing an
  async submit worker or copy queue. **Falsifier:** If `nothing of ours queued
  while the submitting thread was working` is near `0 ms` and GPU busy % is
  already high, adding `ASYNC_SUBMIT` or `QUEUE2` will add cross-queue fence
  complexity without lifting FPS.

### Hypothesis 4 — Indirect clamp hoisting (mode 7) and barrier deduplication

- **Observed:** `VIctory-Belobug-AMD` reported lower GPU overhead when clamping
  indirect draw arguments in hoisted batches rather than interleaving compute
  dispatches between draws, and found inter-clamp ordering necessary to prevent
  mesh flicker.
- **Target delta:** `victory-perf` (`52a60947`) enables `VICTORY_HOIST_CLAMPS=7`
  (with budget-scoped UAV barrier from `ffa37e7`), `VICTORY_CLAMP_SCOPED=1`,
  `VICTORY_CLAMP_DEDUP=1`, `VICTORY_BARRIER_MERGE=1`, `VICTORY_INDIRECT_WINDOW=1`,
  and `VICTORY_INDIRECT_MERGE=1` by default.
- **Hypothesis & falsifier:** Toggle `VICTORY_HOIST_CLAMPS=0`, `VICTORY_CLAMP_DEDUP=0`,
  and `VICTORY_INDIRECT_MERGE=0` one variable at a time on the exact same
  ~10,000-draw scene while monitoring both `VICTORY_GPU_TIMING=1` and visual
  stability (no missing/flickering meshes). **Falsifier:** If mode 7 matches or
  improves `ms/frame` without flicker, keep it default; any dropped draws or
  flicker in `capture.rs` PNGs fails the correctness gate.

### Hypothesis 5 — VRAM residency headroom (8 GB vs. 12 GB+) and ReBAR boundaries

- **Observed:** `README.md` sets `VICTORY_CARD_BUFFERS_MB=2048` and `VICTORY_CARD_WINDOWS_MB`
  defaults to `1536`, calibrated on a 12 GB GPU (`RTX 4070 SUPER`). On 8 GB
  GPUs, default card buffers (`2048 MiB`) + card windows (`1536 MiB`) + texture
  uploads (`VICTORY_TEXTURE_ROOM_PERCENT=82%`) + render targets can exceed local
  VRAM budget (`DXGI_MEMORY_SEGMENT_GROUP_LOCAL`), triggering WDDM `VidMm`
  paging across PCIe. Meanwhile `rebar.rs` shows `VICTORY_REBAR` only helps on
  PCIe systems with >256 MiB BAR1 and never places CPU-read buffers in ReBAR.
- **Hypothesis & falsifier:** On 8 GB cards, scaling `VICTORY_CARD_BUFFERS_MB`
  down (e.g. `768–1024`), capping `VICTORY_CARD_WINDOWS_MB` (`256–512` or `0`),
  lowering `VICTORY_TEXTURE_ROOM_PERCENT` (`70–75`), and keeping `VICTORY_LIBRARY_TRIM=1`
  should prevent `card: ... MB in use of a budget of ... MB` from oversubscribing
  and spiking shared system memory. **Falsifier:** Track `VICTORY_CARD_ACCOUNTING=1`
  and `copystat` video-memory lines; if `shared with it in use` drops and
  frame-time stability improves without black/deferred textures, the gain is
  purely residency headroom.

### Hypothesis 6 — Wave64-on-Wave32 LDS & shader correctness before timing

- **Observed:** Console `WaveSize(64)` compute shaders running on 32-lane hosts
  (NVIDIA warps or Wave32 AMD modes) fail or require cross-half LDS synchronization
  when LDS accesses are not proven disjoint. Open `victory-perf` PR #12
  (`emit/lds.rs`) statically classifies `Fields::Ds` reads/writes to recover
  disjoint and lane-private Wave64 LDS shaders on Wave32 hosts.
- **Hypothesis & falsifier:** Differential unit tests + `victory translate --validate`
  and live visual verification must establish bit-level/visual equivalence for
  newly recovered shaders before benchmarking frame times under `VICTORY_WAVE_WIDE`,
  `VICTORY_WAVE_LOCAL`, or `VICTORY_GROUP_SYNC`.

---

## 4. Proposed investigation order

| Priority | Focus | Verification gate |
|---|---|---|
| 0 · Baseline & telemetry | Establish warm-cache baseline on matched checkpoints with `docs/diagnostics-and-benchmarking.md` | Separate DXIL/PSO hits, GPU busy % (`card_clock_line`), and VRAM/RAM usage; log results in `docs/test-log.md` |
| 1 · Residency & 16 GB / 8 GB tuning | Validate `VICTORY_LIBRARY_TRIM`, card buffer/window caps, and `VICTORY_PIPELINE_WAIT_MS` on constrained RAM/VRAM | Verify `0` paging spikes without dropped one-off loading passes (Issue #3) |
| 2 · Descriptor & CPU scaling | Evaluate sharded `HostHeap::mirror` (PR #11) and thread affinity (PR #10) | Measure `RecordNs`, lock waits, and P95/P99 frame times on matching heavy scenes |
| 3 · Shader Wave64 recovery & RT readiness | Validate disjoint LDS recovery (PR #12), build `1.0.2.1` scale crash (Issue #2), and RT builder gaps (`docs/fork-ray-tracing.md`) | Pass `cargo test --workspace`, `victory translate --validate`, and visual inspection before any timing claims |
| 4 · Offline graphics PSO manifest | Explore capturing normalized $P_{\text{obs}}$ graphics/mesh states for `victory precompile --pipelines` | Measure remaining cold `made` count after offline precompile on fresh driver caches |

---

## Primary reference index

- Grossman, M., & Andrews, J. (2021). *The Xbox Series X System Architecture*,
  IEEE Micro 41(2), 22–28.
- Microsoft Learn — *Direct3D 12 Multi-Engine Synchronization*:
  https://learn.microsoft.com/en-us/windows/win32/direct3d12/user-mode-heap-synchronization
- Microsoft Learn — *Fence-Based Resource Management*:
  https://learn.microsoft.com/en-us/windows/win32/direct3d12/fence-based-resource-management
- Microsoft Learn — *Readback Data Using Heaps*:
  https://learn.microsoft.com/en-us/windows/win32/direct3d12/readback-data-using-heaps
- Microsoft DirectX-Specs — *D3D12 GPU Upload Heaps*:
  https://microsoft.github.io/DirectX-Specs/d3d/D3D12GPUUploadHeaps.html
- Microsoft Learn — *`DXGI_QUERY_VIDEO_MEMORY_INFO`*:
  https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_4/ns-dxgi1_4-dxgi_query_video_memory_info
- Microsoft Learn — *Use DRED to diagnose GPU faults*:
  https://learn.microsoft.com/en-us/windows/win32/direct3d12/use-dred
- Microsoft Learn — *PIX Timing Captures*:
  https://learn.microsoft.com/en-us/windows/pix/timing-captures/timing-captures-overview
- Microsoft Learn — *Using GPUView*:
  https://learn.microsoft.com/en-us/windows-hardware/drivers/display/using-gpuview
- NVIDIA — *Nsight Graphics GPU Trace Activity*:
  https://docs.nvidia.com/nsight-graphics/UserGuide/gpu-trace-activity.html
- NVIDIA — *Optimizing DX12 Resource Uploads Using GPU Upload Heaps*:
  https://developer.nvidia.com/blog/optimizing-dx12-resource-uploads-to-the-gpu-using-gpu-upload-heaps/
- AMD GPUOpen — *Radeon GPU Profiler (RGP)*:
  https://gpuopen.com/rgp/
- For shader cache, pipeline library, and RDNA 2 / Wave32 details, see
  [docs/shader-pipeline-architecture.md](shader-pipeline-architecture.md).
