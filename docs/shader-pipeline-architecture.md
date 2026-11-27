# Shader Compilation, Pipeline Precompilation, and Cache Architecture

This note documents how `victory-perf` translates Xbox Series X|S console shaders,
precompiles and caches host Direct3D 12 pipelines, and isolates compilation
stalls from runtime memory and synchronization costs.

All repository behavior below is verified against `main` at `52a60947`.
Background on console hardware, compiler backends, and PC pipeline caching is
grounded in the primary vendor specifications and engineering literature listed
in [Primary references](#primary-references). Closed-source internals (PC GPU
driver user-mode compilers and NDA `d3d12_xs.dll` paths) are marked as outside
public verification.

---

## 1. Console AOT vs. PC JIT: why even AMD-on-AMD requires lifting and recompilation

On Xbox Series S and Series X, shader compilation happens ahead of time (AOT)
on the studio's build machines. Both consoles share one target GPU
microarchitecture — AMD RDNA 2 (`GFX10.3`) — and each title runs on a versioned
Game OS paired with its matching `d3d12_xs.dll` runtime. Because the ISA and
graphics ABI are fixed for the life of the title build, GDK tools
(`DxCompiler_xs.dll` / `DXC.exe`, and experimentally the Mesa ACO-based
`aco_xbox.dll` / `xdxc.exe` backend in GDK 2610) emit native RDNA 2 microcode
(and optionally offline-serialized PSOs) directly into the read-only `G:\`
package. Console OS updates do not invalidate those binaries.

Running the same title on Windows PC — even on an AMD Radeon GPU such as RDNA 3
or RDNA 4 — inverts that lifecycle for three structural reasons:

1. **ISA and microarchitecture divergence:** GPU ISAs are not binary-compatible
   across vendors (AMD vs. NVIDIA vs. Intel) or across generations of the same
   vendor (RDNA 2 vs. RDNA 3 vs. RDNA 4). The host D3D12 driver does not accept
   console RDNA 2 shader containers directly; it expects portable SM 6.6 DXIL.
2. **State-coupled host PSO compilation:** On D3D12, `ID3D12PipelineState`
   couples shader bytecode with the Root Signature, input layout, render-target
   formats, blend, depth-stencil, and rasterizer state. On RDNA hardware,
   vertex-attribute fetch prologs and pixel-export packing epilogs depend on
   those fixed states, so translating shaders into DXIL alone does not produce
   ready-to-bind host pipelines.
3. **WDDM driver lifecycle and memory split:** Windows drivers compile DXIL to
   native host ISA (RDNA 3 ISA, NVIDIA SASS, etc.) JIT inside the vendor
   user-mode driver (UMD), under WDDM (`dxgkrnl.sys`, `VidMm`, `VidSch`, TDR),
   with separate host RAM and GPU VRAM across PCIe instead of the console's
   coherent unified GDDR6 memory.

As a result, `victory-perf` runs a two-stage pipeline:

```text
Console container (GFX10.3 / RDNA 2 microcode in title files or memory)
  │
  ├─► Stage 1: Reverse lifting (victory-shader)
  │     GFX10 → SSA IR → structurization → resource tracking → DXIL (SM 6.6)
  │     Persisted in: <cache_dir>\shaders  (crates/victory-gpu/src/shader_cache.rs)
  │     Pre-warmed by: victory precompile  (apps/victory/src/precompile.rs)
  │
  └─► Stage 2: Host PSO compilation (host D3D12 UMD + ID3D12PipelineLibrary1)
        DXIL + Root Signature + GraphicsState → host native ISA + PSO object
        Persisted in: <cache_dir>\pipelines\host-pipelines.bin
        Used-set list: <cache_dir>\pipelines\early-pipelines.txt
        Pre-warmed (compute only) by: victory precompile --pipelines
        (crates/victory-gpu/src/pipeline/{library,make,host}.rs)
```

---

## 2. Two-layer persistence model and cache-key audit

Keeping the DXIL translation cache separate from the D3D12 Pipeline Library
mirrors standard multi-layer caching practice: portable IR (DXIL) survives
host driver updates, while native machine-code PSOs are tightly coupled to the
adapter and driver ABI.

| Property | Layer 1: DXIL translation cache | Layer 2: Host PSO library & early list |
|---|---|---|
| Location | `<cache_dir>\shaders` | `<cache_dir>\pipelines\host-pipelines.bin` + `early-pipelines.txt` |
| Stored payload | Serialized `HostShader` (`VSH1`: DXIL + binding metadata, no cached failures) | D3D12 `ID3D12PipelineLibrary1` blob of driver-compiled PSOs, plus up to `EARLY_MAX` (`8192`) title pipeline keys used in prior runs |
| Implementation | `crates/victory-gpu/src/shader_cache.rs`, `crates/victory-shader/src/lib.rs`, `crates/victory-pipeline/src/cache.rs` | `crates/victory-gpu/src/pipeline/library.rs`, `crates/victory-gpu/src/pipeline/make.rs` |
| Identity key | `CacheKey::new` (`"shader.dxil"` v1) over `[build() (32 bytes), container_bytes]` | Library entry: `library::name(kind, root, code, state)` (32-hex BLAKE3). Early key: `Pending::key` (32-hex BLAKE3 of shaders + `GraphicsState::describe()`) |
| Survives rebuilds outside `victory-shader` / `shader_cache.rs`? | **Yes** (provided `Cargo.toml` pkg version, `diagnostics` feature, switches, and GPU facts match) | **Yes** (unless translated DXIL, root signature, or host graphics state changes) |
| Survives GPU driver updates (same GPU)? | **Yes**, provided `host_max_wave()` and `view_heap_limit()` are unchanged | **No** (D3D12 rejects the blob on driver mismatch and starts an empty library) |
| Survives GPU swap? | Only if `host_max_wave()` and `view_heap_limit()` match | **No** (adapter mismatch) |

### Layer 1 details (`crates/victory-gpu/src/shader_cache.rs`)

`victory-pipeline`'s content-addressed `Cache` (`crates/victory-pipeline/src/cache.rs`)
writes entries atomically via write-then-rename (`tempfile_in` + `fs::rename`)
and refuses any path inside the repository (`victory_core::ensure_outside_repo`).
For each shader container, `shader_cache::cached()` passes `[build(), container]`
to converter `"shader.dxil"` version `1`.

`build()` computes a 32-byte BLAKE3 digest over:

- every source file in `victory_shader::sources()` (`crates/victory-shader/src/lib.rs`,
  37 `.rs` files guarded by `source_tests::every_source_is_listed`) plus
  `include_str!("shader_cache.rs")`;
- `env!("CARGO_PKG_VERSION")` and `cfg!(feature = "diagnostics")`;
- runtime/device parameters: `loop_limit()` (`VICTORY_SHADER_LOOP_LIMIT` in
  `diagnostics` builds, default `1 << 20`), `view_heap_limit()`,
  `host_max_wave()`, `group_sync()` (`VICTORY_GROUP_SYNC`, default `2`), and
  `shader_opt()` (`VICTORY_SHADER_OPT`, default all passes);
- four explicit environment switches: `VICTORY_WAVE_WIDE`,
  `VICTORY_WAVE_DEEP_SKIPS`, `VICTORY_WAVE_OWN_BITS`, and `VICTORY_F16_NEAREST`.

Corrupted entries fail `decode()` (`VSH1`), are removed from disk, and are
retranslated; translation errors (`ShaderError::Unsupported`, etc.) are **not**
written to `<cache_dir>\shaders`.

#### Verified cache-key caveats at `52a60947`

Auditing `shader_cache.rs`, `victory-shader/src/emit/`, and `precompile.rs`
against the code reveals five concrete caveats that affect reuse and troubleshooting:

1. **Stale executable-hash comments:** Module doc comments in
   `crates/victory-gpu/src/shader_cache.rs`, `crates/victory-gpu/src/precompile.rs`,
   and `apps/victory/src/precompile.rs` still state that the cache keys on the
   BLAKE3 hash of `victory.exe`. In reality, `build()` hashes `victory_shader::sources()`
   and `shader_cache.rs`, **not** `victory.exe`. Rebuilding `victory.exe` after
   changes solely in `victory-gpu` (outside `shader_cache.rs`), `victory-hle`, or
   `victory-loader` still reuses `<cache_dir>\shaders` in the same build profile.
2. **Release vs. `diag` vs. debug profiles:** `build()` hashes
   `cfg!(feature = "diagnostics")`, so `--release` and `--profile diag --features diagnostics`
   use separate cache keys cleanly. However, `cfg!(debug_assertions)` is not hashed
   into `build()`.
3. **Five translator switches omitted from `build()`:** `crates/victory-shader/src/emit/`
   reads `VICTORY_WAVE_LOCAL`, `VICTORY_FIRST_HALF`, `VICTORY_FIRSTLANE_OWN`,
   `VICTORY_MESH_OUTPUT_STRICT`, and `VICTORY_STRICT_JOINS`, which are **not**
   currently included in `shader_cache::build()`. Toggling any of those five
   variables without clearing `<cache_dir>\shaders` (or isolating `VICTORY_CACHE_DIR`)
   can serve stale translations.
4. **Device initialization order (`OnceLock`):** `build()` memoizes its digest in a
   `OnceLock` on first lookup and calls `probe_view_limit()`, which probes the heap
   limit only if `Host` is already initialized (`set_host_max_wave()` happens in
   `Host::new()`). `victory precompile` calls `init_host()` up front; running
   without `--pipelines` when no D3D12 device is available leaves `host_max_wave()`
   at `0`, producing a different key than a live GPU run.
5. **Stand-in and exposure stubs bypass `<cache_dir>\shaders`:** When
   `VICTORY_AMBIENT_STUB=1`, `VICTORY_SHADOWMASK_STUB=1`, or
   `VICTORY_EXPOSURE_STUB=1` replaces a matching hash in `host_shader_cached()`,
   the synthetic stand-in is emitted directly without reading or writing the disk
   shader cache (`cached = false`).

### Layer 2 details (`crates/victory-gpu/src/pipeline/library.rs`)

PC pipeline cache engineering highlights three failure modes for large PSO
caches: lock serialization across worker threads, excessive RAM retention from
warm caches (which reach ~1.35–1.49 GB on NVIDIA per comments in `library.rs`),
and torn or corrupted files across crashes and concurrent saves. `library.rs`
addresses each explicitly:

- **Striped load locking (`same_name()`):** D3D12 documents
  `ID3D12PipelineLibrary` as thread-safe except for concurrent loads of the
  same pipeline name. Instead of a global lock (which serialized 8 workers in
  earlier profiling, leaving them waiting in 87–96 of 110 samples), `same_name()`
  hashes the pipeline name across 64 mutexes (`VICTORY_LIBRARY_LOCK=1` restores
  a single lock for diagnostics).
- **Memory-mapped read/write and adaptive working-set trimming:** By default,
  `host-pipelines.bin` is mapped read-only via `MapViewOfFile` (`Blob::Mapped`,
  disabled with `VICTORY_LIBRARY_MAP=0`) and serialized through a writable
  mapping so saves avoid allocating a duplicate ~1+ GB heap buffer. Once loads
  quiet down (`ram_idle_ms()` in `crates/victory-gpu/src/host.rs`: `10 s` at
  `>= 4 GiB` available RAM, `5 s` at `2–4 GiB`, `2 s` below `2 GiB`), `trim()`
  releases resident pages from the process working set via `VirtualUnlock`
  (`VICTORY_LIBRARY_TRIM=0` disables trimming).
- **Safe in-place replacement (`holds()`, `SAVE_EVERY`, `host-pipelines.next`):**
  Keeping the mapped file handle open (`holds()`, disabled with
  `VICTORY_LIBRARY_HANDLE=0`) prevents Windows from zero-filling evicted pages
  when a save renames `host-pipelines.tmp` over the mapped `host-pipelines.bin`
  — a failure mode that otherwise hands zeroed PSO bytes to the driver and triggers
  `DXGI_ERROR_DRIVER_INTERNAL_ERROR`. Saves run at most once every `20 s`
  (`SAVE_EVERY`) plus on shutdown, and fall back to `host-pipelines.next`
  (promoted on next startup) if the OS refuses an in-place rename.

---

## 3. Offline precompilation framed by AOT compilation sets

Graphics compiler literature distinguishes five sets when evaluating any
ahead-of-time compilation strategy:

1. $S_{\text{static}}$ — static source shaders;
2. $S_{\text{perm}}$ — all known compiled shader permutations shipped in the title;
3. $P_{\text{obs}}$ — the subset of complete pipelines empirically observed in
   prior runs;
4. $P_{\text{theor}}$ — the full Cartesian product of shader permutations and
   fixed states (combinatorially intractable);
5. $P_{\text{real}}$ — the exact set of pipelines that an arbitrary future play
   session will actually request (undecidable in general because it depends on
   runtime scene state and player choices).

The offline commands in `apps/victory/src/translate.rs`, `apps/victory/src/shader.rs`,
and `apps/victory/src/precompile.rs` map onto these sets as follows:

| Command | Target set | Cache layer populated | Notes and trade-offs |
|---|---|---|---|
| `victory translate` (`--validate`, `--pipelines`, `--list`, `--jobs`) | $S_{\text{perm}}$ (census only) | None | Reports translator / validator / diagnostic PSO coverage without writing `<cache_dir>\shaders` or `host-pipelines.bin` |
| `victory precompile` (`--only`, `--limit`, `--jobs`) | $S_{\text{perm}}$ (accessible files $\le 2$ GiB) | `<cache_dir>\shaders` | Scans raw DXBC containers sliced to header size (`0x18..0x1c` in `apps/victory/src/shader.rs`) and SGD5 records, deduplicates by full BLAKE3, probes the host D3D12 device, and populates DXIL translations |
| `victory precompile --pipelines` (`--pipeline-jobs`, default `4`) | $S_{\text{perm}}$ + offline compute PSOs | `<cache_dir>\shaders` + `<cache_dir>\pipelines\host-pipelines.bin` | Also compiles every translated **compute** shader through `pipeline::host::precompile_compute` (bounded by `--pipeline-jobs` to cap driver compile RAM), preserving any existing `early-pipelines.txt` |

Why offline precompilation covers **compute PSOs** today — and what still
compiles at runtime:

- **Compute PSOs need only the shader and its derived root signature:** `precompile_compute()`
  derives the root signature directly from `HostShader` bindings and calls
  `CreateComputePipelineState` via `library::pipeline_with_status()`, which
  matches the exact `library::name("compute", &root_blob, &[&shader.dxil], "")`
  used during gameplay.
- **Graphics and mesh PSOs depend on runtime fixed state:** `HostPipeline::new()`
  couples translated VS/PS/MS/GS bytecode with runtime `GraphicsState`
  (render-target formats, DSV format, blend state, rasterizer state, depth/stencil
  state, sample counts, and input layout or topology) that only the title supplies
  when creating pipelines. Graphics/mesh PSOs therefore compile on first
  encounter (or early at title pipeline creation from `early-pipelines.txt`,
  representing $P_{\text{obs}}$ up to `8192` entries) and persist in
  `host-pipelines.bin` for subsequent runs.
- **Containers unseen on disk or packed differently:** Shaders compressed inside
  unparsed archives or generated/modified in memory still translate on first use.
- **Ecosystem context:** Newer PC APIs mitigate monolithic PSO explosion via
  pipeline decoupling (`VK_EXT_graphics_pipeline_library`, `VK_EXT_shader_object`,
  `VK_KHR_pipeline_binary`, and D3D12 Partial Graphics Programs) or
  cloud-precompiled driver databases (DirectX Advanced Shader Delivery PSDBs).
  `victory-perf` currently targets standard D3D12 monolithic PSOs backed by
  `ID3D12PipelineLibrary1`, `early-pipelines.txt`, and offline compute
  precompilation.

---

## 4. Runtime worker pool, wait budgets, exclusions, and hardware divergences

### Background worker pool and frame-budget gating (`crates/victory-gpu/src/pipeline/make.rs`)

Because D3D12 drivers do not maintain an application-facing compile thread pool,
`pipeline/make.rs` manages background compilation with deduplication and bounded
concurrency so compilation neither stalls recording threads indefinitely nor
triggers the title's 30 s GPU watchdog:

- **Worker count and queue bands:** `workers()` spawns `(cores / 2).clamp(2, 8)`
  threads at `THREAD_PRIORITY_BELOW_NORMAL`. `Queue` prioritizes `REQUESTED`
  (needed by an active draw/dispatch) ahead of `EARLY` (speculatively queued
  from `early-pipelines.txt` at pipeline creation).
- **Per-frame wait budget (`DEFAULT_WAIT_MS = 12_000`):** In `victory-perf`,
  `VICTORY_PIPELINE_WAIT_MS` defaults to `12_000 ms` (`12 s`) per frame and per
  pipeline kind (graphics vs. compute) — putting **picture completeness ahead of
  frame rate** while keeping the worst-case sum of both kinds (`24 s`) under
  `TITLE_WATCHDOG` (`30 s`). Setting `VICTORY_PIPELINE_WAIT_MS=500` restores the
  shorter 500 ms cap (dropping draws whose PSOs take longer).
- **Shared wait window (`VICTORY_PIPELINE_WAIT_SHARED`):** Defaults to
  `Sharing::Window`, where concurrent recording threads share one elapsed wait
  window per frame so one thread waiting does not immediately force other
  threads to drop draws (`reserve` and `0` preserve prior modes).
- **Loading-phase grace window (`VICTORY_PIPELINE_GATE_S = 120`):** When a base
  budget lower than `4 s` (graphics) or `12 s` (compute) is configured,
  `frame_budget()` still grants up to `LOADING_FRAME` (`4 s`) / `LOADING_FRAME_COMPUTE`
  (`12 s`) per frame until the cumulative loading allowance (`120 s` per kind;
  `0` disables) is exhausted, protecting one-off world-unpacking compute passes.
- **Slow-compile cutoff (`held_too_long()`):** When `VICTORY_PIPELINE_WAIT_MS`
  is set below `10 s` (compute) or `3 s` (graphics), frames stop waiting once a
  driver compile exceeds that threshold (with the default `12_000 ms` budget,
  `base < most` is false, so `wait_budget()` governs instead — a behavior that
  matters on 16 GB RAM systems experiencing heavy driver paging, as observed in
  `vitutu1/victory-perf` Issue #3).

### Selective pipeline exclusions (`make.rs` & `pipeline.rs`)

| Switch | Default | Target & effect |
|---|---|---|
| `VICTORY_RT_RESOLVE=0` | Made by default | When `VICTORY_BVH_BUILDER` is not `1`, omits `CS_RaytracingResolve` pipelines (about 1,000 material resolve compute programs, each taking 5–9 s on AMD's compiler in comments from `make.rs`), freeing worker queue time when BVH building is off |
| `VICTORY_ACTIVATE_RAYS=0` | Made by default | Omits `CS_ScreenProbes_ActivateRays` (`39f0e9aeb156a61c`, 66,501 instructions / 9,948 branches; `make.rs` notes 85 s / 1.3 GB with `shader_opt` enabled vs. 47 GB after 8 min unoptimized on RX 7800 XT) |
| `VICTORY_ACCC=0` | Made by default | Refuses DXIL-less compute containers (`ACCC`, used for title upscaler passes) with `E_NOTIMPL` in `crates/victory-gpu/src/pipeline.rs` so the title falls back |

### Host-GPU hardware divergences relevant to translation

Even after lifting GFX10.3 to SM 6.6 DXIL, several hardware differences between
the console's RDNA 2 GPU and desktop GPUs directly affect shader translation
and cache compatibility:

- **Wavefront width (`Wave32` vs. `Wave64`, `host_max_wave()`):** RDNA 2/3
  supports both native Wave32 (1 cycle per SIMD32) and Wave64 (2 cycles),
  whereas NVIDIA GPUs execute 32-lane warps exclusively. Console Wave64 compute
  shaders must be lowered differently when `WaveLaneCountMax == 32`
  (`crates/victory-gpu/src/host.rs` calls `set_host_max_wave()`). That is why
  `host_max_wave()` is part of `shader_cache::build()`. Switches such as
  `VICTORY_WAVE_WIDE`, `VICTORY_WAVE_LOCAL`, `VICTORY_WAVE_OWN_BITS`, and
  `VICTORY_GROUP_SYNC` (default `2`) govern Wave64-on-Wave32 lowering and
  barrier rendezvous semantics.
- **Pixel 2×2 quads and helper lanes:** Pixel shaders execute in 2×2 quads so
  derivatives (`ddx`/`ddy`) and mip selection operate across triangle edges
  where neighboring lanes are inactive helper pixels. On AMD hardware,
  wave-level cross-lane reads return `0` from helper lanes; commit `caa4e44`
  (ported from `VIctory-Belobug-AMD`) updates quad reads, helper `readfirstlane`,
  row interpolation, and `ACCC` handling in `crates/victory-shader/src/emit/lower.rs`.
- **Shader-visible descriptor heap limits (`view_heap_limit()`):** D3D12
  resource binding tiers 1–2 cap shader-visible CBV/SRV/UAV heaps at
  `PORTABLE_VIEWS = 1_000_000` (NVIDIA), whereas AMD tier 3 hardware supports
  `UNCAPPED_VIEWS = 1 << 21` (`2,097,152`). `probe_view_limit()`
  (`crates/victory-gpu/src/descriptor/host.rs`) probes this once and clamps
  out-of-bounds view indices to the final null slot, entering `shader_cache::build()`
  so translations built for one cap are never served on the other.
- **VGPR pressure, cleanup passes, and loop bounds:** In RDNA 2, each SIMD32 has
  1,024 physical 32-bit VGPRs shared across up to 16 wavefronts (allocated in
  granules of 16 VGPRs in Wave32 or 8 in Wave64) plus 128 KiB LDS per WGP.
  Running `emit::shader_opt` cleanup passes reduces IR bloat before the host
  driver's register allocator runs. Translated loops carry a per-lane iteration
  cap of `DEFAULT_LOOP_LIMIT = 1 << 20` (`VICTORY_SHADER_LOOP_LIMIT` overrides in
  `diagnostics` builds).

---

## 5. Verifying both cache layers in logs

Check the two layers separately in a run's `stderr` log:

### Layer 1: DXIL translation cache (`<cache_dir>\shaders`)

| Log message (level) | Fields | Source & how to read it |
|---|---|---|
| `shaders from the cache and translated so far` (`INFO`, emitted on power-of-two presented frames `>= 256`: `256`, `512`, `1024`, ...) | `hits`, `misses` | Cumulative counts since startup (`crates/victory-gpu/src/queue.rs`, `shader_cache::counts()`). After `victory precompile`, `hits` should dominate and `misses` should stay low |
| `host wave lanes` (`INFO`, at D3D12 device creation) | `min`, `max` | `crates/victory-gpu/src/host.rs`; `max` enters `shader_cache::build()` as `host_max_wave()` |
| `host view heaps hold <views> views; view indices are clamped to the null slot at the end from the first shader on` (`INFO`) | `views` | `crates/victory-gpu/src/descriptor/host.rs`; `views - 1` enters `shader_cache::build()` as `view_heap_limit()` |
| `shader translated` (`INFO`, once per distinct shader in the run) | `stage`, `hash`, `user_data`, `samplers` | `crates/victory-gpu/src/pipeline.rs`. **Does not indicate a cache miss:** logged whenever `Shader::translated()` succeeds, whether served from `<cache_dir>\shaders` (`Hit`) or translated on demand |
| `shader from the cache` / `shader translated and cached` / `unreadable shader cache entry; translating again` (`DEBUG`, per shader) | `path` | `crates/victory-gpu/src/shader_cache.rs`. Appears **only** in builds compiled with `--features diagnostics` (e.g., `VICTORY_LOG=victory_gpu::shader_cache=debug,info`), because standard `--release` builds cap `tracing` at `INFO` |

Before launching the title, running `victory precompile` (or `victory precompile --pipelines`)
a second time confirms disk-cache reuse directly via its summary lines (`0 translated`,
remainder under `already cached`, and for compute PSOs `0 compiled`, remainder under
`already in library`).

### Layer 2: Host PSO library (`<cache_dir>\pipelines`)

| Log message (level) | Fields | Source & how to read it |
|---|---|---|
| `host pipeline library loaded` (`INFO`, opening `host-pipelines.bin`) | `bytes`, `mapped` | `crates/victory-gpu/src/pipeline/library.rs`. Confirms the driver accepted `host-pipelines.bin` (`mapped=true` when `MapViewOfFile` succeeded). First runs log `no host pipeline library yet; starting an empty one`; driver/GPU changes or corruption log `host pipeline library rejected; starting an empty one` (`error`) |
| `title pipelines to start early` (`INFO`, reading `early-pipelines.txt`) | `pipelines` | `crates/victory-gpu/src/pipeline/library.rs`. Count of prior-run pipeline keys queued for early creation |
| `host pipelines all made` (`INFO`, when workers drain a batch with peak `>= 10`) | `made`, `loaded`, `started_early` | `crates/victory-gpu/src/pipeline/make.rs`. Cumulative run counters: `made` = newly driver-compiled PSOs (library miss); `loaded` = PSOs loaded from `host-pipelines.bin` (library hit); `started_early` = title pipelines queued at creation from `early-pipelines.txt` |
| `host pipeline library saved` (`INFO`, at most every `20 s` when changed, plus shutdown) | `bytes`, `early`, `made`, `loaded`, `started_early` | `crates/victory-gpu/src/pipeline/library.rs`. Same cumulative counters plus `bytes` serialized to `host-pipelines.bin` (`0` if only `early-pipelines.txt` changed) and total `early` entries |
| `host pipeline wait budget per frame, per kind` / `host pipeline loading allowance per kind` (`INFO`) | `millis` / `seconds` | `crates/victory-gpu/src/pipeline/make.rs`. Confirms active `VICTORY_PIPELINE_WAIT_MS` and `VICTORY_PIPELINE_GATE_S` |

After running `victory precompile --pipelines` on the same GPU and driver, the
first live run should show `host pipeline library loaded` with compute PSOs
already hitting `loaded` (while graphics/mesh PSOs contribute to `made` until the
second run, when `loaded` and `started_early` dominate).

---

## Primary references

1. Grossman, M., & Andrews, J. (2021). *The Xbox Series X System Architecture*,
   IEEE Micro 41(2), 22–28; and Hot Chips 32 presentation (Aug. 2020):
   https://hc32.hotchips.org/assets/program/conference/day1/HotChips2020_GPU_Microsoft_Jeff_Andrews_v2.pdf
2. Microsoft Learn — *Xbox GDK Samples: HlslCompile* (`DxCompiler_xs.dll`,
   `aco_xbox.dll`, `AcoOfflineSerializePipelineStateXEx`,
   `DeserializeGraphicsPipelineStateX`):
   https://learn.microsoft.com/en-us/samples/microsoft/xbox-gdk-samples/hlslcompile/
3. Microsoft Learn — *Introduction to the Microsoft Game Development Kit
   (GDK)* (Game OS versioning, `D3D12.x`, `XMemAlloc`):
   https://learn.microsoft.com/en-us/gaming/gdk/docs/gdk-dev/intro/introduction?view=gdk-2604
4. Microsoft Learn — *Local Storage on Xbox Consoles* (`G:\`, `T:\`, PLS,
   XR-133):
   https://learn.microsoft.com/en-us/gaming/gdk/docs/features/console/storage/local-storage?view=gdk-2610
5. Microsoft Learn — *Windows Display Driver Model (WDDM) Architecture*:
   https://learn.microsoft.com/en-us/windows-hardware/drivers/display/windows-vista-and-later-display-driver-model-architecture
6. Microsoft Learn — *Direct3D 12 API Reference*
   (`ID3D12Device1::CreatePipelineLibrary`,
   `D3D12_GRAPHICS_PIPELINE_STATE_DESC`, Root Signatures):
   https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device1-createpipelinelibrary
7. Microsoft DirectX-Specs — *D3D12 Shader Cache APIs* and *D3D Shader Cache
   Registration (D3DSCR) API*:
   https://microsoft.github.io/DirectX-Specs/d3d/ShaderCache.html and
   https://microsoft.github.io/DirectX-Specs/d3d/ShaderCacheRegistrationAPI.html
8. Microsoft DirectX Developer Blog — *DirectX Advanced Shader Delivery* and
   *Partial Graphics Programs*:
   https://devblogs.microsoft.com/directx/introducing-advanced-shader-delivery/
   and https://devblogs.microsoft.com/directx/partial-graphics-programs/
9. AMD — *"RDNA 2" Instruction Set Architecture: Reference Guide* (Doc. ID
   70648):
   https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
10. AMD GPUOpen — *RDNA Performance Guide* and *Occupancy Explained*:
    https://gpuopen.com/learn/rdna-performance-guide/ and
    https://gpuopen.com/learn/occupancy-explained/
11. LLVM Project — *User Guide for AMDGPU Backend* (`gfx1030` VGPR allocation
    granularity and occupancy):
    https://llvm.org/docs/AMDGPUUsage.html
12. Mesa 3D — *RADV Driver Documentation* (NIR / ACO compiler architecture):
    https://docs.mesa3d.org/drivers/radv.html
13. Khronos Group — *Reducing Draw Time Hitching with
    `VK_EXT_graphics_pipeline_library`* and *Bringing Explicit Pipeline Caching
    Control to Vulkan (`VK_KHR_pipeline_binary`)*:
    https://www.khronos.org/blog/reducing-draw-time-hitching-with-vk-ext-graphics-pipeline-library
    and
    https://www.khronos.org/blog/bringing-explicit-pipeline-caching-control-to-vulkan
14. Epic Games — *Game Engines and Shader Stuttering: Unreal Engine's Solution
    to the Problem* and *PSO Precaching*:
    https://www.unrealengine.com/tech-blog/game-engines-and-shader-stuttering-unreal-engines-solution-to-the-problem
    and
    https://dev.epicgames.com/documentation/unreal-engine/pso-precaching-for-unreal-engine
15. ValveSoftware/Fossilize — *README* and parallel replay memory-usage report
    (#312):
    https://github.com/ValveSoftware/Fossilize/blob/master/README.md and
    https://github.com/ValveSoftware/Fossilize/issues/312
16. Roblox Engineering — *Creating a Robust Pipeline Cache with Vulkan*
    (header validation and atomic persistence):
    https://about.roblox.com/newsroom/2021/02/creating-a-robust-pipeline-cache-with-vulkan
17. Pettineo, M. (TheRealMJP) — *Shader Permutations Part 1 & Part 2*:
    https://therealmjp.github.io/posts/shader-permutations-part1/ and
    https://therealmjp.github.io/posts/shader-permutations-part2/
