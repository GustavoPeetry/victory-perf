# Run & Bisection Test Log

Text-only log template for recording reproducible runs, controlled A/B
comparisons, session stability durations, and `git bisect` steps without
violating [docs/asset-policy.md](asset-policy.md).

- **Existing baseline figures:** Maintainer-reported numbers for the Ryzen 7
  9800X3D + RTX 4070 SUPER 12 GB setup live in [README.md](../README.md#status--measured-results).
- **Measurement procedure:** Follow [docs/diagnostics-and-benchmarking.md](diagnostics-and-benchmarking.md)
  before adding rows here.

---

## Recording rules

1. **Text only:** Never commit screenshots, frame dumps, shader dumps, memory
   captures, or commercial title names (`docs/asset-policy.md`). Describe scenes
   in generic terms (e.g. *intro*, *cell checkpoint*, *corridor*, *heavy 10k-draw exterior*).
2. **Pin exact inputs:** Record the full or 7+ char `Commit SHA`, hardware/driver,
   RAM + pagefile, `--console` profile, non-default `VICTORY_*` overrides, and
   cache state (`Cold`, `DXIL-warm`, or `PSO-warm` verified via logs).
3. **Separate phases & report stability:** Log steady-state 3D frame timings
   (`fps`, `ms/frame`, `preparing`, `on the GPU`, `copying back`, and `GPU busy %`
   from `card_clock_line`) separately from loading screens, and always record the
   **session duration** and **stop reason** (manual exit, watchdog, device
   removal, crash) plus any visual/correctness regressions.

---

## Run Log

| Date (UTC) | Commit SHA | Hardware / VRAM / RAM / Driver | Console & Key Switches | Cache State | Scene / Checkpoint | Median FPS & ms/frame (`prep / gpu / back / GPU busy %`) | Stability Duration & Stop Reason | Visual / Correctness Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| *YYYY-MM-DD* | *`<sha>`* | *`<CPU> / <GPU> (<VRAM>) / <RAM> / <driver>`* | *`series-s`, overrides...* | *`Cold` / `DXIL-warm` / `PSO-warm`* | *`<checkpoint>, ~<draws> draws`* | *`<fps> fps (<ms> ms: <p>/<g>/<b> ms, <busy>% busy)`* | *`<min> min — <exit reason>`* | *`<HUD, lighting, flicker, missing meshes>`* |

---

## Bisection Notes

Use the short form below when isolating regressions across commits or single switches:

```text
Symptom:       <what regressed and how it is observed>
Good rev:      <commit SHA or baseline switch set>
Bad rev:       <commit SHA or candidate switch set>
Controls:      <adapter, driver, console profile, checkpoint, cache state>
Steps:
  - <rev / switch> -> <outcome, median fps/ms, duration, stop reason>
Isolated to:   <commit SHA or single VICTORY_* switch>
```
