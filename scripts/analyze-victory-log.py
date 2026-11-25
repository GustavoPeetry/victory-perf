#!/usr/bin/env python3
"""Analyze and compare Victory runtime logs for RTX 2060 SUPER A/B runs.

Distinguishes periodic aggregated windows (`fps, ms/frame`) from per-frame
telemetry when present, attributes >=1s spikes using overlapping stage and
card-clock signals without double-counting multi-thread sums, and classifies
termination causes cleanly.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mK]")
TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)\s+(.*)$")

FRAME_RE = re.compile(
    r"^(\S+Z)\s+\S+\s+victory_gpu::host:\s+"
    r"([0-9.]+)\s+fps,\s+([0-9.]+)\s+ms/frame\s+"
    r"\(([0-9.]+)\s+preparing:\s+copies\s+([0-9.]+)\s+ms\s+([0-9.]+)\s+MB,\s+"
    r"textures\s+([0-9.]+)\s+ms\s+([0-9.]+)\s+MB;\s+"
    r"([0-9.]+)\s+on the GPU;\s+"
    r"([0-9.]+)\s+copying back\s+([0-9.]+)\s+MB\),\s+"
    r"([0-9.]+)\s+draws/frame,\s+"
    r"dropped\s+([0-9.]+)\s+without a host pipeline\s+\((\d+)\s+being made\)\s+and\s+([0-9.]+)\s+indirect;\s+"
    r"heap buffers\s+([0-9.]+)\s+passes\s+([0-9.]+)\s+ms.*?"
    r"textures picked by index:\s+(\d+)\s+uploaded so far\s+\((\d+)\s+MB\),\s+(\d+)\s+waiting,\s+(\d+)\s+passed over unmapped"
)

CLOCK_RE = re.compile(
    r"the card's clock per frame:\s+nothing of ours queued\s+([0-9.]+)\s+ms.*?"
    r"([0-9.]+)\s+ms of that while the submitting thread was working"
)

CPU_CMP_RE = re.compile(
    r"shader writes looked for on the CPU per frame:\s+"
    r"([0-9.]+)\s+MB compared in\s+(\d+)\s+ranges of windows outside buffers.*?"
    r"([0-9.]+)\s+MB in\s+(\d+)\s+ranges of title buffers' copies"
)

CARD_ACCT_RE = re.compile(
    r"card accounting:\s+room\s+(true|false),\s+the card\s+(\d+)\s+MB in use of a budget of\s+(\d+)\s+MB"
)

VRAM_INIT_RE = re.compile(
    r"the card's own:\s+budget\s+(\d+)\s+MB,\s+in use\s+(\d+)\s+MB.*?"
    r"the PC's shared with the card:\s+budget\s+(\d+)\s+MB,\s+in use\s+(\d+)\s+MB"
)

PIPELINE_TEL_RE = re.compile(
    r"pipeline telemetry \(window / total\):\s+lib_loaded\s+\d+/(\d+),\s+"
    r"driver_compiled\s+\d+/(\d+)\s+\([0-9.]+/([0-9.]+)\s+ms,\s+inval\s+(\d+),\s+err\s+(\d+)\),\s+early\s+(\d+);\s+"
    r"shader_cache hit\s+\d+/(\d+),\s+translated\s+\d+/(\d+)\s+\([0-9.]+/([0-9.]+)\s+ms,\s+fail\s+(\d+)\);\s+"
    r"wait_n\s+\d+/(\d+),\s+wait_ms\s+[0-9.]+/([0-9.]+),\s+timeouts\s+\d+/(\d+),\s+pending_drops\s+\d+/(\d+),\s+queued\s+(\d+)"
)

CARD_TEL_RE = re.compile(
    r"card telemetry\s+\([^)]+\):\s+rejected_for_card\(calls=(\d+),\s+total_mb=[0-9.]+,\s+budget=(\d+),\s+cap=(\d+),\s+thresh=(\d+),\s+create=(\d+)\)"
)

INDIRECT_TEL_RE = re.compile(
    r"indirect execution per frame:\s+ops\s+([0-9.]+),\s+clamp passes\s+([0-9.]+),\s+gathered\s+([0-9.]+),\s+host ExecuteIndirect\s+([0-9.]+)"
)

P2_DIAG_RE = re.compile(
    r"p2 diag:\s+unimpl_pso\(0x1c8=(\d+),\s+0x218=(\d+),\s+other=(\d+)\),\s+"
    r"oob_desc\(total=(\d+),\s+draw=(\d+),\s+min=(\d+),\s+max=(\d+)\),\s+"
    r"drop_cause\(null=(\d+),\s+pending=(\d+),\s+failed=(\d+)\)"
)


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = (len(s) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


@dataclass
class WindowSample:
    line: int
    ts: datetime
    fps: float
    ms_frame: float
    prep_ms: float
    copy_ms: float
    copy_mb: float
    tex_ms: float
    tex_mb: float
    gpu_ms: float
    back_ms: float
    back_mb: float
    draws: float
    dropped: float
    making: int
    indirect: float
    heap_passes: float
    heap_ms: float
    tex_waiting: int
    tex_away: int
    idle_total_ms: Optional[float] = None
    idle_behind_us_ms: Optional[float] = None
    cpu_compare_mb: Optional[float] = None


@dataclass
class LogReport:
    path: str
    build_line: str = ""
    adapter_line: str = ""
    render_scale_lines: int = 0
    telemetry_kind: str = "aggregated periodic windows"
    duration_s: float = 0.0
    total_windows: int = 0
    in_world_windows: int = 0
    windows: List[WindowSample] = field(default_factory=list)
    spikes_1s: List[Tuple[int, float, float, str]] = field(default_factory=list)
    cpu_compares: List[float] = field(default_factory=list)
    room_checks: int = 0
    room_false: int = 0
    peak_local_vram_mb: int = 0
    peak_budget_vram_mb: int = 0
    peak_nonlocal_vram_mb: int = 0
    translate_failures: int = 0
    pso_refusals: int = 0
    unimpl_0x1c8: int = 0
    unimpl_0x218: int = 0
    unimpl_warns: int = 0
    oob_total: int = 0
    oob_draw: int = 0
    drop_null: int = 0
    drop_pending: int = 0
    drop_failed: int = 0
    pipeline_totals: Dict[str, float] = field(default_factory=dict)
    card_rejects: Dict[str, int] = field(default_factory=dict)
    indirect_samples: List[Tuple[float, float, float, float]] = field(default_factory=list)
    exit_kind: str = "unknown/incomplete"


def classify_spike(w: WindowSample) -> str:
    # Overlapping stage & title/PSO waits are not additive; name primary bottleneck.
    if w.idle_total_ms is not None and w.idle_behind_us_ms is not None:
        external_ms = max(0.0, w.idle_total_ms - w.idle_behind_us_ms)
        if external_ms >= 0.5 * w.ms_frame:
            return f"external/title/PSO-wait ({external_ms:.0f} ms card idle outside submit)"
    parts = [
        ("copy-back", w.back_ms),
        ("textures", w.tex_ms),
        ("copies", w.copy_ms),
        ("preparing", w.prep_ms),
        ("GPU", w.gpu_ms),
    ]
    top_name, top_val = max(parts, key=lambda kv: kv[1])
    return f"{top_name} ({top_val:.0f} ms)"


def analyze_file(path: str, min_draws: float = 500.0) -> LogReport:
    report = LogReport(path=path)
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        raw_lines = fh.read().splitlines()

    first_ts: Optional[datetime] = None
    last_ts: Optional[datetime] = None
    last_idle: Optional[Tuple[float, float]] = None
    last_cpu_cmp: Optional[float] = None
    shutdown_requested = False
    shutdown_timeout = False
    watchdog_hit = False
    device_removed = False
    dred_seen = False

    for idx, raw in enumerate(raw_lines, 1):
        clean = ANSI_RE.sub("", raw).strip()
        norm = re.sub(r"\s+", " ", clean)
        if not norm:
            continue
        if norm.startswith("build ") and not report.build_line:
            report.build_line = norm
        if "gpu_upload_heaps=" in norm:
            report.adapter_line = norm
        if "VICTORY_RENDER_SCALE" in norm:
            report.render_scale_lines += 1

        ts_m = TS_RE.match(norm)
        cur_ts: Optional[datetime] = None
        if ts_m:
            cur_ts = datetime.fromisoformat(ts_m.group(1).replace("Z", "+00:00"))
            if first_ts is None:
                first_ts = cur_ts
            last_ts = cur_ts

        if "does not translate" in norm:
            report.translate_failures += 1
        if "host pipeline refused" in norm or "host mesh pipeline refused" in norm:
            report.pso_refusals += 1
        if "pipeline creation method that is not implemented" in norm:
            report.unimpl_warns += 1
            if "slot=0x1c8" in norm:
                report.unimpl_0x1c8 += 1
            elif "slot=0x218" in norm:
                report.unimpl_0x218 += 1

        if "window closed:" in norm or "WM_CLOSE" in norm:
            shutdown_requested = True
        if "did not finish suspending in time" in norm:
            shutdown_timeout = True
        if "watchdog" in norm and "hook applied" not in norm and "VICTORY_NO_WATCHDOG" not in norm:
            watchdog_hit = True
        if "device removed" in norm.lower() or "getdeviceremovedreason" in norm.lower() or "dxgi_error_device" in norm.lower():
            device_removed = True
        if "dred" in norm.lower() and ("fault" in norm.lower() or "breadcrumbs" in norm.lower()):
            dred_seen = True

        m_clk = CLOCK_RE.search(norm)
        if m_clk:
            last_idle = (float(m_clk.group(1)), float(m_clk.group(2)))

        m_cmp = CPU_CMP_RE.search(norm)
        if m_cmp:
            total_cmp = float(m_cmp.group(1)) + float(m_cmp.group(3))
            last_cpu_cmp = total_cmp
            report.cpu_compares.append(total_cmp)

        m_vram = VRAM_INIT_RE.search(norm)
        if m_vram:
            report.peak_budget_vram_mb = max(report.peak_budget_vram_mb, int(m_vram.group(1)))
            report.peak_local_vram_mb = max(report.peak_local_vram_mb, int(m_vram.group(2)))
            report.peak_nonlocal_vram_mb = max(report.peak_nonlocal_vram_mb, int(m_vram.group(4)))

        m_acct = CARD_ACCT_RE.search(norm)
        if m_acct:
            report.room_checks += 1
            if m_acct.group(1) == "false":
                report.room_false += 1
            report.peak_local_vram_mb = max(report.peak_local_vram_mb, int(m_acct.group(2)))
            report.peak_budget_vram_mb = max(report.peak_budget_vram_mb, int(m_acct.group(3)))

        m_ptel = PIPELINE_TEL_RE.search(norm)
        if m_ptel:
            report.pipeline_totals = {
                "lib_loaded": int(m_ptel.group(1)),
                "driver_compiled": int(m_ptel.group(2)),
                "compile_ms": float(m_ptel.group(3)),
                "lib_inval": int(m_ptel.group(4)),
                "lib_err": int(m_ptel.group(5)),
                "early": int(m_ptel.group(6)),
                "sc_hit": int(m_ptel.group(7)),
                "sc_translated": int(m_ptel.group(8)),
                "sc_ms": float(m_ptel.group(9)),
                "sc_fail": int(m_ptel.group(10)),
                "wait_n": int(m_ptel.group(11)),
                "wait_ms": float(m_ptel.group(12)),
                "timeouts": int(m_ptel.group(13)),
                "pending_drops": int(m_ptel.group(14)),
                "queued": int(m_ptel.group(15)),
            }

        m_ctel = CARD_TEL_RE.search(norm)
        if m_ctel:
            report.card_rejects = {
                "calls": int(m_ctel.group(1)),
                "budget": int(m_ctel.group(2)),
                "cap": int(m_ctel.group(3)),
                "thresh": int(m_ctel.group(4)),
                "create": int(m_ctel.group(5)),
            }

        m_ind = INDIRECT_TEL_RE.search(norm)
        if m_ind:
            report.indirect_samples.append(
                (float(m_ind.group(1)), float(m_ind.group(2)), float(m_ind.group(3)), float(m_ind.group(4)))
            )

        m_p2 = P2_DIAG_RE.search(norm)
        if m_p2:
            report.unimpl_0x1c8 += int(m_p2.group(1))
            report.unimpl_0x218 += int(m_p2.group(2))
            report.oob_total += int(m_p2.group(4))
            report.oob_draw += int(m_p2.group(5))
            report.drop_null += int(m_p2.group(8))
            report.drop_pending += int(m_p2.group(9))
            report.drop_failed += int(m_p2.group(10))

        m_fr = FRAME_RE.search(norm)
        if m_fr and cur_ts is not None:
            w = WindowSample(
                line=idx,
                ts=cur_ts,
                fps=float(m_fr.group(2)),
                ms_frame=float(m_fr.group(3)),
                prep_ms=float(m_fr.group(4)),
                copy_ms=float(m_fr.group(5)),
                copy_mb=float(m_fr.group(6)),
                tex_ms=float(m_fr.group(7)),
                tex_mb=float(m_fr.group(8)),
                gpu_ms=float(m_fr.group(9)),
                back_ms=float(m_fr.group(10)),
                back_mb=float(m_fr.group(11)),
                draws=float(m_fr.group(12)),
                dropped=float(m_fr.group(13)),
                making=int(m_fr.group(14)),
                indirect=float(m_fr.group(15)),
                heap_passes=float(m_fr.group(16)),
                heap_ms=float(m_fr.group(17)),
                tex_waiting=int(m_fr.group(20)),
                tex_away=int(m_fr.group(21)),
                idle_total_ms=last_idle[0] if last_idle else None,
                idle_behind_us_ms=last_idle[1] if last_idle else None,
                cpu_compare_mb=last_cpu_cmp,
            )
            report.windows.append(w)
            report.total_windows += 1
            if w.ms_frame >= 1000.0:
                report.spikes_1s.append((w.line, w.ms_frame, w.draws, classify_spike(w)))

    in_world = [w for w in report.windows if w.draws >= min_draws]
    report.in_world_windows = len(in_world)
    if first_ts is not None and last_ts is not None:
        report.duration_s = (last_ts - first_ts).total_seconds()

    exits = []
    if device_removed or dred_seen:
        exits.append("device_removal/DRED" if dred_seen else "device_removal")
    if watchdog_hit:
        exits.append("watchdog")
    if shutdown_requested:
        exits.append("requested_shutdown")
    if shutdown_timeout:
        exits.append("shutdown_timeout")
    report.exit_kind = "+".join(exits) if exits else "normal_or_unlogged"
    return report


def summarize(r: LogReport, min_draws: float = 500.0) -> Dict[str, object]:
    in_world = [w for w in r.windows if w.draws >= min_draws]
    ms = [w.ms_frame for w in in_world]
    med_ms = percentile(ms, 0.5)
    p95_ms = percentile(ms, 0.95)
    p99_ms = percentile(ms, 0.99)
    med_fps = (1000.0 / med_ms) if med_ms > 0 else 0.0
    over_250 = [w for w in in_world if w.ms_frame > 250.0]
    time_gt_250_s = 0.0
    for idx, w in enumerate(r.windows):
        if w.draws >= min_draws and w.ms_frame > 250.0 and idx > 0:
            time_gt_250_s += max(0.0, (w.ts - r.windows[idx - 1].ts).total_seconds())
    in_world_1s = [w for w in in_world if w.ms_frame >= 1000.0]
    cmp_in_world = [w.cpu_compare_mb for w in in_world if w.cpu_compare_mb is not None]
    wait_max = max((w.tex_waiting for w in r.windows), default=0)

    return {
        "file": os.path.basename(r.path),
        "duration_s": round(r.duration_s, 1),
        "exit": r.exit_kind,
        "telemetry": f"{r.telemetry_kind} ({r.in_world_windows}/{r.total_windows} in-world >= {int(min_draws)} draws)",
        "fps_median": round(med_fps, 2),
        "ms_median": round(med_ms, 2),
        "ms_p95": round(p95_ms, 2),
        "ms_p99": round(p99_ms, 2),
        "ms_max": round(max(ms), 2) if ms else 0.0,
        "win_gt_250ms": len(over_250),
        "time_gt_250ms_s": round(time_gt_250_s, 1),
        "win_gt_1s_inworld": len(in_world_1s),
        "win_gt_1s_all": len(r.spikes_1s),
        "prep_ms_med": round(percentile([w.prep_ms for w in in_world], 0.5), 2),
        "copy_ms_med": round(percentile([w.copy_ms for w in in_world], 0.5), 2),
        "tex_ms_med": round(percentile([w.tex_ms for w in in_world], 0.5), 2),
        "gpu_ms_med": round(percentile([w.gpu_ms for w in in_world], 0.5), 2),
        "back_ms_med": round(percentile([w.back_ms for w in in_world], 0.5), 2),
        "heap_ms_med": round(percentile([w.heap_ms for w in in_world], 0.5), 2),
        "cpu_cmp_mb_med": round(percentile(cmp_in_world, 0.5), 2),
        "draws_med": round(percentile([w.draws for w in in_world], 0.5), 0),
        "dropped_med": round(percentile([w.dropped for w in in_world], 0.5), 0),
        "tex_wait_max": wait_max,
        "room_false": f"{r.room_false}/{r.room_checks}",
        "peak_local_vram_mb": r.peak_local_vram_mb,
        "peak_nonlocal_vram_mb": r.peak_nonlocal_vram_mb,
        "translate_fail": r.translate_failures,
        "pso_refused": r.pso_refusals,
        "unimpl_pso_0x1c8_0x218": f"{r.unimpl_0x1c8}/{r.unimpl_0x218} (warns={r.unimpl_warns})",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Victory runtime logs")
    parser.add_argument("logs", nargs="+", help="Path(s) to victory log files")
    parser.add_argument("--min-draws", type=float, default=500.0, help="In-world draw threshold (default: 500)")
    parser.add_argument("--json", action="store_true", help="Print JSON summary")
    args = parser.parse_args()

    reports = [analyze_file(p, args.min_draws) for p in args.logs]
    summaries = [summarize(r, args.min_draws) for r in reports]
    if args.json:
        print(json.dumps(summaries, indent=2))
        return 0

    keys = list(summaries[0].keys())
    print("| Metric | " + " | ".join(str(s["file"]) for s in summaries) + " |")
    print("|---|" + "|".join(["---"] * len(summaries)) + "|")
    for k in keys[1:]:
        print(f"| {k} | " + " | ".join(str(s[k]) for s in summaries) + " |")

    for r in reports:
        print(f"\n### Details: {os.path.basename(r.path)}")
        print(f"- Build: `{r.build_line}`")
        if r.adapter_line:
            print(f"- Adapter: `{r.adapter_line}`")
        print("- Note: frame summaries in current runtime are periodic aggregated ~5s windows; multi-thread and overlapped stages are not additive.")
        if r.pipeline_totals:
            print(f"- Pipeline telemetry totals: {r.pipeline_totals}")
        if r.card_rejects:
            print(f"- Card rejects totals: {r.card_rejects}")
        if r.indirect_samples:
            ops = percentile([x[0] for x in r.indirect_samples], 0.5)
            clamps = percentile([x[1] for x in r.indirect_samples], 0.5)
            gathered = percentile([x[2] for x in r.indirect_samples], 0.5)
            calls = percentile([x[3] for x in r.indirect_samples], 0.5)
            print(f"- Indirect median/frame: ops={ops:.0f}, clamps={clamps:.0f}, gathered={gathered:.0f}, host_calls={calls:.0f}")
        if r.spikes_1s:
            print(f"- Windows >= 1000 ms/frame ({len(r.spikes_1s)} total):")
            for ln, ms_val, draws, why in r.spikes_1s:
                print(f"  - L{ln}: {ms_val:.1f} ms/frame (draws={draws:.0f}) -> {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
