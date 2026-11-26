<#
.SYNOPSIS
    Compares two or more VIctory telemetry logs (.jsonl / .log) for REA-003.1 verification,
    checking run_id/seq integrity, filtering 3D-world windows by -MinDraws, and summarizing
    medians, P95, P99, low1_fps, CPU prep phases, VRAM, sparse residency, and backend states.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$Baseline,
    [Parameter(Mandatory)]
    [string]$Candidate,
    [string[]]$Extra = @(),
    [double]$MinDraws = 1000.0,
    [int]$SkipWarmupSamples = 1,
    [string]$OutJson
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if ($OutJson) {
    $ResolvedOut = [System.IO.Path]::GetFullPath($OutJson)
    if ($ResolvedOut.StartsWith($RepoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to write OutJson inside repository: $ResolvedOut"
    }
}

function Get-Percentile {
    param([double[]]$Values, [double]$P)
    if (-not $Values -or $Values.Count -eq 0) { return $null }
    $sorted = @($Values | Sort-Object)
    $idx = [int][math]::Floor(($sorted.Count - 1) * $P)
    return $sorted[$idx]
}

function Parse-Rea0031Telemetry {
    param([string]$Path, [double]$MinDraws, [int]$SkipWarmupSamples)
    if (-not (Test-Path $Path -PathType Leaf)) {
        throw "Log file not found: $Path"
    }

    $jsonRows = New-Object System.Collections.Generic.List[object]
    $runIds = New-Object System.Collections.Generic.HashSet[string]
    $seqResets = 0
    $prevSeq = -1L
    $parseErrors = 0

    foreach ($line in [System.IO.File]::ReadLines($Path)) {
        $trimmed = $line.Trim()
        if ($trimmed.StartsWith('{') -and $trimmed.EndsWith('}')) {
            try {
                $obj = $trimmed | ConvertFrom-Json
                $jsonRows.Add($obj)
                if ($obj.PSObject.Properties['run_id'] -and $obj.run_id) {
                    [void]$runIds.Add("$($obj.run_id)")
                }
                if ($obj.PSObject.Properties['seq']) {
                    $s = [long]$obj.seq
                    if ($prevSeq -ge 0 -and $s -le $prevSeq) {
                        $seqResets++
                    }
                    $prevSeq = $s
                }
            } catch {
                $parseErrors++
            }
        }
    }

    if ($jsonRows.Count -eq 0) {
        return [ordered]@{
            path                = $Path
            total_windows       = 0
            qualified_windows   = 0
            used_fallback       = $false
            run_ids             = @()
            seq_resets          = $seqResets
            multi_run_corrupted = $false
            parse_errors        = $parseErrors
            metrics             = [ordered]@{}
            backends            = [ordered]@{}
            strings             = [ordered]@{}
        }
    }

    $startIdx = if ($jsonRows.Count -gt ($SkipWarmupSamples + 1)) { $SkipWarmupSamples } else { 0 }
    $afterWarmup = New-Object System.Collections.Generic.List[object]
    $qualified = New-Object System.Collections.Generic.List[object]
    for ($idx = $startIdx; $idx -lt $jsonRows.Count; $idx++) {
        $row = $jsonRows[$idx]
        $afterWarmup.Add($row)
        if ($row.PSObject.Properties['draws'] -and ([double]$row.draws) -ge $MinDraws) {
            $qualified.Add($row)
        }
    }
    $usedFallback = $false
    $windowSet = if ($qualified.Count -gt 0) {
        $qualified
    } else {
        $usedFallback = $true
        $afterWarmup
    }

    $series = [ordered]@{}
    $strings = [ordered]@{}
    $lastBackendMap = [ordered]@{}

    foreach ($row in $windowSet) {
        foreach ($prop in $row.PSObject.Properties) {
            if ($prop.Name -in @('seq', 'pid', 'monotonic_ms', 'utc_unix_ms')) { continue }
            if ($prop.Name -eq 'backends' -and $null -ne $prop.Value) {
                foreach ($bp in $prop.Value.PSObject.Properties) {
                    $st = $bp.Value
                    if ($null -ne $st -and $st.PSObject.Properties['state']) {
                        $lastBackendMap[$bp.Name] = "$($st.state) (calls=$($st.executed_calls), active=$($st.active))"
                    }
                }
                continue
            }
            if ($null -eq $prop.Value) { continue }
            [double]$num = 0
            if ($prop.Value -is [bool]) {
                continue
            } elseif ([double]::TryParse("$($prop.Value)", [System.Globalization.NumberStyles]::Float, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$num)) {
                if (-not $series.Contains($prop.Name)) {
                    $series[$prop.Name] = New-Object System.Collections.Generic.List[double]
                }
                $series[$prop.Name].Add($num)
            } else {
                $strings[$prop.Name] = "$($prop.Value)"
            }
        }
    }

    $metrics = [ordered]@{}
    foreach ($k in $series.Keys) {
        $vals = [double[]]$series[$k].ToArray()
        $m = $vals | Measure-Object -Average -Minimum -Maximum
        $metrics[$k] = [ordered]@{
            count  = $vals.Length
            median = [math]::Round((Get-Percentile -Values $vals -P 0.5), 3)
            p95    = [math]::Round((Get-Percentile -Values $vals -P 0.95), 3)
            p99    = [math]::Round((Get-Percentile -Values $vals -P 0.99), 3)
            mean   = [math]::Round($m.Average, 3)
            min    = [math]::Round($m.Minimum, 3)
            max    = [math]::Round($m.Maximum, 3)
        }
    }

    $corrupted = ($runIds.Count -gt 1) -or ($seqResets -gt 0)
    return [ordered]@{
        path                = $Path
        total_windows       = $jsonRows.Count
        qualified_windows   = $qualified.Count
        used_fallback       = $usedFallback
        run_ids             = @($runIds)
        seq_resets          = $seqResets
        multi_run_corrupted = $corrupted
        parse_errors        = $parseErrors
        metrics             = $metrics
        backends            = $lastBackendMap
        strings             = $strings
    }
}

$Inputs = @($Baseline, $Candidate) + $Extra
$Parsed = @()
foreach ($p in $Inputs) {
    $Parsed += Parse-Rea0031Telemetry -Path $p -MinDraws $MinDraws -SkipWarmupSamples $SkipWarmupSamples
}

Write-Output "=== REA-003.1 Log Comparison Summary (MinDraws=$MinDraws, SkipWarmup=$SkipWarmupSamples) ==="
foreach ($item in $Parsed) {
    $name = Split-Path -Leaf $item.path
    $flag = if ($item.multi_run_corrupted) { "[CORRUPT: seq_resets=$($item.seq_resets), run_ids=$($item.run_ids.Count)]" } else { "[OK: unique_run]" }
    $fpsCount = if ($item.metrics.Contains('fps')) { $item.metrics['fps'].count } else { $item.total_windows }
    $qualNote = if ($item.used_fallback) { "0 windows >= $MinDraws draws (showing all $fpsCount post-warmup windows)" } else { "$($item.qualified_windows)/$($item.total_windows) windows >= $MinDraws draws" }
    $gpuSt = if ($item.strings.Contains('gpu_time_status')) { $item.strings['gpu_time_status'] } else { 'unknown' }
    $upSt = if ($item.strings.Contains('upscaler_active_backend')) { $item.strings['upscaler_active_backend'] } elseif ($item.strings.Contains('ffx_hle_mode')) { $item.strings['ffx_hle_mode'] } else { 'unknown' }
    $bvhSt = if ($item.strings.Contains('bvh_provenance')) { $item.strings['bvh_provenance'] } else { 'unknown' }
    Write-Output ("- {0}: {1} | {2} | gpu_time_status={3} | upscaler={4} | bvh={5}" -f `
        $name, $flag, $qualNote, $gpuSt, $upSt, $bvhSt)
    if ($item.multi_run_corrupted) {
        Write-Warning "Detected multi-run append or seq reset in $($item.path): seq_resets=$($item.seq_resets), run_ids=$($item.run_ids -join ',')"
    }
}

$Base = $Parsed[0]
$Cand = $Parsed[1]
$KeyMetrics = @(
    'fps', 'avg_ms', 'p95_ms', 'p99_ms', 'low1_fps',
    'cpu_prep_ms', 'gpu_ms', 'gpu_hw_ms', 'back_ms', 'back_mb', 'draws',
    'vram_used_mb', 'vram_budget_mb', 'acc_comm_mb', 'acc_res_mb', 'sparse_mapped_mb', 'sparse_virt_mb', 'sparse_reclaimed_tiles',
    'audit_runs', 'audit_mismatches', 'rb_tagged', 'rb_miss', 'rb_validate_warnings',
    'auth_skipped_cmp', 'clzero_fast_calls', 'wait_yields', 'wait_sleeps',
    'prep_ask_once_ms', 'prep_copy_ms', 'prep_fill_ms', 'prep_materialize_ms', 'prep_authority_ms', 'prep_note_gpu_write_ms', 'prep_shadow_update_ms', 'prep_subpage_split_ms', 'prep_tile_lock_ms',
    'ffx_0x218_calls', 'slot_0x218_calls', 'ffx_hle_dispatches', 'ffx_hle_calls'
)

Write-Output ""
Write-Output "| Metric | Base Median (mean) | Cand Median (mean) | Median Delta % | Cand P95 / Max |"
Write-Output "| --- | ---: | ---: | ---: | ---: |"
foreach ($k in $KeyMetrics) {
    $b = if ($Base.metrics.Contains($k)) { $Base.metrics[$k] } else { $null }
    $c = if ($Cand.metrics.Contains($k)) { $Cand.metrics[$k] } else { $null }
    if (-not $b -and -not $c) { continue }
    $bStr = if ($b) { "$($b.median) ($($b.mean))" } else { 'N/A' }
    $cStr = if ($c) { "$($c.median) ($($c.mean))" } else { 'N/A' }
    $pct = 'N/A'
    if ($b -and $c -and [math]::Abs($b.median) -gt 1e-9) {
        $p = [math]::Round((($c.median - $b.median) / $b.median) * 100.0, 2)
        $pct = if ($p -ge 0) { "+$p%" } else { "$p%" }
    }
    $cTail = if ($c) { "$($c.p95) / $($c.max)" } else { 'N/A' }
    Write-Output "| ``$k`` | $bStr | $cStr | $pct | $cTail |"
}

if ($Cand.backends.Count -gt 0) {
    Write-Output ""
    Write-Output "Candidate last-window backend lifecycle states:"
    foreach ($kv in $Cand.backends.GetEnumerator()) {
        Write-Output "  $($kv.Key): $($kv.Value)"
    }
}

if ($OutJson) {
    $Dir = Split-Path -Parent $OutJson
    if ($Dir) { New-Item -ItemType Directory -Force -Path $Dir | Out-Null }
    $Report = [ordered]@{
        generated_at = (Get-Date).ToString('o')
        min_draws    = $MinDraws
        skip_warmup  = $SkipWarmupSamples
        sessions     = $Parsed
    }
    $Report | ConvertTo-Json -Depth 6 | Set-Content -Path $OutJson -Encoding UTF8
    Write-Output "Wrote comparison JSON: $OutJson"
}
