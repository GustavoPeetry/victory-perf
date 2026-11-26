<#
.SYNOPSIS
    Compares two VIctory runtime logs (.log / .jsonl / translate outputs) for REA-003 verification.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$Baseline,
    [Parameter(Mandatory)]
    [string]$Candidate,
    [string]$OutJson,
    [int]$SkipWarmupSamples = 1
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

function Add-MetricSample {
    param($Map, [string]$Key, [double]$Value)
    if (-not $Map.Contains($Key)) {
        $Map[$Key] = New-Object System.Collections.Generic.List[double]
    }
    $Map[$Key].Add($Value)
}

function Parse-Rea003File {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        throw "Log file not found: $Path"
    }
    $Series = [ordered]@{}
    $DxilValidated = 0
    $PsoCreated = 0
    $TranslateFailures = 0

    foreach ($line in [System.IO.File]::ReadLines($Path)) {
        $trimmed = $line.Trim()
        if ($trimmed.StartsWith('{') -and $trimmed.EndsWith('}')) {
            try {
                $obj = $trimmed | ConvertFrom-Json
                foreach ($prop in $obj.PSObject.Properties) {
                    if ($prop.Name -ne 'seq' -and $null -ne $prop.Value) {
                        [double]$val = 0
                        if ([double]::TryParse("$($prop.Value)", [System.Globalization.NumberStyles]::Float, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$val)) {
                            Add-MetricSample $Series $prop.Name $val
                        }
                    }
                }
                continue
            } catch {
            }
        }

        if ($line -match '(\d+(?:\.\d+)?) fps, (\d+(?:\.\d+)?) ms/frame \(([\d.]+) preparing: copies ([\d.]+) ms ([\d.]+) MB, textures ([\d.]+) ms ([\d.]+) MB; ([\d.]+) on the GPU; ([\d.]+) copying back ([\d.]+) MB\), (\d+(?:\.\d+)?) draws/frame') {
            Add-MetricSample $Series 'fps' ([double]$Matches[1])
            Add-MetricSample $Series 'avg_ms' ([double]$Matches[2])
            Add-MetricSample $Series 'cpu_prep_ms' ([double]$Matches[3])
            Add-MetricSample $Series 'copy_ms' ([double]$Matches[4])
            Add-MetricSample $Series 'copy_mb' ([double]$Matches[5])
            Add-MetricSample $Series 'tex_ms' ([double]$Matches[6])
            Add-MetricSample $Series 'tex_mb' ([double]$Matches[7])
            Add-MetricSample $Series 'gpu_ms' ([double]$Matches[8])
            Add-MetricSample $Series 'back_ms' ([double]$Matches[9])
            Add-MetricSample $Series 'back_mb' ([double]$Matches[10])
            Add-MetricSample $Series 'draws' ([double]$Matches[11])
        }

        if ($line -match 'vram local=(\d+)/(\d+) MB, non_local=(\d+)/(\d+) MB; tex churn: alive=(\d+) \((\d+) MB\)') {
            Add-MetricSample $Series 'vram_used_mb' ([double]$Matches[1])
            Add-MetricSample $Series 'vram_budget_mb' ([double]$Matches[2])
            Add-MetricSample $Series 'nonlocal_mb' ([double]$Matches[3])
            Add-MetricSample $Series 'tex_alive_mb' ([double]$Matches[6])
        }

        if ($line -match 'REA-00[23] telemetry:\s*(.+)$') {
            foreach ($token in ($Matches[1] -split '\s+')) {
                if ($token -match '^([a-zA-Z0-9_]+)=(-?\d+(?:\.\d+)?)$') {
                    $k = $Matches[1]
                    if ($k -ne 'seq') { Add-MetricSample $Series "rea_$k" ([double]$Matches[2]) }
                }
            }
        }

        if ($line -match '\svalidated\s*$') { $DxilValidated++ }
        elseif ($line -match '\spipeline created\s*$') { $PsoCreated++ }
        elseif ($line -notmatch '^\s' -and $line -match '\s(emit:|validator:|pipeline:|not translated|DXIL validation)') { $TranslateFailures++ }
    }

    if ($DxilValidated -gt 0 -or $PsoCreated -gt 0 -or $TranslateFailures -gt 0) {
        Add-MetricSample $Series 'translate_dxil_validated' $DxilValidated
        Add-MetricSample $Series 'translate_pso_created' $PsoCreated
        Add-MetricSample $Series 'translate_failures' $TranslateFailures
    }

    $Summary = [ordered]@{}
    foreach ($k in $Series.Keys) {
        $vals = $Series[$k]
        $startIdx = if ($vals.Count -gt ($SkipWarmupSamples + 1)) { $SkipWarmupSamples } else { 0 }
        $slice = @($vals)[$startIdx..($vals.Count - 1)]
        $measure = $slice | Measure-Object -Average -Minimum -Maximum
        $Summary[$k] = [ordered]@{
            count = $slice.Count
            mean  = [math]::Round($measure.Average, 3)
            min   = [math]::Round($measure.Minimum, 3)
            max   = [math]::Round($measure.Maximum, 3)
        }
    }
    return $Summary
}

$BaseStats = Parse-Rea003File -Path $Baseline
$CandStats = Parse-Rea003File -Path $Candidate

$AllKeys = New-Object System.Collections.Generic.List[string]
foreach ($k in $BaseStats.Keys) { $AllKeys.Add($k) }
foreach ($k in $CandStats.Keys) {
    if (-not $AllKeys.Contains($k)) { $AllKeys.Add($k) }
}

$Rows = @()
Write-Output "| Metric | Baseline (mean) | Candidate (mean) | Delta | Delta % | Samples (B/C) |"
Write-Output "| --- | ---: | ---: | ---: | ---: | :---: |"

foreach ($k in $AllKeys) {
    $b = if ($BaseStats.Contains($k)) { $BaseStats[$k] } else { $null }
    $c = if ($CandStats.Contains($k)) { $CandStats[$k] } else { $null }
    $bStr = if ($b) { "$($b.mean)" } else { 'N/A' }
    $cStr = if ($c) { "$($c.mean)" } else { 'N/A' }
    $deltaStr = 'N/A'
    $pctStr = 'N/A'
    if ($b -and $c) {
        $d = [math]::Round($c.mean - $b.mean, 3)
        $deltaStr = if ($d -ge 0) { "+$d" } else { "$d" }
        if ([math]::Abs($b.mean) -gt 1e-9) {
            $pct = [math]::Round(($d / $b.mean) * 100.0, 2)
            $pctStr = if ($pct -ge 0) { "+$pct%" } else { "$pct%" }
        } else {
            $pctStr = '0%'
        }
    }
    $nStr = "$(if ($b) { $b.count } else { 0 }) / $(if ($c) { $c.count } else { 0 })"
    Write-Output "| ``$k`` | $bStr | $cStr | $deltaStr | $pctStr | $nStr |"
    $Rows += [ordered]@{
        metric    = $k
        baseline  = $b
        candidate = $c
        delta     = $deltaStr
        delta_pct = $pctStr
    }
}

if ($OutJson) {
    $OutDir = Split-Path -Parent $OutJson
    if ($OutDir) { New-Item -ItemType Directory -Force -Path $OutDir | Out-Null }
    $Report = [ordered]@{
        timestamp  = (Get-Date).ToString('o')
        baseline   = (Resolve-Path $Baseline).Path
        candidate  = (Resolve-Path $Candidate).Path
        skip_warm  = $SkipWarmupSamples
        comparison = $Rows
    }
    $Report | ConvertTo-Json -Depth 6 | Set-Content -Path $OutJson -Encoding UTF8
    Write-Host "Wrote comparison JSON to $OutJson"
}
