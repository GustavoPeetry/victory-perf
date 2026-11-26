<#
.SYNOPSIS
    Runs the genuine frozen REA-002 baseline binary (commit f11885b) with isolated env and cache (Experiment R00).
#>
[CmdletBinding()]
param(
    [string]$ExePath,
    [string]$GameDir = 'C:\Users\gabeg\Downloads\GTA6.Xbox.v1.0.2.1',
    [string]$LogDir = (Join-Path $env:LOCALAPPDATA 'victory\logs'),
    [ValidateSet('series-s', 'series-x')]
    [string]$Console = 'series-s',
    [string]$RenderScale = '0.4',
    [string[]]$Disable = @(),
    [switch]$NoBaseEnv,
    [switch]$DryRun,
    [string[]]$ExtraArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot 'rea0031-common.ps1')

$RepoRoot = Get-Rea0031RepoRoot
if (-not $ExePath) {
    $ExePath = Join-Path $RepoRoot 'target\baseline-f11885b\victory.exe'
}

$EnvMap = Get-Rea0031BaseEnv -RenderScale $RenderScale -OwnUpscaler -WithStubs
$PresetVars = [ordered]@{
    VICTORY_REA_INCREMENTAL_ACCOUNTING = '1'
    VICTORY_REA_HOST_SCHEDULER         = '1'
    VICTORY_REA_NATIVE_CPU_FASTPATH    = '1'
    VICTORY_REA_DIRTY_RANGES           = '1'
    VICTORY_REA_ASYNC_STREAMING        = '1'
    VICTORY_REA_GPU_FASTPATH           = '1'
}
foreach ($kv in $PresetVars.GetEnumerator()) { $EnvMap[$kv.Key] = $kv.Value }

Invoke-VictoryIsolated `
    -PresetName 'rea0031-baseline-original' `
    -ExePath $ExePath `
    -CacheBucket 'baseline-f11885b' `
    -ManifestRole 'baseline_f11885b' `
    -EnvMap $EnvMap `
    -LogDir $LogDir `
    -GameDir $GameDir `
    -Console $Console `
    -NoBaseEnv:$NoBaseEnv `
    -Disable $Disable `
    -DryRun:$DryRun `
    -ExtraArgs $ExtraArgs
