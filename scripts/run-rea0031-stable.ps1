<#
.SYNOPSIS
    Primary recommended REA-003.1 gameplay profile: validated CPU/VRAM/sparse/rendergraph/shader-opt stack with unsafe experiments off (Experiment R18).
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
    $ExePath = Join-Path $RepoRoot 'target\release\victory.exe'
}

$EnvMap = Get-Rea0031BaseEnv -RenderScale $RenderScale -OwnUpscaler -WithStubs
$StableVars = [ordered]@{
    VICTORY_REA_GPU_AUTHORITY           = '1'
    VICTORY_REA_READBACK_VALIDATE       = '1'
    VICTORY_REA_SPARSE_TEXTURES         = '1'
    VICTORY_REA_SPARSE_RECLAIM          = '0'
    VICTORY_REA_INCREMENTAL_ACCOUNTING  = '1'
    VICTORY_REA_HOST_SCHEDULER          = '1'
    VICTORY_REA_NATIVE_CPU_FASTPATH     = '1'
    VICTORY_REA_DIRTY_RANGES            = '1'
    VICTORY_REA_ASYNC_STREAMING         = '1'
    VICTORY_REA_GPU_FASTPATH            = '1'
    VICTORY_REA003_RENDERGRAPH_FASTPATH = '1'
    VICTORY_REA003_GPU_DRIVEN           = '1'
    VICTORY_REA003_SHADER_OPT           = '1'
    VICTORY_REA003_SHADER_RECOVERY      = '0'
    VICTORY_REA003_AMBIENT_NATIVE       = '0'
    VICTORY_REA003_SHADOWMASK_NATIVE    = '0'
    VICTORY_REA003_GPU_DETILE           = '0'
    VICTORY_REA003_DXR                  = '0'
    VICTORY_REA003_UPSCALER             = 'original'
}
foreach ($kv in $StableVars.GetEnumerator()) { $EnvMap[$kv.Key] = $kv.Value }

Invoke-VictoryIsolated `
    -PresetName 'rea0031-stable' `
    -ExePath $ExePath `
    -CacheBucket 'rea0031-stable' `
    -ManifestRole 'rea0031_release' `
    -EnvMap $EnvMap `
    -LogDir $LogDir `
    -GameDir $GameDir `
    -Console $Console `
    -NoBaseEnv:$NoBaseEnv `
    -Disable $Disable `
    -DryRun:$DryRun `
    -ExtraArgs $ExtraArgs
