<#
.SYNOPSIS
    Runs the diagnostics build (target\diag\victory.exe) with D3D12 Debug Layer, DRED, GPU breadcrumbs, full readback validation, and accounting audit.
#>
[CmdletBinding()]
param(
    [string]$ExePath,
    [string]$GameDir = 'C:\Users\gabeg\Downloads\GTA6.Xbox.v1.0.2.1',
    [string]$LogDir = (Join-Path $env:LOCALAPPDATA 'victory\logs'),
    [ValidateSet('series-s', 'series-x')]
    [string]$Console = 'series-s',
    [ValidateSet('stable', 'experimental', 'shaders', 'memory')]
    [string]$Profile = 'stable',
    [string]$RenderScale = '0.4',
    [switch]$Gbv,
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
    $ExePath = Join-Path $RepoRoot 'target\diag\victory.exe'
}

$EnvMap = Get-Rea0031BaseEnv -RenderScale $RenderScale -OwnUpscaler -WithStubs

# Layer profile settings first
switch ($Profile) {
    'stable' {
        $EnvMap['VICTORY_REA_GPU_AUTHORITY'] = '1'
        $EnvMap['VICTORY_REA_SPARSE_TEXTURES'] = '1'
        $EnvMap['VICTORY_REA_SPARSE_RECLAIM'] = '0'
        $EnvMap['VICTORY_REA_INCREMENTAL_ACCOUNTING'] = '1'
        $EnvMap['VICTORY_REA_HOST_SCHEDULER'] = '1'
        $EnvMap['VICTORY_REA_NATIVE_CPU_FASTPATH'] = '1'
        $EnvMap['VICTORY_REA_DIRTY_RANGES'] = '1'
        $EnvMap['VICTORY_REA_ASYNC_STREAMING'] = '1'
        $EnvMap['VICTORY_REA_GPU_FASTPATH'] = '1'
        $EnvMap['VICTORY_REA003_RENDERGRAPH_FASTPATH'] = '1'
        $EnvMap['VICTORY_REA003_GPU_DRIVEN'] = '1'
        $EnvMap['VICTORY_REA003_SHADER_OPT'] = '1'
    }
    'experimental' {
        $EnvMap['VICTORY_HOST_MAX_WAVE'] = '32'
        $EnvMap['VICTORY_REA_GPU_AUTHORITY'] = '1'
        $EnvMap['VICTORY_REA_SPARSE_TEXTURES'] = '1'
        $EnvMap['VICTORY_REA_INCREMENTAL_ACCOUNTING'] = '1'
        $EnvMap['VICTORY_REA_HOST_SCHEDULER'] = '1'
        $EnvMap['VICTORY_REA_NATIVE_CPU_FASTPATH'] = '1'
        $EnvMap['VICTORY_REA_DIRTY_RANGES'] = '1'
        $EnvMap['VICTORY_REA_ASYNC_STREAMING'] = '1'
        $EnvMap['VICTORY_REA_GPU_FASTPATH'] = '1'
        $EnvMap['VICTORY_REA003_RENDERGRAPH_FASTPATH'] = '1'
        $EnvMap['VICTORY_REA003_GPU_DRIVEN'] = '1'
        $EnvMap['VICTORY_REA003_SHADER_OPT'] = '1'
        $EnvMap['VICTORY_REA003_SHADER_RECOVERY'] = '1'
        $EnvMap['VICTORY_REA003_SHADER_RECOVERY_ALLOW'] = 'buf_meta,readfirstlane,const_mask'
    }
    'shaders' {
        $EnvMap['VICTORY_HOST_MAX_WAVE'] = '32'
        $EnvMap['VICTORY_REA_GPU_AUTHORITY'] = '1'
        $EnvMap['VICTORY_REA_SPARSE_TEXTURES'] = '1'
        $EnvMap['VICTORY_REA003_SHADER_RECOVERY'] = '1'
        $EnvMap['VICTORY_REA003_SHADER_RECOVERY_ALLOW'] = 'buf_meta,readfirstlane,const_mask'
        $EnvMap['VICTORY_REA003_SHADER_OPT'] = '1'
    }
    'memory' {
        $EnvMap['VICTORY_REA_GPU_AUTHORITY'] = '1'
        $EnvMap['VICTORY_REA_SPARSE_TEXTURES'] = '1'
        $EnvMap['VICTORY_REA_INCREMENTAL_ACCOUNTING'] = '1'
        $EnvMap['VICTORY_REA_DIRTY_RANGES'] = '1'
    }
}

$DiagVars = [ordered]@{
    VICTORY_D3D12_DEBUG             = '1'
    VICTORY_D3D12_DRED              = '1'
    VICTORY_GPU_BREADCRUMBS         = '1'
    VICTORY_REA_ACCOUNTING_AUDIT    = '1'
    VICTORY_REA_READBACK_VALIDATE   = 'full'
    VICTORY_REA_FFX_TRACE           = '1'
}
foreach ($kv in $DiagVars.GetEnumerator()) { $EnvMap[$kv.Key] = $kv.Value }
if ($Gbv) {
    $EnvMap['VICTORY_D3D12_GBV'] = '1'
}

Invoke-VictoryIsolated `
    -PresetName "rea0031-diag-$Profile" `
    -ExePath $ExePath `
    -CacheBucket "rea0031-diag-$Profile" `
    -ManifestRole 'rea0031_diag' `
    -EnvMap $EnvMap `
    -LogDir $LogDir `
    -GameDir $GameDir `
    -Console $Console `
    -NoBaseEnv:$NoBaseEnv `
    -Disable $Disable `
    -DryRun:$DryRun `
    -ExtraArgs $ExtraArgs
