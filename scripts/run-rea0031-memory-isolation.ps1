<#
.SYNOPSIS
    Runs isolated single-variable memory and CPU-prep experiments R02..R06 and R13 (authority, sparse, MWAITX backoff, GPU detile, accounting audit).
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateSet('baseline', 'authority', 'readback-validate', 'sparse', 'authority-sparse', 'mwaitx-spin', 'mwaitx-yield', 'mwaitx-progressive', 'gpu-detile', 'audit')]
    [string]$Mode,
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
$EnvMap['VICTORY_REA_INCREMENTAL_ACCOUNTING'] = '1'
$EnvMap['VICTORY_REA_HOST_SCHEDULER'] = '1'
$EnvMap['VICTORY_REA_NATIVE_CPU_FASTPATH'] = '1'
$EnvMap['VICTORY_REA_DIRTY_RANGES'] = '1'
$EnvMap['VICTORY_REA_ASYNC_STREAMING'] = '1'
$EnvMap['VICTORY_REA_GPU_FASTPATH'] = '1'

switch ($Mode) {
    'baseline' {
        $EnvMap['VICTORY_REA_GPU_AUTHORITY'] = '0'
        $EnvMap['VICTORY_REA_READBACK_VALIDATE'] = '0'
        $EnvMap['VICTORY_REA_SPARSE_TEXTURES'] = '0'
    }
    'authority' {
        $EnvMap['VICTORY_REA_GPU_AUTHORITY'] = '1'
        $EnvMap['VICTORY_REA_READBACK_VALIDATE'] = '0'
        $EnvMap['VICTORY_REA_SPARSE_TEXTURES'] = '0'
    }
    'readback-validate' {
        $EnvMap['VICTORY_REA_GPU_AUTHORITY'] = '1'
        $EnvMap['VICTORY_REA_READBACK_VALIDATE'] = '1'
        $EnvMap['VICTORY_REA_SPARSE_TEXTURES'] = '0'
    }
    'sparse' {
        $EnvMap['VICTORY_REA_GPU_AUTHORITY'] = '0'
        $EnvMap['VICTORY_REA_READBACK_VALIDATE'] = '0'
        $EnvMap['VICTORY_REA_SPARSE_TEXTURES'] = '1'
        $EnvMap['VICTORY_REA_SPARSE_RECLAIM'] = '0'
    }
    'authority-sparse' {
        $EnvMap['VICTORY_REA_GPU_AUTHORITY'] = '1'
        $EnvMap['VICTORY_REA_READBACK_VALIDATE'] = '1'
        $EnvMap['VICTORY_REA_SPARSE_TEXTURES'] = '1'
        $EnvMap['VICTORY_REA_SPARSE_RECLAIM'] = '0'
    }
    'mwaitx-spin' {
        $EnvMap['VICTORY_REA_GPU_AUTHORITY'] = '1'
        $EnvMap['VICTORY_REA_READBACK_VALIDATE'] = '1'
        $EnvMap['VICTORY_REA_SPARSE_TEXTURES'] = '1'
        $EnvMap['VICTORY_REA_MWAITX_MODE'] = 'spin'
    }
    'mwaitx-yield' {
        $EnvMap['VICTORY_REA_GPU_AUTHORITY'] = '1'
        $EnvMap['VICTORY_REA_READBACK_VALIDATE'] = '1'
        $EnvMap['VICTORY_REA_SPARSE_TEXTURES'] = '1'
        $EnvMap['VICTORY_REA_MWAITX_MODE'] = 'yield'
    }
    'mwaitx-progressive' {
        $EnvMap['VICTORY_REA_GPU_AUTHORITY'] = '1'
        $EnvMap['VICTORY_REA_READBACK_VALIDATE'] = '1'
        $EnvMap['VICTORY_REA_SPARSE_TEXTURES'] = '1'
        $EnvMap['VICTORY_REA_MWAITX_MODE'] = 'progressive'
    }
    'gpu-detile' {
        $EnvMap['VICTORY_REA_GPU_AUTHORITY'] = '1'
        $EnvMap['VICTORY_REA_READBACK_VALIDATE'] = '1'
        $EnvMap['VICTORY_REA_SPARSE_TEXTURES'] = '1'
        $EnvMap['VICTORY_REA003_GPU_DETILE'] = '1'
    }
    'audit' {
        $EnvMap['VICTORY_REA_GPU_AUTHORITY'] = '1'
        $EnvMap['VICTORY_REA_READBACK_VALIDATE'] = 'full'
        $EnvMap['VICTORY_REA_SPARSE_TEXTURES'] = '1'
        $EnvMap['VICTORY_REA_ACCOUNTING_AUDIT'] = '1'
    }
}

Invoke-VictoryIsolated `
    -PresetName "rea0031-mem-$Mode" `
    -ExePath $ExePath `
    -CacheBucket "rea0031-mem-$Mode" `
    -ManifestRole 'rea0031_release' `
    -EnvMap $EnvMap `
    -LogDir $LogDir `
    -GameDir $GameDir `
    -Console $Console `
    -NoBaseEnv:$NoBaseEnv `
    -Disable $Disable `
    -DryRun:$DryRun `
    -ExtraArgs $ExtraArgs
