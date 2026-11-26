<#
.SYNOPSIS
    Runs isolated single-variable shader experiments R07..R11 (wave32, recovery allowlists, shader-opt, native passes).
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateSet('baseline', 'wave32', 'recovery-safe', 'recovery-all', 'shader-opt', 'ambient-native', 'shadowmask-native', 'native-passes', 'allowlist')]
    [string]$Mode,
    [string]$ExePath,
    [string]$GameDir = 'C:\Users\gabeg\Downloads\GTA6.Xbox.v1.0.2.1',
    [string]$TargetsPath,
    [string]$LogDir = (Join-Path $env:LOCALAPPDATA 'victory\logs'),
    [ValidateSet('series-s', 'series-x')]
    [string]$Console = 'series-s',
    [string]$RenderScale = '0.4',
    [string]$AllowShaders = '',
    [string]$DenyShaders = '',
    [int]$Jobs = 4,
    [switch]$TranslateOnly,
    [switch]$UseDiag,
    [bool]$Dred = $true,
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
    $ExePath = if ($UseDiag) {
        Join-Path $RepoRoot 'target\diag\victory.exe'
    } else {
        Join-Path $RepoRoot 'target\release\victory.exe'
    }
}

$EnvMap = Get-Rea0031BaseEnv -RenderScale $RenderScale -OwnUpscaler -WithStubs
$EnvMap['VICTORY_REA_GPU_AUTHORITY'] = '1'
$EnvMap['VICTORY_REA_READBACK_VALIDATE'] = '1'
$EnvMap['VICTORY_REA_SPARSE_TEXTURES'] = '1'
$EnvMap['VICTORY_REA_INCREMENTAL_ACCOUNTING'] = '1'
$EnvMap['VICTORY_REA_HOST_SCHEDULER'] = '1'
$EnvMap['VICTORY_REA_NATIVE_CPU_FASTPATH'] = '1'
$EnvMap['VICTORY_REA_DIRTY_RANGES'] = '1'
$EnvMap['VICTORY_REA_ASYNC_STREAMING'] = '1'
$EnvMap['VICTORY_REA_GPU_FASTPATH'] = '1'
$EnvMap['VICTORY_GPU_BREADCRUMBS'] = '1'
if ($Dred) { $EnvMap['VICTORY_D3D12_DRED'] = '1' }
if ($Gbv) { $EnvMap['VICTORY_D3D12_GBV'] = '1' }

switch ($Mode) {
    'baseline' {
        $EnvMap['VICTORY_REA003_SHADER_RECOVERY'] = '0'
        $EnvMap['VICTORY_REA003_SHADER_OPT'] = '0'
    }
    'wave32' {
        $EnvMap['VICTORY_HOST_MAX_WAVE'] = '32'
        $EnvMap['VICTORY_REA003_SHADER_RECOVERY'] = '0'
    }
    'recovery-safe' {
        $EnvMap['VICTORY_HOST_MAX_WAVE'] = '32'
        $EnvMap['VICTORY_REA003_SHADER_RECOVERY'] = '1'
        $EnvMap['VICTORY_REA003_SHADER_RECOVERY_ALLOW'] = 'buf_meta,readfirstlane,const_mask'
    }
    'recovery-all' {
        $EnvMap['VICTORY_HOST_MAX_WAVE'] = '32'
        $EnvMap['VICTORY_REA003_SHADER_RECOVERY'] = '1'
        $EnvMap['VICTORY_REA003_SHADER_RECOVERY_ALLOW'] = 'all'
    }
    'shader-opt' {
        $EnvMap['VICTORY_HOST_MAX_WAVE'] = '32'
        $EnvMap['VICTORY_REA003_SHADER_OPT'] = '1'
    }
    'ambient-native' {
        $EnvMap['VICTORY_HOST_MAX_WAVE'] = '32'
        $EnvMap['VICTORY_REA003_AMBIENT_NATIVE'] = '1'
    }
    'shadowmask-native' {
        $EnvMap['VICTORY_HOST_MAX_WAVE'] = '32'
        $EnvMap['VICTORY_REA003_SHADOWMASK_NATIVE'] = '1'
    }
    'native-passes' {
        $EnvMap['VICTORY_HOST_MAX_WAVE'] = '32'
        $EnvMap['VICTORY_REA003_AMBIENT_NATIVE'] = '1'
        $EnvMap['VICTORY_REA003_SHADOWMASK_NATIVE'] = '1'
    }
    'allowlist' {
        $EnvMap['VICTORY_HOST_MAX_WAVE'] = '32'
        $EnvMap['VICTORY_REA003_SHADER_RECOVERY'] = '1'
    }
}

if ($AllowShaders) {
    $EnvMap['VICTORY_REA003_SHADER_RECOVERY'] = '1'
    $EnvMap['VICTORY_REA003_SHADER_RECOVERY_ALLOW'] = $AllowShaders
}
if ($DenyShaders) {
    $EnvMap['VICTORY_REA003_SHADER_RECOVERY_DENY'] = $DenyShaders
}

$CliOverride = if ($TranslateOnly) {
    $Target = if ($TargetsPath) { $TargetsPath } else { $GameDir }
    @('translate', '--validate', '--pipelines', '--list', '--jobs', "$Jobs", $Target)
} else {
    $null
}

$Bucket = "rea0031-shader-$Mode"
if ($AllowShaders -or $DenyShaders) {
    $raw = "$AllowShaders|$DenyShaders"
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($raw)
    $sha = [System.BitConverter]::ToString([System.Security.Cryptography.SHA256]::Create().ComputeHash($bytes)).Replace('-', '').Substring(0, 8).ToLowerInvariant()
    $Bucket = "$Bucket-$sha"
}

Invoke-VictoryIsolated `
    -PresetName "rea0031-shader-$Mode" `
    -ExePath $ExePath `
    -CacheBucket $Bucket `
    -ManifestRole $(if ($UseDiag) { 'rea0031_diag' } else { 'rea0031_release' }) `
    -EnvMap $EnvMap `
    -CliArgsOverride $CliOverride `
    -LogDir $LogDir `
    -GameDir $GameDir `
    -Console $Console `
    -NoBaseEnv:$NoBaseEnv `
    -Disable $Disable `
    -DryRun:$DryRun `
    -ExtraArgs $ExtraArgs
