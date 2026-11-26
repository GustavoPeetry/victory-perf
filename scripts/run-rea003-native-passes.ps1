<#
.SYNOPSIS
    Launches VIctory with Frente C native ambient and shadowmask shader execution enabled (with automatic stand-in fallback).
#>
[CmdletBinding()]
param(
    [string]$ExePath,
    [string]$GameDir = 'C:\Users\gabeg\Downloads\GTA6.Xbox.v1.0.2.1',
    [string]$LogDir = (Join-Path $env:LOCALAPPDATA 'victory\logs'),
    [ValidateSet('series-s', 'series-x')]
    [string]$Console = 'series-s',
    [string[]]$Disable = @(),
    [switch]$NoBaseEnv,
    [switch]$DryRun,
    [string[]]$ExtraArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $ExePath) {
    $ExePath = Join-Path $RepoRoot 'target\release\victory.exe'
}
$GameExe = if (Test-Path $GameDir -PathType Leaf) { $GameDir } else { Join-Path $GameDir 'game_scarlett_master.exe' }

foreach ($name in @(
    'VICTORY_AMBIENT_STUB', 'VICTORY_SHADOWMASK_STUB', 'VICTORY_REA003_DXR',
    'VICTORY_REA003_UPSCALER', 'VICTORY_REA003_GPU_DRIVEN', 'VICTORY_REA003_SHADER_OPT',
    'VICTORY_REA003_GPU_DETILE', 'VICTORY_REA003_RENDERGRAPH_FASTPATH'
)) {
    Remove-Item "Env:$name" -ErrorAction SilentlyContinue
}

$Stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$LogFile = Join-Path $LogDir "rea003-native-passes-$Stamp.log"
$JsonFile = Join-Path $LogDir "rea003-native-passes-$Stamp.jsonl"
$MetaFile = "$LogFile.meta.json"

$EnvMap = [ordered]@{}
if (-not $NoBaseEnv) {
    $BaseEnv = [ordered]@{
        VICTORY_CARD_BUFFERS_MB            = '1024'
        VICTORY_CARD_WINDOWS_MB            = '512'
        VICTORY_TEXTURE_ROOM_PERCENT       = '72'
        VICTORY_CARD_VRAM_PERCENT          = '86'
        VICTORY_TEXTURE_STAGE_MB           = '32'
        VICTORY_CARD_ACCOUNTING            = '1'
        VICTORY_RELAY                      = '2'
        VICTORY_LATE_WAIT                  = '3'
        VICTORY_WIDE_BACK                  = '1'
        VICTORY_LIBRARY_TRIM               = '1'
        VICTORY_NO_WATCHDOG                = '1'
        VICTORY_REA_GPU_AUTHORITY          = '1'
        VICTORY_REA_READBACK_VALIDATE      = '1'
        VICTORY_REA_SPARSE_TEXTURES        = '1'
        VICTORY_REA_SPARSE_RECLAIM         = '0'
        VICTORY_REA_NATIVE_CPU_FASTPATH    = '1'
        VICTORY_REA_DIRTY_RANGES           = '1'
        VICTORY_REA_INCREMENTAL_ACCOUNTING = '1'
        VICTORY_REA_HOST_SCHEDULER         = '1'
        VICTORY_REA_ASYNC_STREAMING        = '1'
        VICTORY_REA_GPU_FASTPATH           = '1'
        VICTORY_OWN_UPSCALER               = '1'
        VICTORY_RENDER_SCALE               = '0.4'
        NO_COLOR                           = '1'
    }
    foreach ($kv in $BaseEnv.GetEnumerator()) { $EnvMap[$kv.Key] = $kv.Value }
}

$NativePassEnv = [ordered]@{
    VICTORY_HOST_MAX_WAVE            = '32'
    VICTORY_REA003_SHADER_RECOVERY   = '1'
    VICTORY_REA003_AMBIENT_NATIVE    = '1'
    VICTORY_REA003_SHADOWMASK_NATIVE = '1'
    VICTORY_REA_TELEMETRY_JSON       = $JsonFile
}
foreach ($kv in $NativePassEnv.GetEnumerator()) {
    if ($Disable -notcontains $kv.Key) { $EnvMap[$kv.Key] = $kv.Value }
}
foreach ($d in $Disable) {
    $EnvMap.Remove($d)
    Remove-Item "Env:$d" -ErrorAction SilentlyContinue
}
foreach ($kv in $EnvMap.GetEnumerator()) {
    Set-Item "Env:$($kv.Key)" -Value $kv.Value
}

$CliArgs = @('run', '--console', $Console, '--execute', $GameExe) + $ExtraArgs
Write-Host "Preset      : rea003-native-passes"
Write-Host "Executable  : $ExePath"
Write-Host "Arguments   : $($CliArgs -join ' ')"
Write-Host "Log File    : $LogFile"

if ($DryRun) { return }
if (-not (Test-Path $ExePath)) { throw "Executable not found: $ExePath" }
if (-not (Test-Path $GameExe)) { throw "Game executable not found: $GameExe" }
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Meta = [ordered]@{
    timestamp = (Get-Date).ToString('o')
    preset    = 'rea003-native-passes'
    exe       = $ExePath
    game_exe  = $GameExe
    log_file  = $LogFile
    json_file = $EnvMap['VICTORY_REA_TELEMETRY_JSON']
    disabled  = $Disable
    args      = $CliArgs
    env       = $EnvMap
}
$Meta | ConvertTo-Json -Depth 4 | Set-Content -Path $MetaFile -Encoding UTF8

$QuotedArgs = ($CliArgs | ForEach-Object { if ($_ -match '\s') { "`"$_`"" } else { $_ } }) -join ' '
$CmdLine = "`"$ExePath`" $QuotedArgs > `"$LogFile`" 2>&1"
$PrevEap = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    & cmd.exe /d /s /c "`"$CmdLine`""
    $rc = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $PrevEap
}
if ($rc -ne 0) {
    Write-Warning "victory.exe exited with code $rc. See $LogFile"
    exit $rc
}
