<#
.SYNOPSIS
    Launches baseline VIctory with every VICTORY_REA_* feature switch cleared.
#>
[CmdletBinding()]
param(
    [string]$ExePath,
    [string]$GameDir = 'C:\Users\gabeg\Downloads\GTA6.Xbox.v1.0.2.1',
    [string]$LogDir = (Join-Path $env:LOCALAPPDATA 'victory\logs'),
    [ValidateSet('series-s', 'series-x')]
    [string]$Console = 'series-s',
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

$ReaVars = @(
    'VICTORY_REA_SPARSE_TEXTURES',
    'VICTORY_REA_SPARSE_RECLAIM',
    'VICTORY_REA_READBACK_TRACE',
    'VICTORY_REA_SEMANTIC_READBACK',
    'VICTORY_REA_READBACK_VALIDATE',
    'VICTORY_REA_DIRTY_RANGES',
    'VICTORY_REA_INCREMENTAL_ACCOUNTING',
    'VICTORY_REA_ACCOUNTING_AUDIT',
    'VICTORY_REA_NATIVE_CPU_FASTPATH',
    'VICTORY_REA_HOST_SCHEDULER',
    'VICTORY_REA_GPU_AUTHORITY',
    'VICTORY_REA_ASYNC_STREAMING',
    'VICTORY_REA_COPY_QUEUE',
    'VICTORY_REA_GPU_FASTPATH',
    'VICTORY_REA_DXR_PROTO',
    'VICTORY_REA_FFX_TRACE',
    'VICTORY_REA_TELEMETRY_JSON',
    'VICTORY_DYNAMIC_SLOTS'
)

foreach ($name in $ReaVars) {
    Remove-Item "Env:$name" -ErrorAction SilentlyContinue
}

$EnvMap = [ordered]@{}
if (-not $NoBaseEnv) {
    $BaseEnv = [ordered]@{
        VICTORY_CARD_BUFFERS_MB      = '1024'
        VICTORY_CARD_WINDOWS_MB      = '512'
        VICTORY_TEXTURE_ROOM_PERCENT = '72'
        VICTORY_CARD_VRAM_PERCENT    = '86'
        VICTORY_TEXTURE_STAGE_MB     = '32'
        VICTORY_CARD_ACCOUNTING      = '1'
        VICTORY_RELAY                = '2'
        VICTORY_LATE_WAIT            = '3'
        VICTORY_WIDE_BACK            = '1'
        VICTORY_LIBRARY_TRIM         = '1'
        VICTORY_REBAR                = '0'
        VICTORY_COPY_BACK_ONLY_READBACK = '0'
        VICTORY_NO_WATCHDOG          = '1'
        VICTORY_AMBIENT_STUB         = '1'
        VICTORY_SHADOWMASK_STUB      = '1'
        VICTORY_OWN_UPSCALER         = '1'
        VICTORY_RENDER_SCALE         = '0.4'
        NO_COLOR                     = '1'
    }
    foreach ($kv in $BaseEnv.GetEnumerator()) {
        $EnvMap[$kv.Key] = $kv.Value
        Set-Item "Env:$($kv.Key)" -Value $kv.Value
    }
}

$Stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$LogFile = Join-Path $LogDir "rea002-baseline-$Stamp.log"
$MetaFile = "$LogFile.meta.json"
$CliArgs = @('run', '--console', $Console, '--execute', $GameExe) + $ExtraArgs

Write-Host "Preset      : baseline (all VICTORY_REA_* unset)"
Write-Host "Executable  : $ExePath"
Write-Host "Arguments   : $($CliArgs -join ' ')"
Write-Host "Log File    : $LogFile"
Write-Host "REA Env Set : <none>"

if ($DryRun) {
    return
}

if (-not (Test-Path $ExePath)) { throw "Executable not found: $ExePath" }
if (-not (Test-Path $GameExe)) { throw "Game executable not found: $GameExe" }
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Meta = [ordered]@{
    timestamp = (Get-Date).ToString('o')
    preset    = 'baseline'
    exe       = $ExePath
    game_exe  = $GameExe
    log_file  = $LogFile
    args      = $CliArgs
    env       = $EnvMap
    unset_rea = $ReaVars
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
