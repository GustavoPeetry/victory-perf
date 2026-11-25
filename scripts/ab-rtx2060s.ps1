[CmdletBinding()]
param(
    [string]$Variant = "new-default",
    [ValidateSet("preserve", "cold", "warm-copy")]
    [string]$CacheMode = "warm-copy",
    [string]$GameExe = "C:\Users\gabeg\Downloads\GTA6.Xbox.v1.0.2.1\game_scarlett_master.exe",
    [string]$LogDir = "$env:LOCALAPPDATA\victory\logs",
    [string]$Binary = "",
    [switch]$List,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$BaselineExe = Join-Path $RepoRoot "target\baseline-52a60947\victory.exe"
$ReleaseExe = Join-Path $RepoRoot "target\release\victory.exe"
$DiagExe = Join-Path $RepoRoot "target\diag\victory.exe"

$P0BaseEnv = [ordered]@{
    "VICTORY_CARD_BUFFERS_MB"      = "1024"
    "VICTORY_CARD_WINDOWS_MB"      = "512"
    "VICTORY_TEXTURE_ROOM_PERCENT" = "72"
    "VICTORY_CARD_VRAM_PERCENT"    = "86"
    "VICTORY_TEXTURE_STAGE_MB"     = "32"
    "VICTORY_CARD_ACCOUNTING"      = "1"
    "VICTORY_RELAY"                = "2"
    "VICTORY_LATE_WAIT"            = "3"
    "VICTORY_WIDE_BACK"            = "1"
    "VICTORY_LIBRARY_TRIM"         = "1"
    "VICTORY_REBAR"                = "0"
    "VICTORY_COPY_BACK_ONLY_READBACK" = "0"
    "VICTORY_NO_WATCHDOG"          = "1"
    "VICTORY_AMBIENT_STUB"         = "1"
    "VICTORY_SHADOWMASK_STUB"      = "1"
    "VICTORY_OWN_UPSCALER"         = "1"
    "VICTORY_RENDER_SCALE"         = "0.4"
}

$Variants = [ordered]@{
    # --- P0: Baseline binary (52a60947) isolated sweeps ---
    "baseline" = @{
        Group = "P0 (baseline binary)"
        Desc  = "Exact launch-victory.cmd baseline (CARD_BUFFERS_MIN_MB=1)"
        Exe   = $BaselineExe
        Env   = @{ "VICTORY_CARD_BUFFERS_MIN_MB" = "1" }
        Unset = @("VICTORY_TEXTURE_BUDGET_MB")
    }
    "min256k" = @{
        Group = "P0 (baseline binary)"
        Desc  = "Remove VICTORY_CARD_BUFFERS_MIN_MB=1 -> default 256 KiB card buffer threshold"
        Exe   = $BaselineExe
        Env   = @{}
        Unset = @("VICTORY_CARD_BUFFERS_MIN_MB", "VICTORY_TEXTURE_BUDGET_MB")
    }
    "buf1536" = @{
        Group = "P0 (baseline binary)"
        Desc  = "min256k + VICTORY_CARD_BUFFERS_MB=1536"
        Exe   = $BaselineExe
        Env   = @{ "VICTORY_CARD_BUFFERS_MB" = "1536" }
        Unset = @("VICTORY_CARD_BUFFERS_MIN_MB", "VICTORY_TEXTURE_BUDGET_MB")
    }
    "buf2048" = @{
        Group = "P0 (baseline binary)"
        Desc  = "min256k + VICTORY_CARD_BUFFERS_MB=2048"
        Exe   = $BaselineExe
        Env   = @{ "VICTORY_CARD_BUFFERS_MB" = "2048" }
        Unset = @("VICTORY_CARD_BUFFERS_MIN_MB", "VICTORY_TEXTURE_BUDGET_MB")
    }
    "vram92" = @{
        Group = "P0 (baseline binary)"
        Desc  = "min256k + VICTORY_CARD_VRAM_PERCENT=92"
        Exe   = $BaselineExe
        Env   = @{ "VICTORY_CARD_VRAM_PERCENT" = "92" }
        Unset = @("VICTORY_CARD_BUFFERS_MIN_MB", "VICTORY_TEXTURE_BUDGET_MB")
    }
    "win256" = @{
        Group = "P0 (baseline binary)"
        Desc  = "min256k + VICTORY_CARD_WINDOWS_MB=256"
        Exe   = $BaselineExe
        Env   = @{ "VICTORY_CARD_WINDOWS_MB" = "256" }
        Unset = @("VICTORY_CARD_BUFFERS_MIN_MB", "VICTORY_TEXTURE_BUDGET_MB")
    }
    "win0" = @{
        Group = "P0 (baseline binary)"
        Desc  = "min256k + VICTORY_CARD_WINDOWS_MB=0"
        Exe   = $BaselineExe
        Env   = @{ "VICTORY_CARD_WINDOWS_MB" = "0" }
        Unset = @("VICTORY_CARD_BUFFERS_MIN_MB", "VICTORY_TEXTURE_BUDGET_MB")
    }
    "tex76" = @{
        Group = "P0 (baseline binary)"
        Desc  = "min256k + VICTORY_TEXTURE_ROOM_PERCENT=76 (unsets VICTORY_TEXTURE_BUDGET_MB)"
        Exe   = $BaselineExe
        Env   = @{ "VICTORY_TEXTURE_ROOM_PERCENT" = "76" }
        Unset = @("VICTORY_CARD_BUFFERS_MIN_MB", "VICTORY_TEXTURE_BUDGET_MB")
    }
    "tex80" = @{
        Group = "P0 (baseline binary)"
        Desc  = "min256k + VICTORY_TEXTURE_ROOM_PERCENT=80 (unsets VICTORY_TEXTURE_BUDGET_MB)"
        Exe   = $BaselineExe
        Env   = @{ "VICTORY_TEXTURE_ROOM_PERCENT" = "80" }
        Unset = @("VICTORY_CARD_BUFFERS_MIN_MB", "VICTORY_TEXTURE_BUDGET_MB")
    }
    "texnowait" = @{
        Group = "P0/P1 (baseline binary)"
        Desc  = "min256k + VICTORY_TEXTURE_NO_WAIT=1 A/B test"
        Exe   = $BaselineExe
        Env   = @{ "VICTORY_TEXTURE_NO_WAIT" = "1" }
        Unset = @("VICTORY_CARD_BUFFERS_MIN_MB", "VICTORY_TEXTURE_BUDGET_MB")
    }
    "psowait500" = @{
        Group = "P1 (new release binary)"
        Desc  = "VICTORY_PIPELINE_WAIT_MS=500 vs default 12000 (use with warm-copy cache)"
        Exe   = $ReleaseExe
        Env   = @{ "VICTORY_PIPELINE_WAIT_MS" = "500" }
        Unset = @("VICTORY_CARD_BUFFERS_MIN_MB", "VICTORY_TEXTURE_BUDGET_MB")
    }
    "copyback-readback" = @{
        Group = "P1 isolated experiment"
        Desc  = "VICTORY_COPY_BACK_ONLY_READBACK=1 (isolated visual/progression check)"
        Exe   = $ReleaseExe
        Env   = @{ "VICTORY_COPY_BACK_ONLY_READBACK" = "1" }
        Unset = @("VICTORY_CARD_BUFFERS_MIN_MB", "VICTORY_TEXTURE_BUDGET_MB")
    }

    # --- P1/P2: New branch builds (perf/rtx2060s-8gb-deltas) ---
    "new-default" = @{
        Group = "P1/P2 (new release binary)"
        Desc  = "All default-on deltas (slot flags, xmem coalesce, CleanBlocks, Uncommitted memo, defer unmake, word merge, 4 NVIDIA wave-wide PSOs, mesh link prune)"
        Exe   = $ReleaseExe
        Env   = @{}
        Unset = @("VICTORY_CARD_BUFFERS_MIN_MB", "VICTORY_TEXTURE_BUDGET_MB")
    }
    "new-diag" = @{
        Group = "P1/P2 (new diag binary)"
        Desc  = "Diagnostics build with full P0/P1/P2 telemetry, DRED breadcrumbs, and GPU timing hooks"
        Exe   = $DiagExe
        Env   = @{}
        Unset = @("VICTORY_CARD_BUFFERS_MIN_MB", "VICTORY_TEXTURE_BUDGET_MB")
    }
    "new-scoped-remap" = @{
        Group = "P1 opt-in (new release binary)"
        Desc  = "new-default + VICTORY_TEXTURE_SCOPED_REMAP=1"
        Exe   = $ReleaseExe
        Env   = @{ "VICTORY_TEXTURE_SCOPED_REMAP" = "1" }
        Unset = @("VICTORY_CARD_BUFFERS_MIN_MB", "VICTORY_TEXTURE_BUDGET_MB")
    }
    "new-join-parts" = @{
        Group = "P1 opt-in (new release binary)"
        Desc  = "new-default + VICTORY_JOIN_PARTS=1"
        Exe   = $ReleaseExe
        Env   = @{ "VICTORY_JOIN_PARTS" = "1" }
        Unset = @("VICTORY_CARD_BUFFERS_MIN_MB", "VICTORY_TEXTURE_BUDGET_MB")
    }
    "new-indirect-gather" = @{
        Group = "P1 opt-in (new release binary)"
        Desc  = "new-default + VICTORY_INDIRECT_GATHER=1"
        Exe   = $ReleaseExe
        Env   = @{ "VICTORY_INDIRECT_GATHER" = "1" }
        Unset = @("VICTORY_CARD_BUFFERS_MIN_MB", "VICTORY_TEXTURE_BUDGET_MB")
    }
    "new-all-optin" = @{
        Group = "P1 opt-in (new release binary)"
        Desc  = "new-default + SCOPED_REMAP=1 + JOIN_PARTS=1 + INDIRECT_GATHER=1"
        Exe   = $ReleaseExe
        Env   = @{
            "VICTORY_TEXTURE_SCOPED_REMAP" = "1"
            "VICTORY_JOIN_PARTS"           = "1"
            "VICTORY_INDIRECT_GATHER"      = "1"
        }
        Unset = @("VICTORY_CARD_BUFFERS_MIN_MB", "VICTORY_TEXTURE_BUDGET_MB")
    }

    # --- Rollbacks on new release binary ---
    "no-wave-wide" = @{
        Group = "P2 rollback"
        Desc  = "Disable 4 NVIDIA wave-wide compute PSOs (VICTORY_WAVE_WIDE_ONLY=none)"
        Exe   = $ReleaseExe
        Env   = @{ "VICTORY_WAVE_WIDE_ONLY" = "none" }
        Unset = @("VICTORY_CARD_BUFFERS_MIN_MB", "VICTORY_TEXTURE_BUDGET_MB")
    }
    "no-mesh-prune" = @{
        Group = "P2 rollback"
        Desc  = "Disable mesh/pixel linked output pruning (VICTORY_MESH_LINK_PRUNE=0)"
        Exe   = $ReleaseExe
        Env   = @{ "VICTORY_MESH_LINK_PRUNE" = "0" }
        Unset = @("VICTORY_CARD_BUFFERS_MIN_MB", "VICTORY_TEXTURE_BUDGET_MB")
    }
    "no-slot-flags" = @{
        Group = "P1 rollback"
        Desc  = "Disable descriptor slot flags / heap_buffers dedup (VICTORY_HEAP_SLOT_FLAGS=0)"
        Exe   = $ReleaseExe
        Env   = @{ "VICTORY_HEAP_SLOT_FLAGS" = "0" }
        Unset = @("VICTORY_CARD_BUFFERS_MIN_MB", "VICTORY_TEXTURE_BUDGET_MB")
    }
    "no-uncommitted-memo" = @{
        Group = "P1 rollback"
        Desc  = "Disable per-frame uncommitted block memo (VICTORY_UNCOMMITTED_MEMO=0)"
        Exe   = $ReleaseExe
        Env   = @{ "VICTORY_UNCOMMITTED_MEMO" = "0" }
        Unset = @("VICTORY_CARD_BUFFERS_MIN_MB", "VICTORY_TEXTURE_BUDGET_MB")
    }
    "no-xmem-coalesce" = @{
        Group = "P1 rollback"
        Desc  = "Disable xmem::map_alias page-run coalescing (VICTORY_XMEM_COALESCE_ALIAS=0)"
        Exe   = $ReleaseExe
        Env   = @{ "VICTORY_XMEM_COALESCE_ALIAS" = "0" }
        Unset = @("VICTORY_CARD_BUFFERS_MIN_MB", "VICTORY_TEXTURE_BUDGET_MB")
    }
    "no-defer-unmake" = @{
        Group = "P1 rollback"
        Desc  = "Disable deferred unmake while submissions fly (VICTORY_DEFER_UNMAKE=0)"
        Exe   = $ReleaseExe
        Env   = @{ "VICTORY_DEFER_UNMAKE" = "0" }
        Unset = @("VICTORY_CARD_BUFFERS_MIN_MB", "VICTORY_TEXTURE_BUDGET_MB")
    }
    "no-word-merge" = @{
        Group = "P1 rollback"
        Desc  = "Disable 8-byte word copy-back merge (VICTORY_COPY_BACK_WORD_MERGE=0)"
        Exe   = $ReleaseExe
        Env   = @{ "VICTORY_COPY_BACK_WORD_MERGE" = "0" }
        Unset = @("VICTORY_CARD_BUFFERS_MIN_MB", "VICTORY_TEXTURE_BUDGET_MB")
    }
}

if ($List) {
    Write-Output "Available A/B variants for RTX 2060 SUPER 8 GB:"
    foreach ($k in $Variants.Keys) {
        $v = $Variants[$k]
        Write-Output ("  {0,-20} [{1}] {2}" -f $k, $v.Group, $v.Desc)
    }
    Write-Output ""
    Write-Output "Binaries:"
    foreach ($pair in @(
        @("baseline", $BaselineExe),
        @("release",  $ReleaseExe),
        @("diag",     $DiagExe)
    )) {
        $p = $pair[1]
        if (Test-Path $p) {
            $sha = (Get-FileHash -Algorithm SHA256 $p).Hash
            $len = (Get-Item $p).Length
            Write-Output ("  {0,-9} {1} ({2} B) SHA256={3}" -f $pair[0], $p, $len, $sha)
        } else {
            Write-Output ("  {0,-9} {1} (missing)" -f $pair[0], $p)
        }
    }
    exit 0
}

if (-not $Variants.Contains($Variant)) {
    throw "Unknown variant '$Variant'. Run with -List to see valid variants."
}

$Spec = $Variants[$Variant]
$ExePath = if ($Binary -ne "") { $Binary } else { $Spec.Exe }
if (-not (Test-Path $ExePath)) {
    throw "Victory binary not found at: $ExePath"
}
if (-not (Test-Path $GameExe)) {
    Write-Warning "Title executable not found at: $GameExe"
}

$ExeSha = (Get-FileHash -Algorithm SHA256 $ExePath).Hash
$GitHead = (git -C $RepoRoot rev-parse HEAD 2>$null)
if (-not $GitHead) { $GitHead = "unknown" }

$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
}
$LogFile = Join-Path $LogDir "victory-ab-$Variant-$CacheMode-$Stamp.log"
$MetaFile = "$LogFile.meta.json"

# Isolated cache handling: never modify %LOCALAPPDATA%\victory\cache or %LOCALAPPDATA%\victory\data.
$DefaultCacheDir = Join-Path $env:LOCALAPPDATA "victory\cache"
$IsolatedCacheDir = $null
if ($CacheMode -eq "cold") {
    $IsolatedCacheDir = Join-Path $env:LOCALAPPDATA "victory\ab-caches\cold-$Variant-$Stamp"
} elseif ($CacheMode -eq "warm-copy") {
    $IsolatedCacheDir = Join-Path $env:LOCALAPPDATA "victory\ab-caches\warm-$Variant-$Stamp"
}

# Build effective VICTORY_* environment map cleanly.
$EffectiveEnv = [ordered]@{}
foreach ($k in $P0BaseEnv.Keys) {
    $EffectiveEnv[$k] = $P0BaseEnv[$k]
}
foreach ($k in $Spec.Unset) {
    if ($EffectiveEnv.Contains($k)) { $EffectiveEnv.Remove($k) }
}
foreach ($k in $Spec.Env.Keys) {
    $EffectiveEnv[$k] = $Spec.Env[$k]
}
if ($IsolatedCacheDir) {
    $EffectiveEnv["VICTORY_CACHE_DIR"] = $IsolatedCacheDir
}
$EffectiveEnv["NO_COLOR"] = "1"

$CmdArgs = @("run", "--console", "series-s", "--execute", $GameExe)

Write-Output "=== VIctory RTX 2060 SUPER A/B Run ==="
Write-Output "Variant     : $Variant ($($Spec.Desc))"
Write-Output "Git HEAD    : $GitHead"
Write-Output "Binary      : $ExePath"
Write-Output "SHA256      : $ExeSha"
Write-Output "CacheMode   : $CacheMode"
if ($IsolatedCacheDir) {
    Write-Output "CacheDir    : $IsolatedCacheDir (source untouched: $DefaultCacheDir)"
} else {
    Write-Output "CacheDir    : $DefaultCacheDir (preserve)"
}
Write-Output "LogFile     : $LogFile"
Write-Output "Effective env:"
foreach ($k in $EffectiveEnv.Keys) {
    Write-Output ("  {0}={1}" -f $k, $EffectiveEnv[$k])
}
Write-Output ("Command     : `"{0}`" {1}" -f $ExePath, ($CmdArgs -join " "))

if ($DryRun) {
    Write-Output "[DryRun] Not creating directories or launching title."
    exit 0
}

if ($CacheMode -eq "cold") {
    New-Item -ItemType Directory -Force -Path $IsolatedCacheDir | Out-Null
} elseif ($CacheMode -eq "warm-copy") {
    if (Test-Path $DefaultCacheDir) {
        New-Item -ItemType Directory -Force -Path $IsolatedCacheDir | Out-Null
        foreach ($sub in @("images", "shaders", "pipelines")) {
            $srcSub = Join-Path $DefaultCacheDir $sub
            if (Test-Path $srcSub) {
                Copy-Item -Path $srcSub -Destination (Join-Path $IsolatedCacheDir $sub) -Recurse -Force
            }
        }
    } else {
        Write-Warning "Default cache '$DefaultCacheDir' does not exist; starting empty isolated cache."
        New-Item -ItemType Directory -Force -Path $IsolatedCacheDir | Out-Null
    }
}

$Meta = [ordered]@{
    timestamp   = (Get-Date).ToString("o")
    variant     = $Variant
    description = $Spec.Desc
    git_head    = $GitHead
    binary      = $ExePath
    sha256      = $ExeSha
    cache_mode  = $CacheMode
    cache_dir   = $(if ($IsolatedCacheDir) { $IsolatedCacheDir } else { $DefaultCacheDir })
    game_exe    = $GameExe
    log_file    = $LogFile
    env         = $EffectiveEnv
    unset       = $Spec.Unset
}
$Meta | ConvertTo-Json -Depth 5 | Set-Content -Path $MetaFile -Encoding UTF8

# Apply environment only inside a child-safe scope and restore afterward.
$SavedEnv = @{}
$AllTouched = @($EffectiveEnv.Keys) + @($Spec.Unset) + @("VICTORY_CACHE_DIR") | Select-Object -Unique
foreach ($k in $AllTouched) {
    $SavedEnv[$k] = [System.Environment]::GetEnvironmentVariable($k, "Process")
}
try {
    foreach ($k in $Spec.Unset) {
        [System.Environment]::SetEnvironmentVariable($k, $null, "Process")
    }
    if (-not $IsolatedCacheDir) {
        [System.Environment]::SetEnvironmentVariable("VICTORY_CACHE_DIR", $null, "Process")
    }
    foreach ($k in $EffectiveEnv.Keys) {
        [System.Environment]::SetEnvironmentVariable($k, $EffectiveEnv[$k], "Process")
    }
    Push-Location $RepoRoot
    try {
        & $ExePath @CmdArgs *>&1 | Out-File -FilePath $LogFile -Encoding utf8
        $Rc = $LASTEXITCODE
    } finally {
        Pop-Location
    }
} finally {
    foreach ($k in $SavedEnv.Keys) {
        [System.Environment]::SetEnvironmentVariable($k, $SavedEnv[$k], "Process")
    }
}

Write-Output "Run finished (exit code=$Rc). Analyze with:"
Write-Output "  python scripts/analyze-victory-log.py `"$LogFile`""
exit $Rc
