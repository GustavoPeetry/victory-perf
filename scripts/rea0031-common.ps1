<#
.SYNOPSIS
    Shared isolated deterministic launcher utilities for REA-003.1 verification presets.
#>
Set-StrictMode -Version Latest

function Get-Rea0031RepoRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
}

function Resolve-Rea0031GameExe {
    param([string]$GameDir = 'C:\Users\gabeg\Downloads\GTA6.Xbox.v1.0.2.1')
    if (Test-Path $GameDir -PathType Leaf) { return $GameDir }
    return (Join-Path $GameDir 'game_scarlett_master.exe')
}

function Get-Rea0031BaseEnv {
    param(
        [string]$RenderScale = '0.4',
        [switch]$OwnUpscaler,
        [switch]$WithStubs
    )
    $envMap = [ordered]@{
        VICTORY_CARD_BUFFERS_MB         = '1024'
        VICTORY_CARD_WINDOWS_MB         = '512'
        VICTORY_TEXTURE_ROOM_PERCENT    = '72'
        VICTORY_CARD_VRAM_PERCENT       = '86'
        VICTORY_TEXTURE_STAGE_MB        = '32'
        VICTORY_CARD_ACCOUNTING         = '1'
        VICTORY_RELAY                   = '2'
        VICTORY_LATE_WAIT               = '3'
        VICTORY_WIDE_BACK               = '1'
        VICTORY_LIBRARY_TRIM            = '1'
        VICTORY_REBAR                   = '0'
        VICTORY_COPY_BACK_ONLY_READBACK = '0'
        VICTORY_NO_WATCHDOG             = '1'
        VICTORY_RENDER_SCALE            = $RenderScale
        NO_COLOR                        = '1'
    }
    if ($OwnUpscaler) {
        $envMap['VICTORY_OWN_UPSCALER'] = '1'
    } else {
        $envMap['VICTORY_OWN_UPSCALER'] = '0'
    }
    if ($WithStubs) {
        $envMap['VICTORY_AMBIENT_STUB'] = '1'
        $envMap['VICTORY_SHADOWMASK_STUB'] = '1'
    }
    return $envMap
}

function Invoke-VictoryIsolated {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$PresetName,
        [Parameter(Mandatory)][string]$ExePath,
        [Parameter(Mandatory)][string]$CacheBucket,
        [Parameter(Mandatory)][System.Collections.IDictionary]$EnvMap,
        [string]$LogDir = (Join-Path $env:LOCALAPPDATA 'victory\logs'),
        [string]$GameDir = 'C:\Users\gabeg\Downloads\GTA6.Xbox.v1.0.2.1',
        [string]$Console = 'series-s',
        [string]$ManifestRole = '',
        [string[]]$CliArgsOverride,
        [switch]$NoBaseEnv,
        [string[]]$Disable = @(),
        [switch]$DryRun,
        [string[]]$ExtraArgs = @()
    )

    $repoRoot = Get-Rea0031RepoRoot
    $gameExe = Resolve-Rea0031GameExe -GameDir $GameDir
    $resolvedExe = [System.IO.Path]::GetFullPath($ExePath)

    if (-not (Test-Path $resolvedExe -PathType Leaf)) {
        throw "Executable not found for preset '$PresetName': $resolvedExe"
    }
    $exeHash = (Get-FileHash -Path $resolvedExe -Algorithm SHA256).Hash.ToLowerInvariant()
    $exeInfo = Get-Item -Path $resolvedExe

    # Optional cross-check against research/REA-003.1/EXECUTABLE_MANIFEST.json once generated.
    if ($ManifestRole) {
        $manifestPath = Join-Path $repoRoot 'research\REA-003.1\EXECUTABLE_MANIFEST.json'
        if (Test-Path $manifestPath -PathType Leaf) {
            try {
                $manifest = Get-Content -Raw -Path $manifestPath | ConvertFrom-Json
                $entry = $manifest.executables.$ManifestRole
                if ($null -ne $entry -and $entry.sha256) {
                    $expectedHash = "$($entry.sha256)".ToLowerInvariant()
                    if ($expectedHash -ne $exeHash) {
                        throw "SHA-256 mismatch for manifest role '$ManifestRole' ($resolvedExe): expected $expectedHash, got $exeHash"
                    }
                }
            } catch [System.Management.Automation.RuntimeException] {
                throw
            } catch {
                Write-Warning "Could not validate EXECUTABLE_MANIFEST.json: $_"
            }
        }
    }

    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $runGuid = [Guid]::NewGuid().ToString('N').Substring(0, 8)
    $runId = "$PresetName-$stamp-$runGuid"
    $logFile = Join-Path $LogDir "$PresetName-$stamp.log"
    $jsonFile = Join-Path $LogDir "$PresetName-$stamp.jsonl"
    $metaFile = "$logFile.meta.json"
    $cacheRoot = Join-Path (Join-Path $env:LOCALAPPDATA 'victory\cache\rea0031') $CacheBucket

    $cliArgs = if ($CliArgsOverride -and $CliArgsOverride.Count -gt 0) {
        @($CliArgsOverride) + $ExtraArgs
    } else {
        @('run', '--console', $Console, '--execute', $gameExe) + $ExtraArgs
    }

    $effectiveEnv = [ordered]@{}
    if (-not $NoBaseEnv) {
        foreach ($kv in $EnvMap.GetEnumerator()) {
            if ($Disable -notcontains $kv.Key) {
                $effectiveEnv[$kv.Key] = "$($kv.Value)"
            }
        }
    } else {
        foreach ($kv in $EnvMap.GetEnumerator()) {
            if ($kv.Key.StartsWith('VICTORY_REA') -or $kv.Key.StartsWith('VICTORY_D3D12') -or $kv.Key -eq 'NO_COLOR') {
                if ($Disable -notcontains $kv.Key) {
                    $effectiveEnv[$kv.Key] = "$($kv.Value)"
                }
            }
        }
    }
    foreach ($d in $Disable) {
        $null = $effectiveEnv.Remove($d)
    }
    $effectiveEnv['VICTORY_PRESET'] = $PresetName
    $effectiveEnv['VICTORY_RUN_ID'] = $runId
    $effectiveEnv['VICTORY_RUNTIME_SHA256'] = $exeHash
    $effectiveEnv['VICTORY_CACHE_DIR'] = $cacheRoot
    if (-not $effectiveEnv.Contains('VICTORY_REA_TELEMETRY_JSON')) {
        $effectiveEnv['VICTORY_REA_TELEMETRY_JSON'] = $jsonFile
    }

    # Snapshot every VICTORY_* + NO_COLOR in current process and restore unconditionally in finally.
    $prevEnv = @{}
    foreach ($item in Get-ChildItem Env: -ErrorAction SilentlyContinue) {
        if ($item.Name.StartsWith('VICTORY_', [System.StringComparison]::OrdinalIgnoreCase) -or $item.Name -eq 'NO_COLOR') {
            $prevEnv[$item.Name] = $item.Value
            Remove-Item "Env:$($item.Name)" -ErrorAction SilentlyContinue
        }
    }

    $gitCommit = 'unknown'
    try {
        $gitCommit = (git -C $repoRoot rev-parse --short=12 HEAD 2>$null | Out-String).Trim()
        if (-not $gitCommit) { $gitCommit = 'unknown' }
    } catch {
        $gitCommit = 'unknown'
    }

    $prevEap = $ErrorActionPreference
    try {
        foreach ($kv in $effectiveEnv.GetEnumerator()) {
            Set-Item "Env:$($kv.Key)" -Value $kv.Value
        }

        $resolvedChild = [ordered]@{}
        foreach ($item in (Get-ChildItem Env: -ErrorAction SilentlyContinue | Sort-Object Name)) {
            if ($item.Name.StartsWith('VICTORY_', [System.StringComparison]::OrdinalIgnoreCase) -or $item.Name -eq 'NO_COLOR') {
                $resolvedChild[$item.Name] = $item.Value
            }
        }

        Write-Host "Preset      : $PresetName"
        Write-Host "Run ID      : $runId"
        Write-Host "Executable  : $resolvedExe"
        Write-Host "SHA-256     : $exeHash"
        Write-Host "Git Commit  : $gitCommit"
        Write-Host "Cache Bucket: $cacheRoot"
        Write-Host "Arguments   : $($cliArgs -join ' ')"
        Write-Host "Log File    : $logFile"
        Write-Host "Telemetry   : $($effectiveEnv['VICTORY_REA_TELEMETRY_JSON'])"

        if ($DryRun) {
            return
        }

        if ($cliArgs[0] -eq 'run' -and -not (Test-Path $gameExe -PathType Leaf)) {
            throw "Game executable not found: $gameExe"
        }
        New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
        New-Item -ItemType Directory -Force -Path $cacheRoot | Out-Null

        $meta = [ordered]@{
            timestamp       = (Get-Date).ToString('o')
            preset          = $PresetName
            run_id          = $runId
            git_commit      = $gitCommit
            exe             = $resolvedExe
            exe_sha256      = $exeHash
            exe_size_bytes  = $exeInfo.Length
            game_exe        = $gameExe
            cache_bucket    = $CacheBucket
            cache_dir       = $cacheRoot
            log_file        = $logFile
            json_file       = $effectiveEnv['VICTORY_REA_TELEMETRY_JSON']
            disabled        = $Disable
            args            = $cliArgs
            requested_env   = $effectiveEnv
            resolved_env    = $resolvedChild
        }
        $meta | ConvertTo-Json -Depth 5 | Set-Content -Path $metaFile -Encoding UTF8

        $quotedArgs = ($cliArgs | ForEach-Object { if ($_ -match '\s') { "`"$_`"" } else { $_ } }) -join ' '
        $cmdLine = "`"$resolvedExe`" $quotedArgs > `"$logFile`" 2>&1"
        $ErrorActionPreference = 'Continue'
        & cmd.exe /d /s /c "`"$cmdLine`""
        $rc = $LASTEXITCODE
        if ($rc -ne 0) {
            Write-Warning "victory.exe exited with code $rc. See $logFile"
            exit $rc
        }
    } finally {
        $ErrorActionPreference = $prevEap
        foreach ($item in Get-ChildItem Env: -ErrorAction SilentlyContinue) {
            if ($item.Name.StartsWith('VICTORY_', [System.StringComparison]::OrdinalIgnoreCase) -or $item.Name -eq 'NO_COLOR') {
                Remove-Item "Env:$($item.Name)" -ErrorAction SilentlyContinue
            }
        }
        foreach ($kv in $prevEnv.GetEnumerator()) {
            Set-Item "Env:$($kv.Key)" -Value $kv.Value
        }
    }
}
