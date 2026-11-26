Set-StrictMode -Version Latest

function Resolve-Rea0032Path {
    param([Parameter(Mandatory)][string]$Path)

    return [System.IO.Path]::GetFullPath(
        $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path)
    )
}

function Assert-Rea0032PlainPath {
    param([Parameter(Mandatory)][string]$Path)

    $item = Get-Item -LiteralPath $Path -Force
    while ($null -ne $item) {
        if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
            throw "Reparse point in input path: $($item.FullName)"
        }
        if ($item -is [System.IO.DirectoryInfo]) {
            $item = $item.Parent
        } else {
            $item = $item.Directory
        }
    }
}

function Copy-Rea0032Seed {
    param([string]$Source, [string]$Destination)

    if (-not $Source) { return }
    Assert-Rea0032PlainPath $Source
    $junctions = @(Get-ChildItem -LiteralPath $Source -Force -Recurse -Attributes ReparsePoint)
    if ($junctions.Count -ne 0) {
        throw "Reparse point in seed directory: $($junctions[0].FullName)"
    }
    foreach ($item in Get-ChildItem -LiteralPath $Source -Force) {
        Copy-Item -LiteralPath $item.FullName -Destination $Destination -Force -Recurse
    }
}

function Stop-Rea0032OwnedProcess {
    param([System.Diagnostics.Process]$Process)

    if (-not $Process.HasExited) {
        try { $Process.Kill() }
        catch [System.InvalidOperationException] { if (-not $Process.HasExited) { throw } }
    }
    $Process.WaitForExit()
}

function Start-Rea0032ReadOnlyMount {
    param([string]$Source, [string]$RunDir, [string]$Label)

    Assert-Rea0032PlainPath $Source
    $origin = [IO.DriveInfo]::new([IO.Path]::GetPathRoot($Source))
    if ($origin.DriveType -notin @([IO.DriveType]::Fixed, [IO.DriveType]::Removable)) {
        throw "ReadOnlySource must be on a local disk: $Source"
    }
    $rclone = (Get-Command rclone.exe -CommandType Application -ErrorAction Stop).Source
    $used = @([IO.DriveInfo]::GetDrives() | ForEach-Object { $_.Name })
    $driveRoot = 90..75 | ForEach-Object { '{0}:\' -f [char]$_ } |
        Where-Object { $_ -notin $used } | Select-Object -First 1
    if (-not $driveRoot) { throw 'No free drive letter from K: to Z: for read-only mounting.' }
    $mountPoint = $driveRoot.TrimEnd('\')
    $config = Join-Path $RunDir 'rclone-empty.conf'
    $log = Join-Path $RunDir 'rclone.log'
    [IO.File]::WriteAllText($config, '')
    $arguments = @(
        'mount', $Source.TrimEnd('\'), $mountPoint, '--read-only', '--vfs-cache-mode', 'off',
        '--buffer-size', '0', '--config', $config, '--volname', $Label, '--log-file', $log
    )
    # Source is a directory below a drive root; the remaining values cannot
    # contain double quotes or end in a backslash.
    $info = [System.Diagnostics.ProcessStartInfo]::new($rclone)
    $info.Arguments = ($arguments | ForEach-Object { '"' + $_ + '"' }) -join ' '
    $info.WorkingDirectory = $RunDir
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    foreach ($name in @($info.EnvironmentVariables.Keys)) {
        if ($name -like 'RCLONE_*') { $info.EnvironmentVariables.Remove($name) }
    }

    $child = [System.Diagnostics.Process]::Start($info)
    $ready = $false
    try {
        $disk = [IO.DriveInfo]::new($driveRoot)
        $clock = [System.Diagnostics.Stopwatch]::StartNew()
        while (-not $child.HasExited -and $clock.Elapsed.TotalSeconds -lt 90) {
            try {
                $ready = $disk.IsReady -and $disk.VolumeLabel -eq $Label
            } catch [IO.IOException] {
                $ready = $false
            }
            if ($ready) {
                return @{ process = $child; root = $driveRoot; log = $log }
            }
            Start-Sleep -Milliseconds 250
        }
        throw "Read-only rclone mount did not become ready at ${driveRoot}. See $log"
    } finally {
        if (-not $ready) {
            Stop-Rea0032OwnedProcess $child
            $child.Dispose()
        }
    }
}

function Assert-Rea0032Title {
    param([string]$Root, [string]$Exe, [string]$ExpectedHash)

    if (-not (Test-Path -LiteralPath $Exe -PathType Leaf)) {
        throw "Title executable not found: $Exe"
    }
    Assert-Rea0032PlainPath $Exe
    if ((Get-FileHash -LiteralPath $Exe -Algorithm SHA256).Hash -ne $ExpectedHash) {
        throw "Unexpected title build at $Exe. The launcher expects version 1.0.2.1."
    }
    $archiveSizes = [ordered]@{
        'anim.rpf'      = 4407626384
        'audio.rpf'     = 28543454672
        'common.rpf'    = 6642192
        'data.rpf'      = 481544304
        'levels.rpf'    = 76155065056
        'movies.rpf'    = 8832481776
        'navmeshes.rpf' = 6707512496
        'packs.rpf'     = 4798213264
        'script.rpf'    = 4358960
        'shaders.rpf'   = 411695600
        'textures.rpf'  = 192585024
    }
    foreach ($name in $archiveSizes.Keys) {
        $archive = Join-Path $Root $name
        if (-not (Test-Path -LiteralPath $archive -PathType Leaf)) {
            throw "Title directory is missing $name"
        }
        Assert-Rea0032PlainPath $archive
        if ((Get-Item -LiteralPath $archive).Length -ne $archiveSizes[$name]) {
            throw "Title directory has an unexpected or incomplete $name"
        }
    }
}

function Initialize-Rea0032HostMonitor {
    if ('Rea0032MemoryMonitor' -as [type]) { return }

    # GetPerformanceInfo exposes system commit directly. Compile once, outside
    # the timed run, rather than querying a performance provider in each sample.
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;

public static class Rea0032MemoryMonitor
{
    [StructLayout(LayoutKind.Sequential)]
    private struct PerformanceInformation
    {
        public uint Size;
        public UIntPtr CommitTotal, CommitLimit, CommitPeak;
        public UIntPtr PhysicalTotal, PhysicalAvailable, SystemCache;
        public UIntPtr KernelTotal, KernelPaged, KernelNonpaged, PageSize;
        public uint HandleCount, ProcessCount, ThreadCount;
    }

    [DllImport("psapi.dll", ExactSpelling = true)]
    private static extern bool GetPerformanceInfo(out PerformanceInformation info, uint size);

    public static ulong[] Read()
    {
        PerformanceInformation info;
        if (!GetPerformanceInfo(out info, (uint)Marshal.SizeOf(typeof(PerformanceInformation)))) return null;
        ulong page = info.PageSize.ToUInt64();
        return new ulong[] {
            info.PhysicalAvailable.ToUInt64() * page,
            info.CommitTotal.ToUInt64() * page,
            info.CommitLimit.ToUInt64() * page
        };
    }
}
'@
}

function Write-Rea0032HostSample {
    param(
        [System.Diagnostics.Process]$Process,
        [long]$ElapsedMs,
        [long]$PreviousElapsedMs,
        [double]$PreviousCpuMs,
        [string]$LogPath
    )

    try {
        $Process.Refresh()
        $cpuMs = $Process.TotalProcessorTime.TotalMilliseconds
        $privateBytes = $Process.PrivateMemorySize64
        $workingSet = $Process.WorkingSet64
    } catch {
        return $PreviousCpuMs
    }
    $logicalProcessors = [Environment]::ProcessorCount
    $sample = [ordered]@{
        timestamp               = [DateTimeOffset]::Now.ToString('o')
        elapsed_ms              = $ElapsedMs
        interval_ms             = $ElapsedMs - $PreviousElapsedMs
        pid                     = $Process.Id
        logical_processors      = $logicalProcessors
        cpu_total_ms            = $cpuMs
        cpu_percent_of_host     = [Math]::Round(
            100 * ($cpuMs - $PreviousCpuMs) / ($ElapsedMs - $PreviousElapsedMs) / $logicalProcessors, 2
        )
        private_bytes           = $privateBytes
        working_set_bytes       = $workingSet
        available_physical_bytes = $null
        committed_bytes         = $null
        commit_limit_bytes      = $null
    }
    if ('Rea0032MemoryMonitor' -as [type]) {
        $systemMemory = [Rea0032MemoryMonitor]::Read()
        if ($null -ne $systemMemory) {
            $sample.available_physical_bytes = $systemMemory[0]
            $sample.committed_bytes = $systemMemory[1]
            $sample.commit_limit_bytes = $systemMemory[2]
        }
    }
    $sample | ConvertTo-Json -Compress | Add-Content -LiteralPath $LogPath -Encoding Ascii
    return $cpuMs
}

function Invoke-Rea0032Run {
    [CmdletBinding(DefaultParameterSetName = 'Copy')]
    param(
        [Parameter(Mandatory)][ValidateSet('baseline', 'stable', 'experimental')][string]$Profile,
        [Parameter(ParameterSetName = 'Copy')][string]$GameDir,
        [Parameter(Mandatory, ParameterSetName = 'ReadOnly')][string]$ReadOnlySource,
        [string]$ExePath,
        [string]$CacheSeed,
        [string]$DataSeed,
        [ValidateSet('series-s', 'series-x')][string]$Console = 'series-s',
        [string]$RenderScale = '0.4',
        [ValidateRange(0, 86400)][int]$BudgetSeconds = 0,
        [switch]$DryRun
    )

    if ($RenderScale -notmatch '^(0(?:\.\d+)?|1(?:\.0+)?)$' -or
        [double]::Parse($RenderScale, [System.Globalization.CultureInfo]::InvariantCulture) -le 0) {
        throw 'RenderScale must be greater than zero and at most 1, with a decimal point.'
    }
    $repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
    $experimentRoot = Join-Path $env:LOCALAPPDATA 'victory\rea0032'
    $originalRoot = Join-Path $env:USERPROFILE 'Downloads\GTA6.Xbox.v1.0.2.1'
    $sourceMode = 'supplied-directory'
    if ($ReadOnlySource) {
        $ReadOnlySource = Resolve-Rea0032Path $ReadOnlySource
        if ($ReadOnlySource.StartsWith('\\') -or $ReadOnlySource.TrimEnd('\') -eq
            [IO.Path]::GetPathRoot($ReadOnlySource).TrimEnd('\')) {
            throw 'ReadOnlySource must be a specific directory on a local drive.'
        }
        if (-not $DryRun -and -not (Test-Path -LiteralPath $ReadOnlySource -PathType Container)) {
            throw "Read-only source directory not found: $ReadOnlySource"
        }
        $sourceMode = 'rclone-read-only'
        $gameRoot = '<unused-drive>:\'
        $gameExe = "${gameRoot}game_scarlett_master.exe"
    } else {
        if (-not $GameDir) { $GameDir = Join-Path $experimentRoot 'validation-title' }
        $gameRoot = Resolve-Rea0032Path $GameDir
        if ($gameRoot.StartsWith('\\') -or $gameRoot.TrimEnd('\') -eq $originalRoot -or $gameRoot.StartsWith(
                "$originalRoot\", [System.StringComparison]::OrdinalIgnoreCase)) {
            throw 'Use an independent local title copy or specify ReadOnlySource to mount the source.'
        }
        $gameExe = Join-Path $gameRoot 'game_scarlett_master.exe'
    }
    $gameHash = 'e95020b2fa4a721db3df59fdecf75621730ac559d8835252f6c3b658d5a08993'

    $commit = $null
    $expectedHash = $null
    $manifestPath = $null
    if ($Profile -eq 'baseline') {
        if (-not $ExePath) {
            $ExePath = Join-Path $env:USERPROFILE 'OneDrive\Desktop\VIctory-Perftry\target\baseline-f11885b\victory.exe'
        }
        $commit = 'f11885b086aa002662da5591c31029accb3b2e98'
        $expectedHash = '1387ec6e3c49acc53697c9bf9a0bccd3f3e83e7dc46a7fdb379472953bbd3735'
    } else {
        if (-not $ExePath) { $ExePath = Join-Path $repoRoot 'target\release\victory.exe' }
        $manifestPath = Join-Path $repoRoot 'research\REA-003.2\RELEASE_MANIFEST.json'
        if (Test-Path -LiteralPath $manifestPath -PathType Leaf) {
            $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
            $commit = [string]$manifest.executable.commit
            $expectedHash = [string]$manifest.executable.sha256
            if ($commit -notmatch '^[a-f0-9]{40}$' -or $expectedHash -notmatch '^[a-f0-9]{64}$') {
                throw "Invalid executable commit or SHA-256 in $manifestPath"
            }
        } elseif (-not $DryRun) {
            throw "Release manifest not found: $manifestPath"
        }
    }
    $resolvedExe = Resolve-Rea0032Path $ExePath
    $exeHash = $null
    if (Test-Path -LiteralPath $resolvedExe -PathType Leaf) {
        $exeHash = (Get-FileHash -LiteralPath $resolvedExe -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($expectedHash -and $exeHash -ne $expectedHash) {
            throw "Executable SHA-256 mismatch: $resolvedExe (expected $expectedHash, got $exeHash)"
        }
    } elseif (-not $DryRun) {
        throw "Runtime executable not found: $resolvedExe"
    }

    if ($CacheSeed) { $CacheSeed = Resolve-Rea0032Path $CacheSeed }
    if ($DataSeed) { $DataSeed = Resolve-Rea0032Path $DataSeed }
    foreach ($seed in @($CacheSeed, $DataSeed)) {
        if ($seed -and -not (Test-Path -LiteralPath $seed -PathType Container)) {
            throw "Seed directory not found: $seed"
        }
    }
    if (-not $DryRun -and -not $ReadOnlySource) { Assert-Rea0032Title $gameRoot $gameExe $gameHash }

    $runId = 'rea0032-{0}-{1}-{2}' -f $Profile, (Get-Date -Format 'yyyyMMdd-HHmmss'),
        ([Guid]::NewGuid().ToString('N').Substring(0, 12))
    $runDir = Join-Path (Join-Path $experimentRoot 'runs') $runId
    $cacheDir = Join-Path $runDir 'cache'
    $dataDir = Join-Path $runDir 'data'
    $stdoutPath = Join-Path $runDir 'stdout.log'
    $stderrPath = Join-Path $runDir 'stderr.log'
    $configPath = Join-Path $runDir 'victory.toml'
    $telemetryPath = Join-Path $runDir 'telemetry.jsonl'
    $hostTelemetryPath = Join-Path $runDir 'host.jsonl'
    $metaPath = Join-Path $runDir 'run.json'
    $cliArgs = @('run', '--console', $Console, '--execute', $gameExe)
    $settings = [ordered]@{
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
        VICTORY_OWN_UPSCALER            = '1'
        VICTORY_CACHE_DIR               = $cacheDir
        VICTORY_DATA_DIR                = $dataDir
        VICTORY_GAME_DIR                = $gameRoot
        VICTORY_CONFIG                  = $configPath
        VICTORY_REA_TELEMETRY_JSON       = $telemetryPath
        VICTORY_PRESET                  = "rea0032-$Profile"
        VICTORY_RUN_ID                  = $runId
        VICTORY_GIT_COMMIT              = $commit
        VICTORY_RUNTIME_SHA256          = $exeHash
        VICTORY_REA_HOST_SCHEDULER       = '1'
        NO_COLOR                        = '1'
    }
    if ($Profile -eq 'baseline') {
        $settings['VICTORY_AMBIENT_STUB'] = '1'
        $settings['VICTORY_SHADOWMASK_STUB'] = '1'
        $settings['VICTORY_REA_INCREMENTAL_ACCOUNTING'] = '1'
        $settings['VICTORY_REA_NATIVE_CPU_FASTPATH'] = '1'
        $settings['VICTORY_REA_DIRTY_RANGES'] = '1'
        $settings['VICTORY_REA_ASYNC_STREAMING'] = '1'
        $settings['VICTORY_REA_GPU_FASTPATH'] = '1'
    } else {
        $settings['VICTORY_REA_SHADER_WORKERS'] = '2'
        $settings['VICTORY_PIPELINE_COMPILE_GUARD'] = '1'
    }
    if ($Profile -eq 'experimental') { $settings['VICTORY_PIPELINE_HISTORY'] = '0' }

    $meta = [ordered]@{
        profile           = $Profile
        run_id            = $runId
        run_dir           = $runDir
        executable        = $resolvedExe
        executable_commit = $commit
        executable_sha256 = $exeHash
        identity_verified = ($null -ne $exeHash -and $exeHash -eq $expectedHash)
        release_manifest  = $manifestPath
        title_source_mode = $sourceMode
        read_only_source  = $ReadOnlySource
        mount_log         = $null
        mount_pid         = $null
        game_executable   = $gameExe
        title_sha256      = $gameHash
        args              = $cliArgs
        effective_env     = $settings
        cache_dir         = $cacheDir
        cache_seed        = $CacheSeed
        data_dir          = $dataDir
        data_seed         = $DataSeed
        stdout_log        = $stdoutPath
        stderr_log        = $stderrPath
        telemetry         = $telemetryPath
        host_telemetry    = $hostTelemetryPath
        metadata          = $metaPath
        budget_seconds    = $BudgetSeconds
        started_at        = $null
        finished_at       = $null
        pid               = $null
        exit_code         = $null
        status            = 'dry-run'
        error             = $null
    }
    if ($DryRun) { return $meta }

    New-Item -ItemType Directory -Path $runDir, $cacheDir, $dataDir -Force | Out-Null
    $meta.status = 'preparing'
    $process = $null
    $mount = $null
    $stdout = $null
    $stderr = $null
    try {
        if ($ReadOnlySource) {
            $meta.mount_log = Join-Path $runDir 'rclone.log'
            $label = 'REA0032-' + $runId.Split('-')[-1]
            $mount = Start-Rea0032ReadOnlyMount $ReadOnlySource $runDir $label
            $meta.mount_pid = $mount.process.Id
            $gameRoot = $mount.root
            $gameExe = Join-Path $gameRoot 'game_scarlett_master.exe'
            $cliArgs[-1] = $gameExe
            $settings['VICTORY_GAME_DIR'] = $gameRoot
            $meta.game_executable = $gameExe
            Assert-Rea0032Title $gameRoot $gameExe $gameHash
        }
        Copy-Rea0032Seed $CacheSeed $cacheDir
        Copy-Rea0032Seed $DataSeed $dataDir
        Set-Content -LiteralPath $configPath -Value '[paths]' -Encoding Ascii
        try {
            Initialize-Rea0032HostMonitor
        } catch {
            Write-Warning "Host memory counters unavailable: $($_.Exception.Message)"
        }
        $start = [System.Diagnostics.ProcessStartInfo]::new($resolvedExe)
        $start.WorkingDirectory = $runDir
        $start.UseShellExecute = $false
        $start.RedirectStandardOutput = $true
        $start.RedirectStandardError = $true
        # The only variable native argument is a Windows file path, which cannot
        # contain a quote and ends in .exe. No command shell sees these arguments.
        $start.Arguments = ($cliArgs | ForEach-Object { '"' + $_ + '"' }) -join ' '
        foreach ($name in @($start.EnvironmentVariables.Keys)) {
            if ($name -like 'VICTORY_*' -or $name -in @('NO_COLOR', 'RUST_LOG')) {
                $start.EnvironmentVariables.Remove($name)
            }
        }
        foreach ($entry in $settings.GetEnumerator()) {
            $start.EnvironmentVariables[$entry.Key] = [string]$entry.Value
        }

        $stdout = [System.IO.File]::Open($stdoutPath, 'CreateNew', 'Write', 'Read')
        $stderr = [System.IO.File]::Open($stderrPath, 'CreateNew', 'Write', 'Read')
        $null = New-Item -ItemType File -Path $hostTelemetryPath
        $process = [System.Diagnostics.Process]::Start($start)
        $clock = [System.Diagnostics.Stopwatch]::StartNew()
        $meta.pid = $process.Id
        $meta.started_at = [DateTimeOffset]::Now.ToString('o')
        $meta.status = 'running'
        $outCopy = $process.StandardOutput.BaseStream.CopyToAsync($stdout)
        $errCopy = $process.StandardError.BaseStream.CopyToAsync($stderr)
        $meta | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $metaPath -Encoding UTF8
        Write-Verbose "Running $Profile (PID $($meta.pid)). Logs: $runDir"
        $previousSampleMs = 0L
        $previousCpuMs = 0.0
        $budgetMs = [long]$BudgetSeconds * 1000
        $meta.status = 'exited'
        while (-not $process.HasExited) {
            $intervalRemaining = 2000 - ($clock.ElapsedMilliseconds - $previousSampleMs)
            if ($budgetMs -gt 0) {
                $remaining = $budgetMs - $clock.ElapsedMilliseconds
                if ($remaining -le 0) { break }
                $intervalRemaining = [Math]::Min($intervalRemaining, $remaining)
            }
            if ($intervalRemaining -gt 0 -and $process.WaitForExit([int]$intervalRemaining)) { break }
            $elapsedMs = $clock.ElapsedMilliseconds
            if ($elapsedMs - $previousSampleMs -ge 2000) {
                $previousCpuMs = Write-Rea0032HostSample $process $elapsedMs $previousSampleMs $previousCpuMs $hostTelemetryPath
                $previousSampleMs = $elapsedMs
            }
        }
        if (-not $process.HasExited) {
            if ($process.CloseMainWindow() -and $process.WaitForExit(10000)) {
                $meta.status = 'budget-closed'
            } else {
                Stop-Rea0032OwnedProcess $process
                $meta.status = 'budget-forced'
            }
        }
        $process.WaitForExit()
        $meta.exit_code = $process.ExitCode
        [void]$outCopy.GetAwaiter().GetResult()
        [void]$errCopy.GetAwaiter().GetResult()
    } catch {
        $meta.status = 'error'
        $meta.error = $_.Exception.Message
        throw
    } finally {
        if ($null -ne $process) {
            Stop-Rea0032OwnedProcess $process
            $process.Dispose()
        }
        if ($null -ne $mount) {
            Stop-Rea0032OwnedProcess $mount.process
            $mount.process.Dispose()
        }
        if ($null -ne $stdout) { $stdout.Dispose() }
        if ($null -ne $stderr) { $stderr.Dispose() }
        $meta.finished_at = [DateTimeOffset]::Now.ToString('o')
        $meta | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $metaPath -Encoding UTF8
    }
    return $meta
}
