<#
.SYNOPSIS
    Runs the stable REA-003.2 profile with an isolated cache and data directory.
.PARAMETER GameDir
    Independent local title copy. Defaults to the REA-003.2 validation-title directory.
.PARAMETER ReadOnlySource
    Local title directory mounted through rclone and WinFsp on a temporary drive.
    This mode prevents writes to the source through the mount. Its I/O timings differ
    from native filesystem runs. Requires installed rclone and WinFsp.
.PARAMETER DryRun
    Shows the resolved profile and intended source without mounting, copying, or launching.
#>
[CmdletBinding(DefaultParameterSetName = 'Copy')]
param(
    [Parameter(ParameterSetName = 'Copy')][string]$GameDir,
    [Parameter(Mandatory, ParameterSetName = 'ReadOnly')][string]$ReadOnlySource,
    [string]$ExePath,
    [string]$CacheSeed,
    [string]$DataSeed,
    [ValidateSet('series-s', 'series-x')][string]$Console,
    [string]$RenderScale,
    [ValidateRange(0, 86400)][int]$BudgetSeconds,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'rea0032-common.ps1')
$result = Invoke-Rea0032Run -Profile 'stable' @PSBoundParameters
$result | ConvertTo-Json -Depth 6
if ($null -ne $result.exit_code) { exit $result.exit_code }
