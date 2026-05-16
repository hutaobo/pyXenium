$ErrorActionPreference = "Stop"

$repo = "D:\GitHub\pyXenium"
$script = Join-Path $repo "benchmarking\lazyslide_a100\scripts\autopilot_mtm_wta_defense_24h.py"
$stateDir = Join-Path $repo "docs\_static\tutorials\multimodal_histoseg_lazyslide_breast_wta\naturebiotech_package\autopilot_20260512_defense"
New-Item -ItemType Directory -Force -Path $stateDir | Out-Null

$stdout = Join-Path $stateDir "local_supervisor_stdout.log"
$stderr = Join-Path $stateDir "local_supervisor_stderr.log"
$pidFile = Join-Path $stateDir "local_supervisor.pid"

$python = (Get-Command python).Source
$args = @(
  $script,
  "--repo", $repo,
  "--hours", "24",
  "--interval-minutes", "10",
  "--permutations", "10000",
  "--bootstraps", "2000"
)

$process = Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $repo -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -WindowStyle Hidden
Set-Content -Path $pidFile -Value "$($process.Id)"
Write-Host "Started mTM WTA defense autopilot PID $($process.Id)"
