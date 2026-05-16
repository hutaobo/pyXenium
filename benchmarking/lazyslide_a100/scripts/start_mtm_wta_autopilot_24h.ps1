$ErrorActionPreference = 'Stop'

$repo = Resolve-Path (Join-Path $PSScriptRoot '..\..\..')
$script = Join-Path $repo 'benchmarking\lazyslide_a100\scripts\autopilot_mtm_wta_24h.py'
$stateDir = Join-Path $repo 'docs\_static\tutorials\multimodal_histoseg_lazyslide_breast_wta\naturebiotech_package\autopilot_20260511'
New-Item -ItemType Directory -Force -Path $stateDir | Out-Null

$stdout = Join-Path $stateDir 'local_supervisor_stdout.log'
$stderr = Join-Path $stateDir 'local_supervisor_stderr.log'
$python = (Get-Command python).Source

$process = Start-Process `
  -FilePath $python `
  -ArgumentList @('-u', $script, '--hours', '24', '--interval-minutes', '10', '--gpu', '7', '--force-uni') `
  -WorkingDirectory $repo `
  -RedirectStandardOutput $stdout `
  -RedirectStandardError $stderr `
  -WindowStyle Hidden `
  -PassThru

Set-Content -Path (Join-Path $stateDir 'local_supervisor.pid') -Value $process.Id
Write-Output $process.Id
