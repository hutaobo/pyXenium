param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot

$env:PYTHONPATH = "$RepoRoot\src;$env:PYTHONPATH"

& $Python "benchmarking\lazyslide_a100\scripts\compose_nbt_brief_main_figure.py"
& $Python "benchmarking\lazyslide_a100\scripts\enhance_mtm_nature_assets.py"
& $Python "benchmarking\lazyslide_a100\scripts\prepare_nbt_initial_submission_upload.py"

Write-Host "Local manuscript package rebuild completed."
Write-Host "For full GPU feature regeneration, use run_full_replica.sh on a Linux/A100 host with raw input paths set."
