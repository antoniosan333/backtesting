# Fast-forward this PC to GitHub. Refuses to discard local work.
param(
    [string]$Branch = ""
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))

git fetch origin
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($Branch -ne "") {
    git checkout $Branch
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    git pull --ff-only origin $Branch
    exit $LASTEXITCODE
}

git pull --ff-only
exit $LASTEXITCODE
