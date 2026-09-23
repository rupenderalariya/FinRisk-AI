<#
    FinRisk AI - dashboard launcher.

        .\run_dashboard.ps1

    Why this exists
    ---------------
    `streamlit run app.py` only works when the virtual environment is active.
    Without activation PowerShell reports:

        streamlit : The term 'streamlit' is not recognized ...

    because `streamlit.exe` lives in .venv\Scripts and is not on the system PATH.
    This script resolves the interpreter inside .venv explicitly, so the dashboard
    starts from a clean shell with no activation step and no PATH changes.

    Parameters
    ----------
    -Port      Port to serve on. Default 8501.
    -Headless  Do not open a browser tab. Use for CI and health probes.
#>

[CmdletBinding()]
param(
    [int]$Port = 8501,
    [switch]$Headless
)

$ErrorActionPreference = 'Stop'

$projectRoot = $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$entryPoint = Join-Path $projectRoot 'app.py'

if (-not (Test-Path $python)) {
    Write-Error @"
Virtual environment not found at:
    $python

Create it and install the pinned dependencies first:
    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
"@
    exit 1
}

if (-not (Test-Path $entryPoint)) {
    Write-Error "Entry point not found at: $entryPoint"
    exit 1
}

# Fail early and clearly if the dataset is missing, rather than letting the UI
# render its "unable to load" notice with no explanation in the terminal.
$rawDir = Join-Path $projectRoot 'data\raw'
if (-not (Test-Path $rawDir) -or -not (Get-ChildItem $rawDir -File -ErrorAction SilentlyContinue)) {
    Write-Warning @"
No files found in data\raw. The dashboard will start but cannot load the dataset.
Fetch it first:
    .\.venv\Scripts\python.exe scripts\download_data.py
"@
}

# Streamlit asks for an email on its very first interactive run and waits on stdin
# for the answer. That prompt is suppressed in headless mode, so it never appeared
# while `server.headless = true` was set - and it is easy to mistake a blocked
# prompt for an application that will not start. Seeding an empty credentials file
# declines the prompt once, without changing any other Streamlit behaviour.
$credentials = Join-Path $env:USERPROFILE '.streamlit\credentials.toml'
if (-not (Test-Path $credentials)) {
    Write-Host '  First run: declining the Streamlit email prompt.' -ForegroundColor DarkGray
    New-Item -ItemType Directory -Force -Path (Split-Path $credentials) | Out-Null
    Set-Content -Path $credentials -Value @('[general]', 'email = ""') -Encoding ascii
}

$arguments = @('-m', 'streamlit', 'run', $entryPoint, '--server.port', $Port)
if ($Headless) {
    $arguments += @('--server.headless', 'true')
}

Write-Host ''
Write-Host '  FinRisk AI - Credit Risk Intelligence' -ForegroundColor Cyan
Write-Host "  Starting dashboard on http://localhost:$Port" -ForegroundColor DarkGray
Write-Host '  Press Ctrl+C to stop.' -ForegroundColor DarkGray
Write-Host ''

& $python @arguments
exit $LASTEXITCODE
