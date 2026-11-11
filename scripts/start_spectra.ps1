<#
.SYNOPSIS
    SPECTRA Phase 6 Release Candidate Startup Script for Windows.
    Runs pre-flight checks, launches FastAPI backend (port 8000) and Next.js frontend (port 3000),
    waits for health checks, and records PIDs for graceful shutdown without touching D:\ComfyUI.
#>
param(
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 3000,
    [switch]$DevMode,
    [switch]$KeepAlive
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$BackendDir = Join-Path $ProjectRoot "backend"
$FrontendDir = Join-Path $ProjectRoot "frontend"
$PidFile = Join-Path $ProjectRoot ".spectra_pids.json"
$CheckScript = Join-Path $PSScriptRoot "check_environment.ps1"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  SPECTRA | AI Audio Forensics & Fingerprint Analysis Laboratory" -ForegroundColor Cyan
Write-Host "  Phase 6 Release Candidate Launcher (Windows)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

& $CheckScript
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ABORT] Pre-flight environment check failed." -ForegroundColor Red
    exit 1
}

# Check if ports 8000 or 3000 are already occupied
foreach ($port in @($BackendPort, $FrontendPort)) {
    $existing = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "[WARNING] Port $port is already in use (PID: $($existing[0].OwningProcess)). Run scripts\stop_spectra.ps1 first." -ForegroundColor Yellow
        exit 1
    }
}

$Env:SPECTRA_DEFAULT_DEVICE = "cpu"
$BackendLog = Join-Path $ProjectRoot "storage\temp\backend_server.log"
$BackendErr = Join-Path $ProjectRoot "storage\temp\backend_server.err.log"
$FrontendLog = Join-Path $ProjectRoot "storage\temp\frontend_server.log"
$FrontendErr = Join-Path $ProjectRoot "storage\temp\frontend_server.err.log"

Write-Host "[1/3] Starting FastAPI Backend on http://127.0.0.1:$BackendPort ..." -ForegroundColor Cyan
$BackendProc = Start-Process -FilePath "python" `
    -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "$BackendPort" `
    -WorkingDirectory $BackendDir `
    -RedirectStandardOutput $BackendLog `
    -RedirectStandardError $BackendErr `
    -PassThru -WindowStyle Hidden

$NextBuildId = Join-Path $FrontendDir ".next\BUILD_ID"
if ((-not $DevMode) -and (Test-Path $NextBuildId)) {
    Write-Host "[2/3] Starting Next.js Production Frontend on http://127.0.0.1:$FrontendPort ..." -ForegroundColor Cyan
    $FrontendProc = Start-Process -FilePath "cmd.exe" `
        -ArgumentList "/c", "npm run start -- -p $FrontendPort" `
        -WorkingDirectory $FrontendDir `
        -RedirectStandardOutput $FrontendLog `
        -RedirectStandardError $FrontendErr `
        -PassThru -WindowStyle Hidden
} else {
    Write-Host "[2/3] Starting Next.js Development Frontend on http://127.0.0.1:$FrontendPort ..." -ForegroundColor Cyan
    $FrontendProc = Start-Process -FilePath "cmd.exe" `
        -ArgumentList "/c", "npm run dev -- -p $FrontendPort" `
        -WorkingDirectory $FrontendDir `
        -RedirectStandardOutput $FrontendLog `
        -RedirectStandardError $FrontendErr `
        -PassThru -WindowStyle Hidden
}

$PidData = @{
    backend_pid = $BackendProc.Id
    frontend_pid = $FrontendProc.Id
    backend_port = $BackendPort
    frontend_port = $FrontendPort
    started_at_utc = (Get-Date).ToUniversalTime().ToString("o")
}
$PidData | ConvertTo-Json | Set-Content -Path $PidFile -Encoding UTF8

Write-Host "[3/3] Waiting for backend (/api/health) and frontend readiness..." -ForegroundColor Cyan
$BackendReady = $false
$FrontendReady = $false

for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1
    if (-not $BackendReady) {
        try {
            $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$BackendPort/api/health" -TimeoutSec 2 -ErrorAction Stop
            if ($resp.status -eq "healthy") {
                $BackendReady = $true
            }
        } catch {}
    }
    if (-not $FrontendReady) {
        try {
            $fResp = Invoke-WebRequest -Uri "http://127.0.0.1:$FrontendPort" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
            if ($fResp.StatusCode -eq 200) {
                $FrontendReady = $true
            }
        } catch {}
    }
    if ($BackendReady -and $FrontendReady) {
        break
    }
}

if (-not ($BackendReady -and $FrontendReady)) {
    Write-Host "[ERROR] Services did not become ready within 30 seconds (Backend=$BackendReady, Frontend=$FrontendReady)." -ForegroundColor Red
    & (Join-Path $PSScriptRoot "stop_spectra.ps1")
    exit 1
}

Write-Host "======================================================================" -ForegroundColor Green
Write-Host "  SPECTRA Workstation is LIVE:" -ForegroundColor Green
Write-Host "  • Web Application : http://localhost:$FrontendPort" -ForegroundColor Green
Write-Host "  • FastAPI Backend : http://127.0.0.1:$BackendPort/api/health" -ForegroundColor Green
Write-Host "  • Backend PID     : $($BackendProc.Id) | Frontend PID: $($FrontendProc.Id)" -ForegroundColor Green
Write-Host "  • Stop Cleanly    : powershell -ExecutionPolicy Bypass -File scripts\stop_spectra.ps1" -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Green

if ($KeepAlive) {
    Write-Host "KeepAlive mode active. Press Ctrl+C or run stop_spectra.ps1 to stop." -ForegroundColor Cyan
    while ((-not $BackendProc.HasExited) -and (-not $FrontendProc.HasExited)) {
        Start-Sleep -Seconds 2
    }
}
exit 0
