<#
.SYNOPSIS
    SPECTRA Phase 6 Graceful Shutdown Script for Windows.
    Stops only SPECTRA backend (port 8000) and frontend (port 3000) processes while
    preserving storage/, spectra.db, and never touching D:\ComfyUI or port 8188.
#>
param(
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 3000
)

$ErrorActionPreference = "Continue"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PidFile = Join-Path $ProjectRoot ".spectra_pids.json"

function Stop-ProcessTree([int]$ParentPid) {
    if ($ParentPid -le 0) { return }
    $children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $ParentPid" -ErrorAction SilentlyContinue
    foreach ($child in $children) {
        Stop-ProcessTree -ParentPid $child.ProcessId
    }
    Stop-Process -Id $ParentPid -Force -ErrorAction SilentlyContinue
}

if (Test-Path $PidFile) {
    try {
        $PidInfo = Get-Content $PidFile -Raw | ConvertFrom-Json
        if ($PidInfo.backend_pid) {
            Write-Host "[SPECTRA STOP] Stopping Backend process tree (PID: $($PidInfo.backend_pid))..." -ForegroundColor Cyan
            Stop-ProcessTree -ParentPid ([int]$PidInfo.backend_pid)
        }
        if ($PidInfo.frontend_pid) {
            Write-Host "[SPECTRA STOP] Stopping Frontend process tree (PID: $($PidInfo.frontend_pid))..." -ForegroundColor Cyan
            Stop-ProcessTree -ParentPid ([int]$PidInfo.frontend_pid)
        }
    } catch {}
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
}

# Also clean up any remaining listener on BackendPort (8000) or FrontendPort (3000)
foreach ($port in @($BackendPort, $FrontendPort)) {
    $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($c in $conns) {
        if ($c.OwningProcess -gt 4) {
            Write-Host "[SPECTRA STOP] Releasing port $port (PID: $($c.OwningProcess))..." -ForegroundColor Cyan
            Stop-ProcessTree -ParentPid ([int]$c.OwningProcess)
        }
    }
}

Write-Host "[SPECTRA STOP] Shutdown complete. Saved experiments in storage/ and spectra.db are preserved." -ForegroundColor Green
exit 0
