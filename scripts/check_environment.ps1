<#
.SYNOPSIS
    SPECTRA Phase 6 Environment, Dependency, Model, Hardware & ComfyUI Non-Interference Pre-Flight Check.
#>
param(
    [switch]$Quiet
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$BackendDir = Join-Path $ProjectRoot "backend"
$FrontendDir = Join-Path $ProjectRoot "frontend"
$ModelsDir = Join-Path $ProjectRoot "models"
$StorageDir = Join-Path $ProjectRoot "storage"

function Write-Step([string]$Msg, [string]$Color = "Cyan") {
    if (-not $Quiet) {
        Write-Host "[SPECTRA CHECK] $Msg" -ForegroundColor $Color
    }
}

Write-Step "Validating SPECTRA Release Candidate Workspace at: $ProjectRoot"

# 1. Check Disk Space (require >= 1.0 GB free)
$DriveLetter = (Get-Item $ProjectRoot).PSDrive.Name
$DriveInfo = Get-PSDrive -Name $DriveLetter
$FreeGB = [math]::Round($DriveInfo.Free / 1GB, 2)
if ($FreeGB -lt 1.0) {
    Write-Host "[ERROR] Insufficient disk space on drive ${DriveLetter}: ($FreeGB GB free; minimum 1.0 GB required)." -ForegroundColor Red
    exit 1
}
Write-Step "Disk space on ${DriveLetter}: $FreeGB GB free (OK)" "Green"

# 2. Check Python Environment & Required Packages
$PythonCmd = $null
$VenvPython = Join-Path $BackendDir ".venv\Scripts\python.exe"
$RootVenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $PythonCmd = $VenvPython
} elseif (Test-Path $RootVenvPython) {
    $PythonCmd = $RootVenvPython
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $PythonCmd = (Get-Command python).Source
} else {
    Write-Host "[ERROR] Python 3.10+ interpreter not found. Please install Python or activate a virtual environment." -ForegroundColor Red
    exit 1
}
Write-Step "Python interpreter: $PythonCmd"

$PyCheckScript = @"
import importlib, json, shutil, sys
required = ['fastapi', 'uvicorn', 'numpy', 'scipy', 'librosa', 'soundfile', 'soxr', 'onnxruntime', 'torch', 'c2pa', 'reportlab', 'av']
missing = []
for pkg in required:
    try:
        importlib.import_module(pkg)
    except Exception as e:
        missing.append(f'{pkg} ({e})')

import torch
cuda_ok = torch.cuda.is_available()
gpu_name = torch.cuda.get_device_name(0) if cuda_ok else 'None (CPU Mode)'
vram_free_mb = 0
vram_total_mb = 0
if cuda_ok:
    free_b, total_b = torch.cuda.mem_get_info(0)
    vram_free_mb = round(free_b / (1024*1024), 1)
    vram_total_mb = round(total_b / (1024*1024), 1)

ffmpeg_bin = shutil.which('ffmpeg')
import av
pyav_ver = getattr(av, '__version__', 'installed')

print(json.dumps({
    'python_version': sys.version.split()[0],
    'missing_packages': missing,
    'cuda_available': cuda_ok,
    'gpu_name': gpu_name,
    'vram_free_mb': vram_free_mb,
    'vram_total_mb': vram_total_mb,
    'system_ffmpeg': ffmpeg_bin,
    'pyav_ffmpeg_version': pyav_ver,
}))
"@

$PyCheckJson = & $PythonCmd -c $PyCheckScript
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to execute Python dependency probe." -ForegroundColor Red
    exit 1
}
$PyInfo = $PyCheckJson | ConvertFrom-Json
if ($PyInfo.missing_packages.Count -gt 0) {
    $MissingList = $PyInfo.missing_packages -join ", "
    Write-Host "[ERROR] Missing required Python packages: $MissingList" -ForegroundColor Red
    Write-Host "        Run: pip install -r backend/requirements.txt" -ForegroundColor Yellow
    exit 1
}
Write-Step "Python $($PyInfo.python_version) dependencies verified (FastAPI, PyTorch, ONNXRuntime, Librosa, C2PA, ReportLab, PyAV)" "Green"

# 3. Check FFmpeg / PyAV Codec Backend
if ($PyInfo.system_ffmpeg) {
    Write-Step "FFmpeg detected: System FFmpeg ($($PyInfo.system_ffmpeg)) + PyAV $($PyInfo.pyav_ffmpeg_version)" "Green"
} else {
    Write-Step "FFmpeg detected via bundled PyAV $($PyInfo.pyav_ffmpeg_version) LGPL codec libraries" "Green"
}

# 4. Validate Model Weights & Checkpoints
$RequiredModels = @(
    @{ Rel = "fakeprint\ai_music_detector.onnx"; MinBytes = 10000; License = "Apache-2.0 (lofcz/ai-music-detector)" },
    @{ Rel = "sonics\sonics-spectttra-gamma-5s\pytorch_model.bin"; MinBytes = 10000000; License = "MIT Code / CC BY-NC 4.0 Dataset-Derived Weights (Research/Academic Use)" },
    @{ Rel = "sonics\sonics-spectttra-gamma-5s\config.json"; MinBytes = 100; License = "MIT" }
)
foreach ($m in $RequiredModels) {
    $FullPath = Join-Path $ModelsDir $m.Rel
    if (-not (Test-Path $FullPath)) {
        Write-Host "[ERROR] Required model file missing: $FullPath" -ForegroundColor Red
        exit 1
    }
    $Size = (Get-Item $FullPath).Length
    if ($Size -lt $m.MinBytes) {
        Write-Host "[ERROR] Model file appears incomplete ($Size bytes): $FullPath" -ForegroundColor Red
        exit 1
    }
    Write-Step "Model verified: models\$($m.Rel) ($([math]::Round($Size / 1MB, 2)) MB) [$($m.License)]" "Green"
}

# 5. Check CPU / GPU Resources & ComfyUI Non-Interference
Write-Step "Hardware Accelerator: $($PyInfo.gpu_name) (Free VRAM: $($PyInfo.vram_free_mb) MB / $($PyInfo.vram_total_mb) MB)"
if (Test-Path "D:\ComfyUI") {
    Write-Step "Existing ComfyUI installation at D:\ComfyUI detected (Strictly Read-Only / Untouched)." "Green"
}
$ComfyPortActive = Get-NetTCPConnection -LocalPort 8188 -State Listen -ErrorAction SilentlyContinue
if ($ComfyPortActive) {
    Write-Step "Notice: ComfyUI server is actively listening on port 8188. SPECTRA defaults to CPU inference (SPECTRA_DEFAULT_DEVICE=cpu) to guarantee zero GPU VRAM contention." "Yellow"
}

# 6. Check Frontend Node.js & Build Artifacts
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Node.js not found on PATH. Please install Node.js 18+." -ForegroundColor Red
    exit 1
}
$NodeModules = Join-Path $FrontendDir "node_modules"
if (-not (Test-Path $NodeModules)) {
    Write-Host "[ERROR] Frontend node_modules missing at $NodeModules. Run 'cd frontend && npm install'." -ForegroundColor Red
    exit 1
}
Write-Step "Node.js & frontend dependencies verified." "Green"

# 7. Ensure Storage Directories Exist (Preserving Existing Experiments & Database)
foreach ($sub in @("originals", "processed", "reports", "samples", "temp", "temp\temporal_cache")) {
    $p = Join-Path $StorageDir $sub
    if (-not (Test-Path $p)) {
        New-Item -ItemType Directory -Path $p -Force | Out-Null
    }
}
Write-Step "Storage hierarchy verified at $StorageDir (Existing experiments and spectra.db preserved)." "Green"
exit 0
