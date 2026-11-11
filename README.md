# SPECTRA | AI Audio Forensics & Fingerprint Analysis Laboratory

**SPECTRA** is a local-first AI audio forensics and signal-processing workstation that inspects cryptographic provenance (C2PA & container metadata), neural-vocoder spectral fingerprints (`1–8 kHz` fakeprints), and time-localized AI music signatures (`SONICS SpecTTTra`). It pairs multi-detector forensic attribution with an interactive DSP laboratory and A/B comparison engine to evaluate how acoustic transformations affect synthetic music artifacts.

---

## 1. Five-Tier Forensic Attribution Hierarchy

SPECTRA enforces strict scientific separation across five distinct mechanisms of audio analysis:

| Tier | Mechanism | Implementation | Forensic Scope & Epistemic Boundary |
| :---: | :--- | :--- | :--- |
| **1** | **Cryptographic Provenance (C2PA)** | `c2pa-python` (`c2pa-rs 0.91.0`) | Verifies JUMBF manifests, claim generators, byte-range asset hashes, and X.509 signing chains. Stripped by standard WAV/MP3 re-encoding; manifest absence is neutral and never implies human or AI authorship. |
| **2** | **Conventional Container Metadata** | `Mutagen` (`ID3v2`, `Vorbis`, `RIFF`, `MP4`) | Inspects mutable container tags independently of C2PA and supports authorized metadata-clean export without modifying PCM audio samples. |
| **3** | **Digital Audio Watermarks** | Architectural slot | Intentionally embedded psychoacoustic or neural bitstreams (distinct from passive synthesis artifacts). |
| **4** | **Physical Spectral Artifacts ("Fakeprints")** | `8192`-point STFT @ `16 kHz` (`1–8 kHz` lower-hull residual) | Measures periodic deconvolution/vocoder comb peaks (`200/400 Hz` grid spacing) above the lower spectral hull (`minimum_filter1d`). |
| **5** | **Learned AI Classifiers & Sliding-Window Forensics** | `Fakeprint ONNX LR` + `SONICS SpecTTTra-γ (5s)` + `SONICS SpecTTTra-α (120s)` | Evaluates full-track and sliding-window (`5.0s` window, `1.0s`/`2.5s`/`5.0s` hops) spectro-temporal representations with Out-of-Distribution (OOD) abstention and temperature-calibrated uncertainty. |

> **Scientific Integrity Principle:** No single detector's output is presented as definitive proof of content origin. Absence of a C2PA manifest does not prove human or AI authorship, and experimental DSP artifact reduction measures model sensitivity rather than universal removal of generative provenance.

---

## 2. Project Directory Structure

```text
Spectra/
├── backend/
│   ├── app/
│   │   ├── api/routes.py               # FastAPI REST endpoints
│   │   ├── core/config.py              # Pydantic settings & environment configuration
│   │   ├── database/                   # SQLAlchemy 2.0 ORM & SQLite WAL session
│   │   ├── detectors/                  # Fakeprint ONNX LR & SONICS SpecTTTra-γ/α adapters
│   │   ├── integrations/               # Read-only ComfyUI Desktop ("Nyx") status adapter
│   │   ├── processing/                 # 14 SciPy/Pedalboard DSP operations & A/B Comparison Engine
│   │   ├── schemas/                    # Typed Pydantic v2 request/response schemas
│   │   ├── services/                   # Ingestion, Provenance, Forensics, Temporal Engine, Reports
│   │   ├── workers/                    # Job runner with progress tracking & cancellation
│   │   └── main.py                     # FastAPI application entrypoint
│   ├── tests/
│   │   └── test_spectra_backend.py     # 46-test automated backend & E2E verification suite
│   └── requirements.txt
├── frontend/
│   ├── app/                            # Next.js 15 App Router (TypeScript + Tailwind CSS)
│   ├── components/                     # Workstation Player, Spectrogram Canvas, Temporal Timeline, DSP Lab
│   ├── e2e/spectra.spec.ts             # 6-test Playwright browser verification suite
│   ├── lib/api.ts                      # Typed REST client
│   └── types/spectra.ts                # Shared TypeScript interfaces
├── models/
│   ├── fakeprint/                      # lofcz/ai-music-detector ONNX model & YAML configs
│   └── sonics/                         # awsaf49/sonics-spectttra-gamma-5s & alpha-120s configs/weights
├── research/
│   └── benchmarks/                     # Reproducible cohort evaluation & temporal localization scripts
├── scripts/
│   ├── check_environment.ps1           # Windows pre-flight environment & dependency validator
│   ├── start_spectra.ps1               # Windows startup launcher with health-check polling
│   └── stop_spectra.ps1                # Graceful shutdown script preserving storage/ & spectra.db
├── .env.example
├── start_spectra.bat
├── stop_spectra.bat
└── README.md
```

---

## 3. Integrated Models & Weight Setup

| Detector ID | Model Name | Upstream Source | Input Requirements | License |
| :--- | :--- | :--- | :--- | :--- |
| `fakeprint_onnx` | **Neural-Vocoder Fakeprint Detector (ONNX LR)** | [`lofcz/ai-music-detector`](https://github.com/lofcz/ai-music-detector) (Afchar et al., ISMIR 2025) | `16,000 Hz` mono, `8192`-pt STFT, `3,585` bins (`1–8 kHz`) | Apache-2.0 / MIT |
| `sonics_gamma_5s` | **SONICS SpecTTTra-γ (5s)** | [`awsaf49/sonics-spectttra-gamma-5s`](https://huggingface.co/awsaf49/sonics-spectttra-gamma-5s) | `16,000 Hz` mono, `5.0s` (`80,000` samples), unit-variance normalized | MIT (Code) / CC BY-NC 4.0 (Weights) |
| `sonics_alpha_120s` | **SONICS SpecTTTra-α (120s)** | [`awsaf49/sonics-spectttra-alpha-120s`](https://huggingface.co/awsaf49/sonics-spectttra-alpha-120s) | `16,000 Hz` mono, `120.0s` (`>= 115s` non-silent duration gate) | MIT (Code) / CC BY-NC 4.0 (Weights) |

### Downloading Pretrained SONICS Weights (If Cloning Fresh)
The lightweight `Fakeprint ONNX` model (`models/fakeprint/ai_music_detector.onnx`) and all model configurations are included in the repository. To download the `SONICS` PyTorch checkpoints (`pytorch_model.bin`) into `models/sonics/`:

```powershell
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('awsaf49/sonics-spectttra-gamma-5s', 'pytorch_model.bin', local_dir='models/sonics/sonics-spectttra-gamma-5s'); hf_hub_download('awsaf49/sonics-spectttra-alpha-120s', 'pytorch_model.bin', local_dir='models/sonics/sonics-spectttra-alpha-120s')"
```

---

## 4. Quick Start & Local Execution (Windows)

### Prerequisites
- **Python 3.11+** (`pip install -r backend/requirements.txt`)
- **Node.js 18+** (`cd frontend && npm install && npm run build`)

### Option A: One-Click Windows Scripts
```powershell
# Verify environment, dependencies, and model weights
powershell -ExecutionPolicy Bypass -File scripts/check_environment.ps1

# Launch FastAPI (port 8000) and Next.js (port 3000)
powershell -ExecutionPolicy Bypass -File scripts/start_spectra.ps1 -KeepAlive

# Stop servers cleanly while preserving storage/ and spectra.db
powershell -ExecutionPolicy Bypass -File scripts/stop_spectra.ps1
```
*(You can also double-click `start_spectra.bat` and `stop_spectra.bat`.)*

### Option B: Manual Terminal Launch
```powershell
# Terminal 1 — FastAPI Backend (http://127.0.0.1:8000)
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# Terminal 2 — Next.js Frontend (http://localhost:3000)
cd frontend
npm run start -- -p 3000
```
- **Web Workstation**: `http://localhost:3000`
- **Backend Health**: `http://127.0.0.1:8000/api/health`
- **OpenAPI / Swagger UI**: `http://127.0.0.1:8000/docs`

---

## 5. Key REST API Endpoints

| Method & Endpoint | Description |
| :--- | :--- |
| `GET /api/health` | System health, C2PA SDK version, PyAV/FFmpeg status, and ready detector count |
| `GET /api/detectors` | Metadata, revisions, licenses, and readiness status for all registered detectors |
| `POST /api/audio/upload` | Ingests `.wav`, `.flac`, `.mp3`, `.m4a`/`.aac` with magic-byte validation & SHA-256 hashing |
| `GET /api/jobs/{job_id}/results` | Full forensic payload: C2PA provenance, Mutagen tags, detectors, spectral & time metrics |
| `GET / POST /api/jobs/{job_id}/temporal` | Sliding-window temporal AI analysis (`1.0s`, `2.5s`, `5.0s` hops) with 7-D SHA-256 caching |
| `POST /api/jobs/{job_id}/temporal/cancel` | Requests immediate mid-inference cancellation with atomic cache cleanup |
| `POST /api/jobs/{job_id}/process` | Runs any of 14 SciPy/Pedalboard DSP transformations and computes A/B comparison metrics |
| `POST /api/jobs/{job_id}/metadata-export` | Creates an authorized metadata-sanitized export (strips tags and/or C2PA chunks) |
| `GET /api/jobs/{job_id}/report?format=json\|pdf` | Generates auditable JSON or multi-page PDF forensic report |

---

## 6. Automated Testing & Benchmarks

```powershell
# 1. Run 46-test Backend Pytest Suite (Ingestion, C2PA, Spectral, Detectors, OOD, Temporal, DSP, Reports)
cd backend
python -m pytest tests/test_spectra_backend.py -v

# 2. Run 6-test Playwright End-to-End Browser Verification Suite
cd ../frontend
npx playwright test

# 3. Run Reproducible Cohort & Ground-Truth Temporal Localization Benchmarks
cd ..
python research/benchmarks/evaluate_cohort.py
python research/benchmarks/phase6_validation_and_localization.py
```
