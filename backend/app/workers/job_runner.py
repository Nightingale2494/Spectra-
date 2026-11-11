"""Persistent job execution and bounded worker concurrency for SPECTRA."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import importlib.metadata
import json
import platform
import threading
from typing import Dict, Set

import numpy as np
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.models import AnalysisJob, AudioAsset, DetectorRun, ProvenanceResult
from app.database.session import SessionLocal
from app.detectors.registry import detector_registry
from app.services.forensics_service import run_complete_forensics
from app.services.ingestion_service import resolve_asset_path
from app.services.provenance_service import run_provenance_analysis
from app.services.temporal_service import (
    compute_temporal_forensic_analysis,
    request_temporal_job_cancel,
)


_executor = ThreadPoolExecutor(max_workers=max(1, settings.SPECTRA_MAX_WORKERS), thread_name_prefix="spectra-worker")
_cancelled_jobs: Set[str] = set()
_lock = threading.Lock()


def cancel_job(job_id: str) -> None:
    with _lock:
        _cancelled_jobs.add(job_id)
    request_temporal_job_cancel(job_id)


def is_job_cancelled(job_id: str) -> bool:
    with _lock:
        return job_id in _cancelled_jobs


def get_software_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {"python": platform.python_version()}
    for pkg in (
        "fastapi",
        "numpy",
        "scipy",
        "librosa",
        "soundfile",
        "soxr",
        "av",
        "onnxruntime",
        "c2pa-python",
        "mutagen",
        "torch",
        "torchaudio",
        "pedalboard",
        "sonics",
    ):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            versions[pkg] = "not_installed"
    return versions


def _update_job_stage(db: Session, job: AnalysisJob, status: str, progress: float, stage: str) -> None:
    job.status = status
    job.progress = progress
    job.current_stage = stage
    db.commit()


def execute_analysis_job(job_id: str) -> None:
    """Execute complete provenance, spectral forensics, and multi-detector inference for a job."""
    db = SessionLocal()
    try:
        job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        if not job:
            return

        if is_job_cancelled(job_id):
            job.status = "cancelled"
            job.current_stage = "Cancelled before start"
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            return

        asset = db.query(AudioAsset).filter(AudioAsset.id == job.audio_asset_id).first()
        if not asset:
            job.status = "failed"
            job.error_message = "Associated AudioAsset not found."
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            return

        job.started_at = datetime.now(timezone.utc)
        job.software_versions_json = json.dumps(get_software_versions())
        _update_job_stage(db, job, "running", 15.0, "Inspecting C2PA provenance and conventional metadata")

        # Step 1: Provenance & conventional metadata
        existing_prov = (
            db.query(ProvenanceResult)
            .filter(ProvenanceResult.audio_asset_id == asset.id)
            .first()
        )
        if not existing_prov:
            run_provenance_analysis(db, asset, job_id=job.id)

        if is_job_cancelled(job_id):
            job.status = "cancelled"
            job.current_stage = "Cancelled after provenance inspection"
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            return

        # Step 2: Mathematical signal forensics & fakeprint extraction
        _update_job_stage(db, job, "running", 45.0, "Extracting time-domain, STFT, and 1-8 kHz fakeprint features")
        file_path = resolve_asset_path(asset)
        forensics_full = run_complete_forensics(file_path)

        samples = forensics_full.pop("_samples")
        sr = forensics_full.pop("_sr")
        fakeprint_vec = forensics_full.pop("_fakeprint_vector")

        # Pre-compute sliding-window temporal analysis using already-decoded samples in memory
        prov_row = (
            db.query(ProvenanceResult)
            .filter(ProvenanceResult.audio_asset_id == asset.id)
            .first()
        )
        prov_ctx = {
            "c2pa_status": prov_row.c2pa_status if prov_row else "no_c2pa_found",
            "c2pa_verified": bool(prov_row and prov_row.credential_present and prov_row.signature_valid),
            "claim_generator": prov_row.claim_generator if prov_row else None,
        }
        try:
            temp_res = compute_temporal_forensic_analysis(
                samples=samples,
                sr=sr,
                asset_sha256=asset.sha256_checksum,
                hop_sec=2.5,
                fakeprint_window_sec=5.0,
                include_exploratory_fakeprint=True,
                provenance_context=prov_ctx,
                job_id=job.id,
            )
            forensics_full["temporal_summary"] = {
                "hop_sec": 2.5,
                "window_sec": 5.0,
                "summary_statistics": temp_res.get("summary_statistics", {}),
                "region_summaries": temp_res.get("region_summaries", []),
                "section_transitions": temp_res.get("section_transitions", []),
                "hybrid_forensic_analysis": temp_res.get("hybrid_forensic_analysis", {}),
            }
        except Exception:
            pass

        job.forensics_json = json.dumps(forensics_full)
        db.commit()

        if is_job_cancelled(job_id):
            job.status = "cancelled"
            job.current_stage = "Cancelled after spectral forensics"
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            return

        # Step 3: Run registered AI detectors
        _update_job_stage(db, job, "running", 75.0, "Running AI-music detectors (Fakeprint ONNX & SONICS)")
        # Clear any old DetectorRun records for this job_id + asset_id to avoid duplicates on re-run
        db.query(DetectorRun).filter(
            DetectorRun.job_id == job.id,
            DetectorRun.audio_asset_id == asset.id,
        ).delete()
        db.commit()

        predictions = detector_registry.run_all(samples, sr, precomputed_fakeprint=None)
        warnings_list = []

        for pred in predictions:
            run_row = DetectorRun(
                job_id=job.id,
                audio_asset_id=asset.id,
                detector_id=pred.detector_id,
                detector_name=pred.detector_name,
                detector_version=pred.detector_version,
                model_source=pred.model_source,
                model_revision=pred.model_revision,
                probability=pred.probability,
                raw_score=pred.raw_score,
                threshold=pred.threshold,
                predicted_label=pred.predicted_label,
                confidence=pred.confidence,
                input_sample_rate=pred.input_sample_rate,
                input_duration_sec=pred.input_duration_sec,
                processing_time_ms=pred.processing_time_ms,
                device_used=pred.device_used,
                feature_dim=pred.feature_dim,
                status=pred.status,
                error_message=pred.error_message,
                details_json=json.dumps(pred.details),
            )
            db.add(run_row)
            if pred.details and pred.details.get("ood_warning"):
                warnings_list.append(str(pred.details["ood_warning"]))
            if pred.status == "unavailable":
                warnings_list.append(f"Optional detector '{pred.detector_id}' is unavailable: {pred.error_message}")
            elif pred.status == "error":
                warnings_list.append(f"Detector '{pred.detector_id}' reported error: {pred.error_message}")

        # Check for conflicting detector results as a scientific observation
        valid_probs = [
            p
            for p in predictions
            if p.status == "success"
            and p.probability is not None
            and not bool(p.details.get("abstained", False))
        ]
        if len(valid_probs) >= 2:
            labels_bool = [p.probability >= p.threshold for p in valid_probs if p.probability is not None]
            p_vals = [float(p.probability) for p in valid_probs if p.probability is not None]
            inter_model_std = float(np.std(p_vals)) if len(p_vals) >= 2 else 0.0
            if len(set(labels_bool)) > 1:
                warnings_list.append(
                    f"Model Disagreement Alert (inter-detector σ = {inter_model_std:.3f}): Independent detectors "
                    "produced diverging classifications. Fakeprint ONNX measures 1–8 kHz 200/400 Hz transposed-convolution "
                    "spectral hull residuals, whereas SONICS SpecTTTra-γ (5s) and SpecTTTra-α (120s) evaluate "
                    "spectro-temporal mel tokens."
                )

        job.warnings_json = json.dumps(warnings_list)
        job.status = "completed"
        job.progress = 100.0
        job.current_stage = "Forensic analysis complete"
        job.completed_at = datetime.now(timezone.utc)
        db.commit()

    except Exception as exc:
        try:
            job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
            if job:
                job.status = "failed"
                job.current_stage = "Analysis failed"
                job.error_message = str(exc)
                job.completed_at = datetime.now(timezone.utc)
                db.commit()
        except Exception:
            pass
    finally:
        db.close()


def submit_analysis_job(job_id: str, run_sync: bool = False) -> None:
    """Submit an analysis job to the bounded worker pool or run synchronously when requested."""
    if run_sync:
        execute_analysis_job(job_id)
    else:
        _executor.submit(execute_analysis_job, job_id)
