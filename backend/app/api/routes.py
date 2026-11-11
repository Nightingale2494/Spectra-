"""REST API endpoints for SPECTRA."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import c2pa
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from sqlalchemy import text
from sqlalchemy.orm import Session
import torch

from app.core.config import settings
from app.database.models import (
    AnalysisJob,
    AudioAsset,
    ComparisonResult,
    DetectorRun,
    GeneratedReport,
    ProcessingExperiment,
    ProvenanceResult,
)
from app.database.session import get_db
from app.detectors.registry import detector_registry
from app.integrations.comfyui_adapter import comfyui_adapter
from app.processing.comparison_engine import compute_before_after_comparison
from app.processing.dsp_engine import DSP_PRESETS, run_dsp_experiment
from app.schemas.api_schemas import (
    AnalysisJobOut,
    AnalysisResultsOut,
    AudioAssetOut,
    ComfyUIStatusOut,
    ComparisonResultOut,
    DetectorInfoOut,
    DetectorResultOut,
    GeneratedReportOut,
    HealthOut,
    MetadataExportRequest,
    ProcessingExperimentOut,
    ProcessRequest,
    ProvenanceResultOut,
    ReportGenerateRequest,
    SpectrogramOut,
    TemporalAnalysisRequest,
    UploadResponse,
    WaveformOut,
)
from app.services.forensics_service import (
    compute_spectrogram_payload,
    compute_waveform_envelope,
)
from app.services.ingestion_service import (
    AudioIngestionError,
    decode_audio_pcm,
    ingest_uploaded_bytes,
    resolve_asset_path,
)
from app.services.provenance_service import export_metadata_clean_asset
from app.services.report_service import build_report_payload, generate_and_store_report
from app.services.sample_generator import ensure_reference_samples
from app.services.temporal_service import (
    compute_temporal_dsp_comparison,
    compute_temporal_forensic_analysis,
    request_temporal_job_cancel,
)
from app.workers.job_runner import cancel_job, get_software_versions, submit_analysis_job

router = APIRouter(prefix="/api")


def _safe_loads(raw: Optional[str], default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default


def _serialize_job(job: AnalysisJob) -> AnalysisJobOut:
    return AnalysisJobOut(
        id=job.id,
        audio_asset_id=job.audio_asset_id,
        job_type=job.job_type,
        status=job.status,
        progress=job.progress,
        current_stage=job.current_stage,
        error_message=job.error_message,
        warnings=_safe_loads(job.warnings_json, []),
        started_at=job.started_at,
        completed_at=job.completed_at,
        created_at=job.created_at,
    )


def _serialize_provenance(prov: ProvenanceResult) -> ProvenanceResultOut:
    return ProvenanceResultOut(
        id=prov.id,
        audio_asset_id=prov.audio_asset_id,
        c2pa_status=prov.c2pa_status,
        credential_present=prov.credential_present,
        signature_valid=prov.signature_valid,
        active_manifest_id=prov.active_manifest_id,
        claim_generator=prov.claim_generator,
        assertions=_safe_loads(prov.assertions_json, []),
        ingredients=_safe_loads(prov.ingredients_json, []),
        certificate_info=_safe_loads(prov.certificate_info_json, {}),
        validation_errors=_safe_loads(prov.validation_errors_json, []),
        validation_warnings=_safe_loads(prov.validation_warnings_json, []),
        raw_manifest=_safe_loads(prov.raw_manifest_json, None),
        conventional_metadata=_safe_loads(prov.conventional_metadata_json, {}),
        supported_fields=_safe_loads(prov.supported_fields_json, []),
        unsupported_fields=_safe_loads(prov.unsupported_fields_json, []),
        created_at=prov.created_at,
    )


def _serialize_detector_run(d: DetectorRun) -> DetectorResultOut:
    return DetectorResultOut(
        id=d.id,
        job_id=d.job_id,
        audio_asset_id=d.audio_asset_id,
        detector_id=d.detector_id,
        detector_name=d.detector_name,
        detector_version=d.detector_version,
        model_source=d.model_source,
        model_revision=d.model_revision,
        probability=d.probability,
        raw_score=d.raw_score,
        threshold=d.threshold,
        predicted_label=d.predicted_label,
        confidence=d.confidence,
        input_sample_rate=d.input_sample_rate,
        input_duration_sec=d.input_duration_sec,
        processing_time_ms=d.processing_time_ms,
        device_used=d.device_used,
        feature_dim=d.feature_dim,
        status=d.status,
        error_message=d.error_message,
        details=_safe_loads(d.details_json, {}),
        created_at=d.created_at,
    )


def _serialize_experiment(e: ProcessingExperiment) -> ProcessingExperimentOut:
    return ProcessingExperimentOut(
        id=e.id,
        job_id=e.job_id,
        source_asset_id=e.source_asset_id,
        output_asset_id=e.output_asset_id,
        operation_type=e.operation_type,
        preset_name=e.preset_name,
        parameters=_safe_loads(e.parameters_json, {}),
        engine_used=e.engine_used,
        original_sha256=e.original_sha256,
        processed_sha256=e.processed_sha256,
        duration_preserved=e.duration_preserved,
        channels_preserved=e.channels_preserved,
        clipping_detected=e.clipping_detected,
        nan_inf_detected=e.nan_inf_detected,
        reconstruction_valid=e.reconstruction_valid,
        processing_time_ms=e.processing_time_ms,
        audit_log=_safe_loads(e.audit_log_json, []),
        warnings=_safe_loads(e.warnings_json, []),
        status=e.status,
        error_message=e.error_message,
        created_at=e.created_at,
    )


def _serialize_comparison(c: ComparisonResult) -> ComparisonResultOut:
    return ComparisonResultOut(
        id=c.id,
        experiment_id=c.experiment_id,
        source_asset_id=c.source_asset_id,
        processed_asset_id=c.processed_asset_id,
        delta_rms_db=c.delta_rms_db,
        delta_peak_db=c.delta_peak_db,
        spectral_distance_l2=c.spectral_distance_l2,
        spectral_convergence=c.spectral_convergence,
        snr_db=c.snr_db,
        psnr_db=c.psnr_db,
        loudness_delta_db=c.loudness_delta_db,
        duration_delta_sec=c.duration_delta_sec,
        alignment_offset_samples=c.alignment_offset_samples,
        original_clipping_ratio=c.original_clipping_ratio,
        processed_clipping_ratio=c.processed_clipping_ratio,
        detector_deltas=_safe_loads(c.detector_deltas_json, []),
        spectrum_comparison=_safe_loads(c.spectrum_comparison_json, {}),
        residual_comparison=_safe_loads(c.residual_comparison_json, {}),
        observations=_safe_loads(c.observations_json, []),
        warnings=_safe_loads(c.warnings_json, []),
        created_at=c.created_at,
    )


@router.get("/health", response_model=HealthOut)
def get_system_health(db: Session = Depends(get_db)) -> HealthOut:
    db_ok = False
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    storage_writable = os.access(str(settings.storage_path), os.W_OK)
    ffmpeg_path = settings.resolve_ffmpeg_exe()
    dets = detector_registry.list_all()
    det_ready = sum(1 for d in dets if d["available"])
    comfy_status = comfyui_adapter.check_status()

    return HealthOut(
        status="healthy" if (db_ok and storage_writable) else "degraded",
        version=settings.SPECTRA_VERSION,
        environment=settings.SPECTRA_ENV,
        database_ok=db_ok,
        storage_writable=storage_writable,
        ffmpeg_available=ffmpeg_path is not None,
        ffmpeg_path=Path(ffmpeg_path).name if ffmpeg_path else None,
        pyav_available=True,
        c2pa_sdk_version=c2pa.sdk_version() if hasattr(c2pa, "sdk_version") else "0.91.0",
        cpu_count=os.cpu_count() or 1,
        cuda_available=torch.cuda.is_available(),
        gpu_info=comfy_status.get("gpu_name"),
        detectors_ready=det_ready,
        detectors_total=len(dets),
        comfyui_reachable=bool(comfy_status.get("service_reachable", False)),
    )


@router.get("/integrations/comfyui/status", response_model=ComfyUIStatusOut)
def get_comfyui_status() -> ComfyUIStatusOut:
    status_dict = comfyui_adapter.check_status()
    return ComfyUIStatusOut(**status_dict)


@router.get("/detectors", response_model=List[DetectorInfoOut])
def list_detectors() -> List[DetectorInfoOut]:
    return [DetectorInfoOut(**d) for d in detector_registry.list_all()]


@router.get("/detectors/health")
def get_detectors_health() -> Dict[str, Any]:
    infos = detector_registry.list_all()
    return {
        "total": len(infos),
        "available": sum(1 for d in infos if d["available"]),
        "loaded": sum(1 for d in infos if d["loaded"]),
        "detectors": infos,
    }


@router.get("/dsp/presets")
def get_dsp_presets() -> List[Dict[str, Any]]:
    return DSP_PRESETS


@router.get("/settings/info")
def get_settings_info(db: Session = Depends(get_db)) -> Dict[str, Any]:
    import hashlib
    model_files = []
    for rel_path in [
        "fakeprint/ai_music_detector.onnx",
        "fakeprint/config.yaml",
        "fakeprint/model_config.yaml",
        "sonics/sonics-spectttra-gamma-5s/pytorch_model.bin",
        "sonics/sonics-spectttra-gamma-5s/config.json",
    ]:
        p = settings.models_path / rel_path
        if p.exists():
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            model_files.append({
                "path": f"models/{rel_path}",
                "exists": True,
                "size_bytes": p.stat().st_size,
                "sha256": h,
            })
        else:
            model_files.append({
                "path": f"models/{rel_path}",
                "exists": False,
                "size_bytes": 0,
                "sha256": None,
            })

    total_assets = db.query(AudioAsset).count()
    total_jobs = db.query(AnalysisJob).count()
    total_experiments = db.query(ProcessingExperiment).count()
    total_reports = db.query(GeneratedReport).count()

    return {
        "version": settings.SPECTRA_VERSION,
        "environment": settings.SPECTRA_ENV,
        "max_upload_mb": settings.SPECTRA_MAX_UPLOAD_MB,
        "default_device": settings.SPECTRA_DEFAULT_DEVICE,
        "enable_sonics_alpha_120s": settings.SPECTRA_ENABLE_SONICS_ALPHA_120S,
        "supported_formats": [
            {"ext": "WAV", "mime": "audio/wav", "c2pa_read": True, "lossless": True},
            {"ext": "FLAC", "mime": "audio/flac", "c2pa_read": True, "lossless": True},
            {"ext": "MP3", "mime": "audio/mpeg", "c2pa_read": True, "lossless": False},
            {"ext": "M4A / AAC", "mime": "audio/mp4", "c2pa_read": True, "lossless": False},
        ],
        "storage_stats": {
            "storage_dir": "storage/",
            "models_dir": "models/",
            "total_assets": total_assets,
            "total_jobs": total_jobs,
            "total_experiments": total_experiments,
            "total_reports": total_reports,
        },
        "model_files": model_files,
        "software_versions": get_software_versions(),
        "comfyui": comfyui_adapter.check_status(),
    }


@router.get("/benchmarks/latest")
def get_latest_benchmark_results() -> Dict[str, Any]:
    from app.core.config import PROJECT_ROOT
    bench_file = PROJECT_ROOT / "research" / "benchmarks" / "latest_benchmark_results.json"
    if not bench_file.exists():
        return {"available": False, "data": None}
    try:
        return {"available": True, "data": json.loads(bench_file.read_text(encoding="utf-8"))}
    except Exception as exc:
        return {"available": False, "error": str(exc), "data": None}



@router.get("/audio/samples")
def list_reference_samples() -> List[Dict[str, str]]:
    samples = ensure_reference_samples()
    # Strip internal filesystem path before returning to client
    return [
        {
            "sample_id": s["sample_id"],
            "filename": s["filename"],
            "title": s["title"],
            "description": s["description"],
        }
        for s in samples
    ]


@router.post("/audio/load-sample/{sample_id}", response_model=UploadResponse)
def load_reference_sample(
    sample_id: str,
    run_sync: bool = Query(default=True),
    db: Session = Depends(get_db),
) -> UploadResponse:
    alias_map = {
        "acoustic_studio": "acoustic_reference",
        "neural_vocoder_synth": "vocoder_artifact",
        "c2pa_signed": "c2pa_signed_demo",
    }
    resolved_id = alias_map.get(sample_id, sample_id)
    samples = ensure_reference_samples()
    match = next((s for s in samples if s["sample_id"] == resolved_id), None)
    if not match:
        raise HTTPException(status_code=404, detail=f"Reference sample '{sample_id}' not found.")

    raw_bytes = Path(match["path"]).read_bytes()
    try:
        asset = ingest_uploaded_bytes(
            db=db,
            raw_bytes=raw_bytes,
            original_filename=match["filename"],
        )
    except AudioIngestionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)

    job = AnalysisJob(
        audio_asset_id=asset.id,
        job_type="full_analysis",
        status="queued",
        progress=0.0,
        current_stage="Queued for forensic inspection",
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    submit_analysis_job(job.id, run_sync=run_sync)
    return UploadResponse(
        asset=AudioAssetOut.model_validate(asset),
        initial_job_id=job.id,
        message=f"Loaded reference sample '{match['title']}' and initiated forensic analysis.",
    )


@router.post("/audio/upload", response_model=UploadResponse)
async def upload_audio(
    file: UploadFile = File(...),
    run_sync: bool = Query(default=True),
    db: Session = Depends(get_db),
) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename.")

    raw_bytes = await file.read()
    try:
        asset = ingest_uploaded_bytes(
            db=db,
            raw_bytes=raw_bytes,
            original_filename=file.filename,
        )
    except AudioIngestionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)

    job = AnalysisJob(
        audio_asset_id=asset.id,
        job_type="full_analysis",
        status="queued",
        progress=0.0,
        current_stage="Queued for forensic inspection",
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    submit_analysis_job(job.id, run_sync=run_sync)
    return UploadResponse(
        asset=AudioAssetOut.model_validate(asset),
        initial_job_id=job.id,
        message="Audio uploaded, verified, and forensic analysis initiated.",
    )


@router.get("/audio", response_model=List[AudioAssetOut])
def list_audio_assets(
    include_variants: bool = Query(default=True),
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> List[AudioAssetOut]:
    q = db.query(AudioAsset)
    if not include_variants:
        q = q.filter(AudioAsset.is_processed_variant.is_(False))
    rows = q.order_by(AudioAsset.created_at.desc()).limit(limit).all()
    return [AudioAssetOut.model_validate(r) for r in rows]


@router.get("/audio/{asset_id}", response_model=AudioAssetOut)
def get_audio_asset(asset_id: str, db: Session = Depends(get_db)) -> AudioAssetOut:
    asset = db.query(AudioAsset).filter(AudioAsset.id == asset_id).first()
    if not asset:
        raise HTTPException(status_code=404, detail="AudioAsset not found.")
    return AudioAssetOut.model_validate(asset)


@router.get("/audio/{asset_id}/stream")
def stream_audio_asset(asset_id: str, db: Session = Depends(get_db)) -> FileResponse:
    asset = db.query(AudioAsset).filter(AudioAsset.id == asset_id).first()
    if not asset:
        raise HTTPException(status_code=404, detail="AudioAsset not found.")
    try:
        file_path = resolve_asset_path(asset)
    except AudioIngestionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)

    return FileResponse(
        path=str(file_path),
        media_type=asset.mime_type,
        filename=asset.sanitized_filename,
    )


@router.get("/audio/{asset_id}/waveform", response_model=WaveformOut)
def get_audio_waveform(
    asset_id: str,
    points: int = Query(default=600, ge=64, le=2000),
    db: Session = Depends(get_db),
) -> WaveformOut:
    asset = db.query(AudioAsset).filter(AudioAsset.id == asset_id).first()
    if not asset:
        raise HTTPException(status_code=404, detail="AudioAsset not found.")
    file_path = resolve_asset_path(asset)
    samples, sr = decode_audio_pcm(file_path, target_sr=None, mono=False)
    env = compute_waveform_envelope(samples, sr, num_points=points)
    return WaveformOut(asset_id=asset.id, **env)


@router.get("/audio/{asset_id}/spectrogram", response_model=SpectrogramOut)
def get_audio_spectrogram(
    asset_id: str,
    db: Session = Depends(get_db),
) -> SpectrogramOut:
    asset = db.query(AudioAsset).filter(AudioAsset.id == asset_id).first()
    if not asset:
        raise HTTPException(status_code=404, detail="AudioAsset not found.")
    file_path = resolve_asset_path(asset)
    payload = compute_spectrogram_payload(file_path)
    return SpectrogramOut(asset_id=asset.id, **payload)


@router.get("/jobs")
def list_jobs(
    limit: int = Query(default=30, ge=1, le=100),
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    jobs = db.query(AnalysisJob).order_by(AnalysisJob.created_at.desc()).limit(limit).all()
    out = []
    for j in jobs:
        asset = db.query(AudioAsset).filter(AudioAsset.id == j.audio_asset_id).first()
        dets = db.query(DetectorRun).filter(DetectorRun.job_id == j.id, DetectorRun.audio_asset_id == j.audio_asset_id).all()
        prov = db.query(ProvenanceResult).filter(ProvenanceResult.audio_asset_id == j.audio_asset_id).first()
        exps_count = db.query(ProcessingExperiment).filter(ProcessingExperiment.job_id == j.id).count()
        out.append({
            "job": _serialize_job(j).model_dump(),
            "asset": AudioAssetOut.model_validate(asset).model_dump() if asset else None,
            "c2pa_status": prov.c2pa_status if prov else None,
            "detectors": [_serialize_detector_run(d).model_dump() for d in dets],
            "experiments_count": exps_count,
        })
    return out


@router.post("/jobs/{job_id}/analyze", response_model=AnalysisJobOut)
def trigger_job_analysis(
    job_id: str,
    run_sync: bool = Query(default=True),
    db: Session = Depends(get_db),
) -> AnalysisJobOut:
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        # Also allow passing an audio_asset_id to create a new job
        asset = db.query(AudioAsset).filter(AudioAsset.id == job_id).first()
        if not asset:
            raise HTTPException(status_code=404, detail="AnalysisJob or AudioAsset not found.")
        job = AnalysisJob(
            audio_asset_id=asset.id,
            job_type="full_analysis",
            status="queued",
            progress=0.0,
            current_stage="Queued for forensic inspection",
        )
        db.add(job)
        db.commit()
        db.refresh(job)

    submit_analysis_job(job.id, run_sync=run_sync)
    db.refresh(job)
    return _serialize_job(job)


@router.post("/jobs/{job_id}/cancel", response_model=AnalysisJobOut)
def cancel_analysis_job(job_id: str, db: Session = Depends(get_db)) -> AnalysisJobOut:
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="AnalysisJob not found.")
    cancel_job(job.id)
    if job.status in {"queued", "running"}:
        job.status = "cancelled"
        job.current_stage = "Cancelled by user"
        db.commit()
        db.refresh(job)
    return _serialize_job(job)


@router.get("/jobs/{job_id}", response_model=AnalysisJobOut)
def get_job_status(job_id: str, db: Session = Depends(get_db)) -> AnalysisJobOut:
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="AnalysisJob not found.")
    return _serialize_job(job)


@router.get("/jobs/{job_id}/results", response_model=AnalysisResultsOut)
def get_job_results(job_id: str, db: Session = Depends(get_db)) -> AnalysisResultsOut:
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="AnalysisJob not found.")

    asset = db.query(AudioAsset).filter(AudioAsset.id == job.audio_asset_id).first()
    if not asset:
        raise HTTPException(status_code=404, detail="Associated AudioAsset not found.")

    prov = (
        db.query(ProvenanceResult)
        .filter(ProvenanceResult.audio_asset_id == asset.id)
        .order_by(ProvenanceResult.created_at.desc())
        .first()
    )
    detectors = (
        db.query(DetectorRun)
        .filter(DetectorRun.job_id == job.id, DetectorRun.audio_asset_id == asset.id)
        .all()
    )
    experiments = (
        db.query(ProcessingExperiment)
        .filter(ProcessingExperiment.source_asset_id == asset.id)
        .order_by(ProcessingExperiment.created_at.desc())
        .all()
    )
    comparisons: List[ComparisonResult] = []
    for exp in experiments:
        comp = (
            db.query(ComparisonResult)
            .filter(ComparisonResult.experiment_id == exp.id)
            .order_by(ComparisonResult.created_at.desc())
            .first()
        )
        if comp:
            comparisons.append(comp)

    return AnalysisResultsOut(
        job=_serialize_job(job),
        asset=AudioAssetOut.model_validate(asset),
        provenance=_serialize_provenance(prov) if prov else None,
        forensics=_safe_loads(job.forensics_json, {}),
        detectors=[_serialize_detector_run(d) for d in detectors],
        experiments=[_serialize_experiment(e) for e in experiments],
        comparisons=[_serialize_comparison(c) for c in comparisons],
        software_versions=_safe_loads(job.software_versions_json, get_software_versions()),
    )


def _run_temporal_for_job(
    db: Session,
    job: AnalysisJob,
    hop_sec: float,
    fakeprint_window_sec: float,
    include_exploratory_fakeprint: bool,
    force_recompute: bool,
) -> Dict[str, Any]:
    asset = db.query(AudioAsset).filter(AudioAsset.id == job.audio_asset_id).first()
    if not asset:
        raise HTTPException(status_code=404, detail="Associated AudioAsset not found.")

    file_path = resolve_asset_path(asset)
    samples, sr = decode_audio_pcm(file_path, target_sr=None, mono=False)

    prov = (
        db.query(ProvenanceResult)
        .filter(ProvenanceResult.audio_asset_id == asset.id)
        .order_by(ProvenanceResult.created_at.desc())
        .first()
    )
    prov_ctx = {
        "c2pa_status": prov.c2pa_status if prov else "no_c2pa_found",
        "c2pa_verified": bool(prov and prov.credential_present and prov.signature_valid),
        "claim_generator": prov.claim_generator if prov else None,
        "conventional_metadata": _safe_loads(prov.conventional_metadata_json, {}) if prov else {},
    }

    return compute_temporal_forensic_analysis(
        samples=samples,
        sr=sr,
        asset_sha256=asset.sha256_checksum,
        hop_sec=hop_sec,
        fakeprint_window_sec=fakeprint_window_sec,
        include_exploratory_fakeprint=include_exploratory_fakeprint,
        provenance_context=prov_ctx,
        force_recompute=force_recompute,
        job_id=job.id,
    )


@router.get("/jobs/{job_id}/temporal")
def get_job_temporal_analysis(
    job_id: str,
    hop_sec: float = Query(default=2.5, ge=0.5, le=10.0),
    fakeprint_window_sec: float = Query(default=5.0, ge=2.5, le=30.0),
    include_exploratory_fakeprint: bool = Query(default=True),
    force_recompute: bool = Query(default=False),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="AnalysisJob not found.")
    try:
        return _run_temporal_for_job(
            db=db,
            job=job,
            hop_sec=hop_sec,
            fakeprint_window_sec=fakeprint_window_sec,
            include_exploratory_fakeprint=include_exploratory_fakeprint,
            force_recompute=force_recompute,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Temporal analysis failed: {exc}")


@router.post("/jobs/{job_id}/temporal")
def post_job_temporal_analysis(
    job_id: str,
    req: TemporalAnalysisRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="AnalysisJob not found.")
    try:
        return _run_temporal_for_job(
            db=db,
            job=job,
            hop_sec=req.hop_sec,
            fakeprint_window_sec=req.fakeprint_window_sec,
            include_exploratory_fakeprint=req.include_exploratory_fakeprint,
            force_recompute=req.force_recompute,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Temporal analysis failed: {exc}")


@router.post("/jobs/{job_id}/temporal/cancel")
def cancel_job_temporal_analysis(job_id: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="AnalysisJob not found.")
    request_temporal_job_cancel(job.id)
    return {"job_id": job.id, "status": "cancel_requested"}


@router.get("/jobs/{job_id}/temporal-compare")
def get_job_temporal_comparison(
    job_id: str,
    experiment_id: Optional[str] = Query(default=None),
    hop_sec: float = Query(default=2.5, ge=0.5, le=10.0),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="AnalysisJob not found.")
    orig_asset = db.query(AudioAsset).filter(AudioAsset.id == job.audio_asset_id).first()
    if not orig_asset:
        raise HTTPException(status_code=404, detail="Original AudioAsset not found.")

    exp_q = db.query(ProcessingExperiment).filter(ProcessingExperiment.source_asset_id == orig_asset.id)
    if experiment_id:
        exp_q = exp_q.filter(ProcessingExperiment.id == experiment_id)
    exp = exp_q.order_by(ProcessingExperiment.created_at.desc()).first()
    if not exp or not exp.output_asset_id:
        raise HTTPException(status_code=404, detail="No processed audio experiment found for temporal comparison.")

    proc_asset = db.query(AudioAsset).filter(AudioAsset.id == exp.output_asset_id).first()
    if not proc_asset:
        raise HTTPException(status_code=404, detail="Processed AudioAsset not found.")

    comp = (
        db.query(ComparisonResult)
        .filter(ComparisonResult.experiment_id == exp.id)
        .order_by(ComparisonResult.created_at.desc())
        .first()
    )
    offset_samples = comp.alignment_offset_samples if comp else 0

    orig_path = resolve_asset_path(orig_asset)
    proc_path = resolve_asset_path(proc_asset)
    orig_samples, orig_sr = decode_audio_pcm(orig_path, target_sr=None, mono=False)
    proc_samples, proc_sr = decode_audio_pcm(proc_path, target_sr=None, mono=False)

    return compute_temporal_dsp_comparison(
        orig_samples=orig_samples,
        orig_sr=orig_sr,
        orig_sha256=orig_asset.sha256_checksum,
        proc_samples=proc_samples,
        proc_sr=proc_sr,
        proc_sha256=proc_asset.sha256_checksum,
        alignment_offset_samples=offset_samples,
        hop_sec=hop_sec,
        include_exploratory_fakeprint=True,
    )


@router.post("/jobs/{job_id}/process")
def process_job_audio(
    job_id: str,
    req: ProcessRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="AnalysisJob not found.")

    asset = db.query(AudioAsset).filter(AudioAsset.id == job.audio_asset_id).first()
    if not asset:
        raise HTTPException(status_code=404, detail="Source AudioAsset not found.")

    try:
        out_asset, exp, orig_samples, orig_sr, proc_samples, proc_sr = run_dsp_experiment(
            db=db,
            asset=asset,
            job=job,
            operation_type=req.operation_type,
            parameters=req.parameters,
            preset_name=req.preset_name,
            engine=req.engine,
            export_format=req.export_format,
        )
        comp = compute_before_after_comparison(
            db=db,
            experiment=exp,
            source_asset=asset,
            processed_asset=out_asset,
            orig_samples=orig_samples,
            orig_sr=orig_sr,
            proc_samples=proc_samples,
            proc_sr=proc_sr,
            run_detectors=req.run_detectors_on_output,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"DSP processing error: {exc}")

    return {
        "experiment": _serialize_experiment(exp).model_dump(),
        "processed_asset": AudioAssetOut.model_validate(out_asset).model_dump(),
        "comparison": _serialize_comparison(comp).model_dump(),
    }


@router.post("/jobs/{job_id}/metadata-export")
def export_job_clean_metadata(
    job_id: str,
    req: MetadataExportRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="AnalysisJob not found.")
    asset = db.query(AudioAsset).filter(AudioAsset.id == job.audio_asset_id).first()
    if not asset:
        raise HTTPException(status_code=404, detail="Source AudioAsset not found.")

    out_asset, exp = export_metadata_clean_asset(
        db=db,
        asset=asset,
        job_id=job.id,
        strip_conventional=req.strip_conventional_metadata,
        strip_c2pa=req.strip_c2pa_manifest,
        custom_tags=req.custom_tags,
    )
    return {
        "experiment": _serialize_experiment(exp).model_dump(),
        "processed_asset": AudioAssetOut.model_validate(out_asset).model_dump(),
    }


@router.get("/jobs/{job_id}/compare")
def get_job_comparisons(job_id: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="AnalysisJob not found.")

    asset = db.query(AudioAsset).filter(AudioAsset.id == job.audio_asset_id).first()
    if not asset:
        raise HTTPException(status_code=404, detail="Source AudioAsset not found.")

    experiments = (
        db.query(ProcessingExperiment)
        .filter(ProcessingExperiment.source_asset_id == asset.id)
        .order_by(ProcessingExperiment.created_at.desc())
        .all()
    )
    items = []
    for exp in experiments:
        comp = (
            db.query(ComparisonResult)
            .filter(ComparisonResult.experiment_id == exp.id)
            .order_by(ComparisonResult.created_at.desc())
            .first()
        )
        proc_asset = (
            db.query(AudioAsset).filter(AudioAsset.id == exp.output_asset_id).first()
            if exp.output_asset_id
            else None
        )
        items.append({
            "experiment": _serialize_experiment(exp).model_dump(),
            "processed_asset": AudioAssetOut.model_validate(proc_asset).model_dump() if proc_asset else None,
            "comparison": _serialize_comparison(comp).model_dump() if comp else None,
        })

    return {
        "job_id": job.id,
        "source_asset": AudioAssetOut.model_validate(asset).model_dump(),
        "items": items,
    }


@router.get("/jobs/{job_id}/report")
def get_job_report(
    job_id: str,
    format: str = Query(default="json", pattern="^(json|pdf)$"),
    experiment_id: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
) -> Response:
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="AnalysisJob not found.")
    asset = db.query(AudioAsset).filter(AudioAsset.id == job.audio_asset_id).first()
    if not asset:
        raise HTTPException(status_code=404, detail="AudioAsset not found.")

    report_row, raw_bytes = generate_and_store_report(
        db=db,
        job=job,
        asset=asset,
        report_format=format,
        experiment_id=experiment_id,
    )
    media_type = "application/pdf" if format == "pdf" else "application/json"
    ext = "pdf" if format == "pdf" else "json"
    stem = Path(asset.sanitized_filename).stem
    filename = f"spectra_report_{stem}_{report_row.id[:8]}.{ext}"

    return Response(
        content=raw_bytes,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/jobs/{job_id}/report", response_model=GeneratedReportOut)
def create_job_report_record(
    job_id: str,
    req: ReportGenerateRequest,
    db: Session = Depends(get_db),
) -> GeneratedReportOut:
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="AnalysisJob not found.")
    asset = db.query(AudioAsset).filter(AudioAsset.id == job.audio_asset_id).first()
    if not asset:
        raise HTTPException(status_code=404, detail="AudioAsset not found.")

    report_row, _ = generate_and_store_report(
        db=db,
        job=job,
        asset=asset,
        report_format=req.report_format,
        experiment_id=req.experiment_id,
        include_raw_manifest=req.include_raw_manifest,
    )
    return GeneratedReportOut(
        id=report_row.id,
        job_id=report_row.job_id,
        experiment_id=report_row.experiment_id,
        audio_asset_id=report_row.audio_asset_id,
        report_format=report_row.report_format,
        title=report_row.title,
        sha256_checksum=report_row.sha256_checksum,
        file_size_bytes=report_row.file_size_bytes,
        summary=_safe_loads(report_row.summary_json, {}),
        created_at=report_row.created_at,
    )


@router.get("/jobs/{job_id}/download")
def download_job_artifact(
    job_id: str,
    target: str = Query(default="processed", pattern="^(original|processed|report_json|report_pdf)$"),
    experiment_id: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
) -> Response:
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="AnalysisJob not found.")
    orig_asset = db.query(AudioAsset).filter(AudioAsset.id == job.audio_asset_id).first()
    if not orig_asset:
        raise HTTPException(status_code=404, detail="Original AudioAsset not found.")

    if target == "original":
        path = resolve_asset_path(orig_asset)
        return FileResponse(
            path=str(path),
            media_type=orig_asset.mime_type,
            filename=orig_asset.sanitized_filename,
        )

    if target == "processed":
        exp_q = db.query(ProcessingExperiment).filter(ProcessingExperiment.source_asset_id == orig_asset.id)
        if experiment_id:
            exp_q = exp_q.filter(ProcessingExperiment.id == experiment_id)
        exp = exp_q.order_by(ProcessingExperiment.created_at.desc()).first()
        if not exp or not exp.output_asset_id:
            raise HTTPException(status_code=404, detail="No processed audio output found for this job.")
        proc_asset = db.query(AudioAsset).filter(AudioAsset.id == exp.output_asset_id).first()
        if not proc_asset:
            raise HTTPException(status_code=404, detail="Processed AudioAsset not found.")
        path = resolve_asset_path(proc_asset)
        return FileResponse(
            path=str(path),
            media_type=proc_asset.mime_type,
            filename=proc_asset.sanitized_filename,
        )

    fmt = "pdf" if target == "report_pdf" else "json"
    return get_job_report(job_id=job_id, format=fmt, experiment_id=experiment_id, db=db)


@router.get("/reports", response_model=List[GeneratedReportOut])
def list_generated_reports(
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> List[GeneratedReportOut]:
    rows = db.query(GeneratedReport).order_by(GeneratedReport.created_at.desc()).limit(limit).all()
    return [
        GeneratedReportOut(
            id=r.id,
            job_id=r.job_id,
            experiment_id=r.experiment_id,
            audio_asset_id=r.audio_asset_id,
            report_format=r.report_format,
            title=r.title,
            sha256_checksum=r.sha256_checksum,
            file_size_bytes=r.file_size_bytes,
            summary=_safe_loads(r.summary_json, {}),
            created_at=r.created_at,
        )
        for r in rows
    ]


@router.get("/reports/{report_id}/download")
def download_generated_report(report_id: str, db: Session = Depends(get_db)) -> FileResponse:
    rep = db.query(GeneratedReport).filter(GeneratedReport.id == report_id).first()
    if not rep:
        raise HTTPException(status_code=404, detail="GeneratedReport not found.")
    abs_path = (settings.storage_path / rep.storage_rel_path).resolve()
    if not abs_path.exists():
        raise HTTPException(status_code=404, detail="Report file missing from disk.")
    media_type = "application/pdf" if rep.report_format == "pdf" else "application/json"
    return FileResponse(
        path=str(abs_path),
        media_type=media_type,
        filename=f"spectra_report_{rep.id[:8]}.{rep.report_format}",
    )
