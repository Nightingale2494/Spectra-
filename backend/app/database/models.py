"""SQLAlchemy ORM models for SPECTRA."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)

from .session import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_uuid() -> str:
    return str(uuid.uuid4())


class AudioAsset(Base):
    __tablename__ = "audio_assets"

    id = Column(String(36), primary_key=True, default=new_uuid)
    original_filename = Column(String(255), nullable=False)
    sanitized_filename = Column(String(255), nullable=False)
    storage_rel_path = Column(String(512), nullable=False)
    original_format = Column(String(32), nullable=False)
    mime_type = Column(String(64), nullable=False)
    codec = Column(String(64), nullable=False, default="unknown")
    container_format = Column(String(64), nullable=False, default="unknown")
    duration_seconds = Column(Float, nullable=False, default=0.0)
    sample_rate = Column(Integer, nullable=False, default=0)
    channels = Column(Integer, nullable=False, default=0)
    bit_depth = Column(Integer, nullable=True)
    bitrate = Column(Integer, nullable=True)
    file_size_bytes = Column(Integer, nullable=False, default=0)
    sha256_checksum = Column(String(64), nullable=False, index=True)
    is_processed_variant = Column(Boolean, nullable=False, default=False)
    parent_asset_id = Column(String(36), ForeignKey("audio_assets.id"), nullable=True)
    variant_description = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    expires_at = Column(DateTime(timezone=True), nullable=True)


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    id = Column(String(36), primary_key=True, default=new_uuid)
    audio_asset_id = Column(String(36), ForeignKey("audio_assets.id"), nullable=False, index=True)
    job_type = Column(String(64), nullable=False, default="full_analysis")
    status = Column(String(32), nullable=False, default="queued", index=True)
    progress = Column(Float, nullable=False, default=0.0)
    current_stage = Column(String(128), nullable=False, default="queued")
    parameters_json = Column(Text, nullable=True)
    forensics_json = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    warnings_json = Column(Text, nullable=True)
    software_versions_json = Column(Text, nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


class DetectorRun(Base):
    __tablename__ = "detector_runs"

    id = Column(String(36), primary_key=True, default=new_uuid)
    job_id = Column(String(36), ForeignKey("analysis_jobs.id"), nullable=False, index=True)
    audio_asset_id = Column(String(36), ForeignKey("audio_assets.id"), nullable=False, index=True)
    detector_id = Column(String(64), nullable=False, index=True)
    detector_name = Column(String(128), nullable=False)
    detector_version = Column(String(64), nullable=False)
    model_source = Column(String(255), nullable=False)
    model_revision = Column(String(128), nullable=False)
    probability = Column(Float, nullable=True)
    raw_score = Column(Float, nullable=True)
    threshold = Column(Float, nullable=False, default=0.5)
    predicted_label = Column(String(64), nullable=False, default="unknown")
    confidence = Column(Float, nullable=True)
    input_sample_rate = Column(Integer, nullable=False, default=16000)
    input_duration_sec = Column(Float, nullable=False, default=0.0)
    processing_time_ms = Column(Float, nullable=False, default=0.0)
    device_used = Column(String(32), nullable=False, default="cpu")
    feature_dim = Column(Integer, nullable=True)
    status = Column(String(32), nullable=False, default="success")
    error_message = Column(Text, nullable=True)
    details_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


class ProvenanceResult(Base):
    __tablename__ = "provenance_results"

    id = Column(String(36), primary_key=True, default=new_uuid)
    job_id = Column(String(36), ForeignKey("analysis_jobs.id"), nullable=True, index=True)
    audio_asset_id = Column(String(36), ForeignKey("audio_assets.id"), nullable=False, index=True)
    c2pa_status = Column(String(64), nullable=False)
    credential_present = Column(Boolean, nullable=False, default=False)
    signature_valid = Column(Boolean, nullable=True)
    active_manifest_id = Column(String(255), nullable=True)
    claim_generator = Column(String(255), nullable=True)
    assertions_json = Column(Text, nullable=True)
    ingredients_json = Column(Text, nullable=True)
    certificate_info_json = Column(Text, nullable=True)
    validation_errors_json = Column(Text, nullable=True)
    validation_warnings_json = Column(Text, nullable=True)
    raw_manifest_json = Column(Text, nullable=True)
    conventional_metadata_json = Column(Text, nullable=True)
    supported_fields_json = Column(Text, nullable=True)
    unsupported_fields_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


class ProcessingExperiment(Base):
    __tablename__ = "processing_experiments"

    id = Column(String(36), primary_key=True, default=new_uuid)
    job_id = Column(String(36), ForeignKey("analysis_jobs.id"), nullable=False, index=True)
    source_asset_id = Column(String(36), ForeignKey("audio_assets.id"), nullable=False, index=True)
    output_asset_id = Column(String(36), ForeignKey("audio_assets.id"), nullable=True, index=True)
    operation_type = Column(String(64), nullable=False)
    preset_name = Column(String(128), nullable=True)
    parameters_json = Column(Text, nullable=False)
    engine_used = Column(String(64), nullable=False, default="scipy_dsp")
    original_sha256 = Column(String(64), nullable=False)
    processed_sha256 = Column(String(64), nullable=True)
    duration_preserved = Column(Boolean, nullable=False, default=True)
    channels_preserved = Column(Boolean, nullable=False, default=True)
    clipping_detected = Column(Boolean, nullable=False, default=False)
    nan_inf_detected = Column(Boolean, nullable=False, default=False)
    reconstruction_valid = Column(Boolean, nullable=False, default=True)
    processing_time_ms = Column(Float, nullable=False, default=0.0)
    audit_log_json = Column(Text, nullable=True)
    warnings_json = Column(Text, nullable=True)
    status = Column(String(32), nullable=False, default="completed")
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


class ComparisonResult(Base):
    __tablename__ = "comparison_results"

    id = Column(String(36), primary_key=True, default=new_uuid)
    experiment_id = Column(String(36), ForeignKey("processing_experiments.id"), nullable=False, index=True)
    source_asset_id = Column(String(36), ForeignKey("audio_assets.id"), nullable=False)
    processed_asset_id = Column(String(36), ForeignKey("audio_assets.id"), nullable=False)
    delta_rms_db = Column(Float, nullable=False, default=0.0)
    delta_peak_db = Column(Float, nullable=False, default=0.0)
    spectral_distance_l2 = Column(Float, nullable=False, default=0.0)
    spectral_convergence = Column(Float, nullable=False, default=0.0)
    snr_db = Column(Float, nullable=True)
    psnr_db = Column(Float, nullable=True)
    loudness_delta_db = Column(Float, nullable=False, default=0.0)
    duration_delta_sec = Column(Float, nullable=False, default=0.0)
    alignment_offset_samples = Column(Integer, nullable=False, default=0)
    original_clipping_ratio = Column(Float, nullable=False, default=0.0)
    processed_clipping_ratio = Column(Float, nullable=False, default=0.0)
    detector_deltas_json = Column(Text, nullable=True)
    spectrum_comparison_json = Column(Text, nullable=True)
    residual_comparison_json = Column(Text, nullable=True)
    observations_json = Column(Text, nullable=True)
    warnings_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


class GeneratedReport(Base):
    __tablename__ = "generated_reports"

    id = Column(String(36), primary_key=True, default=new_uuid)
    job_id = Column(String(36), ForeignKey("analysis_jobs.id"), nullable=False, index=True)
    experiment_id = Column(String(36), ForeignKey("processing_experiments.id"), nullable=True)
    audio_asset_id = Column(String(36), ForeignKey("audio_assets.id"), nullable=False)
    report_format = Column(String(16), nullable=False)
    title = Column(String(255), nullable=False)
    storage_rel_path = Column(String(512), nullable=False)
    sha256_checksum = Column(String(64), nullable=False)
    file_size_bytes = Column(Integer, nullable=False, default=0)
    summary_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
