"""Typed Pydantic v2 schemas for SPECTRA API."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class AudioAssetOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    original_filename: str
    sanitized_filename: str
    original_format: str
    mime_type: str
    codec: str
    container_format: str
    duration_seconds: float
    sample_rate: int
    channels: int
    bit_depth: Optional[int] = None
    bitrate: Optional[int] = None
    file_size_bytes: int
    sha256_checksum: str
    is_processed_variant: bool = False
    parent_asset_id: Optional[str] = None
    variant_description: Optional[str] = None
    created_at: datetime


class UploadResponse(BaseModel):
    asset: AudioAssetOut
    initial_job_id: str
    message: str


class WaveformOut(BaseModel):
    asset_id: str
    sample_rate: int
    duration_seconds: float
    channels: int
    num_points: int
    times: List[float]
    peaks_min: List[float]
    peaks_max: List[float]
    rms_envelope: List[float]


class SpectrogramOut(BaseModel):
    asset_id: str
    sample_rate: int
    n_fft: int
    hop_length: int
    times: List[float]
    frequencies: List[float]
    magnitude_db: List[List[float]]
    mean_spectrum_freqs: List[float]
    mean_spectrum_db: List[float]
    fakeprint_freqs: List[float]
    fakeprint_spectrum_db: List[float]
    fakeprint_hull_db: List[float]
    fakeprint_residual_norm: List[float]


class ProvenanceResultOut(BaseModel):
    id: str
    audio_asset_id: str
    c2pa_status: str
    credential_present: bool
    signature_valid: Optional[bool] = None
    active_manifest_id: Optional[str] = None
    claim_generator: Optional[str] = None
    assertions: List[Dict[str, Any]] = Field(default_factory=list)
    ingredients: List[Dict[str, Any]] = Field(default_factory=list)
    certificate_info: Dict[str, Any] = Field(default_factory=dict)
    validation_errors: List[str] = Field(default_factory=list)
    validation_warnings: List[str] = Field(default_factory=list)
    raw_manifest: Optional[Dict[str, Any]] = None
    conventional_metadata: Dict[str, Any] = Field(default_factory=dict)
    supported_fields: List[str] = Field(default_factory=list)
    unsupported_fields: List[str] = Field(default_factory=list)
    scientific_notice: str = (
        "C2PA Content Credentials verify cryptographic provenance of signed manifests. "
        "Absence of a C2PA manifest does not prove human or AI origin."
    )
    created_at: datetime


class DetectorResultOut(BaseModel):
    id: str
    job_id: str
    audio_asset_id: str
    detector_id: str
    detector_name: str
    detector_version: str
    model_source: str
    model_revision: str
    probability: Optional[float] = None
    raw_score: Optional[float] = None
    threshold: float = 0.5
    predicted_label: str
    confidence: Optional[float] = None
    input_sample_rate: int
    input_duration_sec: float
    processing_time_ms: float
    device_used: str
    feature_dim: Optional[int] = None
    status: str
    error_message: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class DetectorInfoOut(BaseModel):
    detector_id: str
    name: str
    version: str
    model_source: str
    model_revision: str
    license: str
    input_sample_rate: int
    input_duration_requirements: str
    supported_devices: List[str]
    active_device: str
    available: bool
    loaded: bool
    status_message: str
    description: str


class AnalysisJobOut(BaseModel):
    id: str
    audio_asset_id: str
    job_type: str
    status: str
    progress: float
    current_stage: str
    error_message: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime


class AnalysisResultsOut(BaseModel):
    job: AnalysisJobOut
    asset: AudioAssetOut
    provenance: Optional[ProvenanceResultOut] = None
    forensics: Dict[str, Any] = Field(default_factory=dict)
    detectors: List[DetectorResultOut] = Field(default_factory=list)
    experiments: List[ProcessingExperimentOut] = Field(default_factory=list)
    comparisons: List[ComparisonResultOut] = Field(default_factory=list)
    software_versions: Dict[str, str] = Field(default_factory=dict)


class ProcessRequest(BaseModel):
    operation_type: str = Field(
        ...,
        description=(
            "One of: highpass, lowpass, bandpass, parametric_eq, eq_3band, "
            "notch_comb, spectral_smoothing, noise_reduction, dynamic_compression, "
            "gain_limiter, resample, codec_simulation, pedalboard_chain, "
            "comfyui_eq3band, comfyui_vae_roundtrip"
        ),
    )
    preset_name: Optional[str] = None
    engine: str = "scipy_dsp"
    parameters: Dict[str, Any] = Field(default_factory=dict)
    export_format: str = "wav"
    run_detectors_on_output: bool = True


class MetadataExportRequest(BaseModel):
    strip_conventional_metadata: bool = True
    strip_c2pa_manifest: bool = False
    custom_tags: Dict[str, str] = Field(default_factory=dict)


class ProcessingExperimentOut(BaseModel):
    id: str
    job_id: str
    source_asset_id: str
    output_asset_id: Optional[str] = None
    operation_type: str
    preset_name: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    engine_used: str
    original_sha256: str
    processed_sha256: Optional[str] = None
    duration_preserved: bool
    channels_preserved: bool
    clipping_detected: bool
    nan_inf_detected: bool
    reconstruction_valid: bool
    processing_time_ms: float
    audit_log: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    status: str
    error_message: Optional[str] = None
    created_at: datetime


class ComparisonResultOut(BaseModel):
    id: str
    experiment_id: str
    source_asset_id: str
    processed_asset_id: str
    delta_rms_db: float
    delta_peak_db: float
    spectral_distance_l2: float
    spectral_convergence: float
    snr_db: Optional[float] = None
    psnr_db: Optional[float] = None
    loudness_delta_db: float
    duration_delta_sec: float
    alignment_offset_samples: int
    original_clipping_ratio: float
    processed_clipping_ratio: float
    detector_deltas: List[Dict[str, Any]] = Field(default_factory=list)
    spectrum_comparison: Dict[str, Any] = Field(default_factory=dict)
    residual_comparison: Dict[str, Any] = Field(default_factory=dict)
    observations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    created_at: datetime


class ReportGenerateRequest(BaseModel):
    report_format: str = "json"
    experiment_id: Optional[str] = None
    include_raw_manifest: bool = True


class TemporalAnalysisRequest(BaseModel):
    hop_sec: float = Field(default=2.5, ge=0.5, le=10.0)
    fakeprint_window_sec: float = Field(default=5.0, ge=2.5, le=30.0)
    include_exploratory_fakeprint: bool = True
    force_recompute: bool = False


class GeneratedReportOut(BaseModel):
    id: str
    job_id: str
    experiment_id: Optional[str] = None
    audio_asset_id: str
    report_format: str
    title: str
    sha256_checksum: str
    file_size_bytes: int
    summary: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ComfyUIStatusOut(BaseModel):
    enabled: bool
    service_reachable: bool
    api_url: str
    install_path: str
    desktop_version: Optional[str] = None
    core_version: Optional[str] = None
    python_version: Optional[str] = None
    torch_version: Optional[str] = None
    gpu_name: Optional[str] = None
    vram_mb: Optional[int] = None
    installed_audio_models: List[Dict[str, Any]] = Field(default_factory=list)
    available_audio_nodes: List[str] = Field(default_factory=list)
    supported_workflows: List[Dict[str, Any]] = Field(default_factory=list)
    integration_notes: List[str] = Field(default_factory=list)


class HealthOut(BaseModel):
    status: str
    version: str
    environment: str
    database_ok: bool
    storage_writable: bool
    ffmpeg_available: bool
    ffmpeg_path: Optional[str] = None
    pyav_available: bool
    c2pa_sdk_version: Optional[str] = None
    cpu_count: int
    cuda_available: bool
    gpu_info: Optional[str] = None
    detectors_ready: int
    detectors_total: int
    comfyui_reachable: bool
