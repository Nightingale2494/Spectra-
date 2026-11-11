export interface AudioAsset {
  id: string;
  original_filename: string;
  sanitized_filename: string;
  original_format: string;
  mime_type: string;
  codec: string;
  container_format: string;
  duration_seconds: number;
  sample_rate: number;
  channels: number;
  bit_depth: number | null;
  bitrate: number | null;
  file_size_bytes: number;
  sha256_checksum: string;
  is_processed_variant: boolean;
  parent_asset_id: string | null;
  variant_description: string | null;
  created_at: string;
}

export interface AnalysisJob {
  id: string;
  audio_asset_id: string;
  job_type: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  progress: number;
  current_stage: string;
  error_message: string | null;
  warnings: string[];
  started_at: string | null;
  completed_at: string | null;
  created_at: string;
}

export interface ProvenanceResult {
  id: string;
  audio_asset_id: string;
  c2pa_status:
    | "manifest_valid"
    | "manifest_invalid"
    | "manifest_absent"
    | "unsupported_format"
    | "inspection_error";
  credential_present: boolean;
  signature_valid: boolean | null;
  active_manifest_id: string | null;
  claim_generator: string | null;
  assertions: Array<{ label: string; data: Record<string, any> }>;
  ingredients: Array<Record<string, any>>;
  certificate_info: Record<string, any>;
  validation_errors: string[];
  validation_warnings: string[];
  raw_manifest: Record<string, any> | null;
  conventional_metadata: Record<string, any>;
  supported_fields: string[];
  unsupported_fields: string[];
  scientific_notice: string;
  created_at: string;
}

export interface DetectorResult {
  id: string;
  job_id: string;
  audio_asset_id: string;
  detector_id: string;
  detector_name: string;
  detector_version: string;
  model_source: string;
  model_revision: string;
  probability: number | null;
  raw_score: number | null;
  threshold: number;
  predicted_label: string;
  confidence: number | null;
  input_sample_rate: number;
  input_duration_sec: number;
  processing_time_ms: number;
  device_used: string;
  feature_dim: number | null;
  status: string;
  error_message: string | null;
  details: Record<string, any>;
  created_at: string;
}

export interface DetectorInfo {
  detector_id: string;
  name: string;
  version: string;
  model_source: string;
  model_revision: string;
  license: string;
  input_sample_rate: number;
  input_duration_requirements: string;
  supported_devices: string[];
  active_device: string;
  available: boolean;
  loaded: boolean;
  status_message: string;
  description: string;
}

export interface ProcessingExperiment {
  id: string;
  job_id: string;
  source_asset_id: string;
  output_asset_id: string | null;
  operation_type: string;
  preset_name: string | null;
  parameters: Record<string, any>;
  engine_used: string;
  original_sha256: string;
  processed_sha256: string | null;
  duration_preserved: boolean;
  channels_preserved: boolean;
  clipping_detected: boolean;
  nan_inf_detected: boolean;
  reconstruction_valid: boolean;
  processing_time_ms: number;
  audit_log: string[];
  warnings: string[];
  status: string;
  error_message: string | null;
  created_at: string;
}

export interface ComparisonResult {
  id: string;
  experiment_id: string;
  source_asset_id: string;
  processed_asset_id: string;
  delta_rms_db: number;
  delta_peak_db: number;
  spectral_distance_l2: number;
  spectral_convergence: number;
  snr_db: number | null;
  psnr_db: number | null;
  loudness_delta_db: number;
  duration_delta_sec: number;
  alignment_offset_samples: number;
  original_clipping_ratio: number;
  processed_clipping_ratio: number;
  detector_deltas: Array<{
    detector_id: string;
    detector_name: string;
    original_probability?: number | null;
    processed_probability?: number | null;
    before_probability: number | null;
    after_probability: number | null;
    delta_probability: number | null;
    original_label?: string;
    processed_label?: string;
    before_label: string;
    after_label: string;
    threshold: number;
  }>;
  spectrum_comparison: {
    frequencies_hz: number[];
    original_db?: number[];
    processed_db?: number[];
    original_power_db?: number[];
    processed_power_db?: number[];
    difference_db?: number[];
  };
  residual_comparison: {
    frequencies_hz: number[];
    original_residual?: number[];
    processed_residual?: number[];
    original_residual_norm?: number[];
    processed_residual_norm?: number[];
    original_peak_ratio?: number;
    processed_peak_ratio?: number;
    original_comb_periodicity?: number;
    processed_comb_periodicity?: number;
    original_periodicity_score?: number;
    processed_periodicity_score?: number;
    temporal_comparison?: TemporalComparisonData;
  };
  observations: string[];
  warnings: string[];
  created_at: string;
}

export interface TemporalWindowStep {
  window_index: number;
  start_sec: number;
  end_sec: number;
  center_sec: number;
  unpadded_duration_sec: number;
  zero_pad_sec: number;
  is_incomplete_final_window: boolean;
  sonics_gamma_probability: number | null;
  sonics_gamma_logit: number | null;
  fakeprint_exploratory_probability: number | null;
  fakeprint_calibrated_probability: number | null;
  fakeprint_periodicity_score: number | null;
  cross_generator_probability?: number | null;
  rms_dbfs: number;
  crest_factor_db: number;
  clipping_ratio: number;
  abstained: boolean;
  ood_detected: boolean;
  region_label:
    | "Elevated AI-associated features"
    | "Low AI-associated features"
    | "Model disagreement"
    | "Insufficient evidence";
  disagreement: boolean;
  disagreement_spread: number | null;
  rationale: string;
  warnings: string[];
}

export interface TemporalRegionSummary {
  region_id: string;
  region_index: number;
  start_sec: number;
  end_sec: number;
  duration_sec: number;
  window_count: number;
  region_label:
    | "Elevated AI-associated features"
    | "Low AI-associated features"
    | "Model disagreement"
    | "Insufficient evidence";
  mean_sonics_probability: number | null;
  max_sonics_probability: number | null;
  min_sonics_probability: number | null;
  mean_fakeprint_exploratory_probability: number | null;
  mean_rms_dbfs: number | null;
  disagreement_present: boolean;
  abstained: boolean;
  scientific_interpretation: string;
  warnings: string[];
}

export interface TemporalSectionTransition {
  transition_index: number;
  timestamp_sec: number;
  previous_window: [number, number];
  next_window: [number, number];
  from_label: string;
  to_label: string;
  sonics_probability_before: number;
  sonics_probability_after: number;
  probability_delta: number;
  description: string;
}

export interface TemporalAnalysisData {
  schema_version: string;
  asset_sha256: string;
  recording_duration_sec: number;
  processing_time_ms: number;
  cache_hit: boolean;
  configuration: {
    sonics_gamma_window_sec: number;
    hop_sec: number;
    fakeprint_segment_window_sec: number;
    include_exploratory_fakeprint: boolean;
    sonics_alpha_context_sec: number;
    incomplete_final_window_policy: string;
    per_window_normalization: string;
  };
  detector_revisions: Record<string, string>;
  summary_statistics: {
    sonics_gamma_5s: {
      valid_window_count: number;
      total_window_count: number;
      mean_probability: number | null;
      median_probability: number | null;
      max_probability: number | null;
      min_probability: number | null;
      std_probability: number | null;
      iqr_probability: number | null;
      elevated_window_fraction: number;
    };
    fakeprint_exploratory_mean_prob: number | null;
    disagreement_window_count: number;
    ood_or_abstained_window_count: number;
    region_count: number;
    transition_count: number;
  };
  timeline_steps: TemporalWindowStep[];
  per_detector_windows: {
    sonics_gamma_5s: Record<string, any>[];
    fakeprint_onnx_exploratory: Record<string, any>[];
    cross_generator_detector: Record<string, any>[];
  };
  long_context_predictions: {
    detector_id: string;
    detector_name?: string;
    model_version?: string;
    model_revision?: string;
    available: boolean;
    status?: string;
    abstained: boolean;
    predicted_label?: string;
    aggregate_probability?: number | null;
    aggregate_raw_score?: number | null;
    context_window_sec: number;
    minimum_required_duration_sec: number;
    recording_duration_sec: number;
    representation_notice?: string;
    status_notice?: string;
    ood_warning?: string | null;
    long_context_windows: Array<{
      window_label: string;
      start_sec: number;
      end_sec: number;
      context_span_sec: number;
      zero_pad_sec: number;
      probability: number | null;
      raw_score?: number | null;
      abstained: boolean;
      warning: string | null;
    }>;
  };
  fakeprint_segment_validation: Record<string, any>;
  region_summaries: TemporalRegionSummary[];
  section_transitions: TemporalSectionTransition[];
  hybrid_forensic_analysis: {
    known_provenance_facts: Record<string, any>;
    temporal_heterogeneity_metrics: {
      inter_window_std: number;
      section_transition_count: number;
      elevated_windows: number;
      low_windows: number;
      disagreement_windows: number;
      insufficient_evidence_windows: number;
      contains_mixed_feature_regimes: boolean;
    };
    forensic_interpretation: string;
  };
  scientific_notices: string[];
}

export interface TemporalComparisonSegment {
  window_index: number;
  start_sec: number;
  end_sec: number;
  center_sec: number;
  original_sonics_probability: number | null;
  processed_sonics_probability: number | null;
  delta_sonics_probability: number | null;
  original_fakeprint_probability: number | null;
  processed_fakeprint_probability: number | null;
  delta_fakeprint_probability: number | null;
  original_periodicity_score: number | null;
  processed_periodicity_score: number | null;
  delta_periodicity_score: number | null;
  original_region_label: string;
  processed_region_label: string;
  segment_snr_db: number;
  segment_lsd_db: number;
  segment_rms_delta_db: number;
  abstained: boolean;
  quality_warnings: string[];
}

export interface TemporalComparisonData {
  schema_version: string;
  aligned_duration_sec: number;
  alignment_offset_samples: number;
  alignment_offset_ms: number;
  hop_sec: number;
  window_sec: number;
  segment_count: number;
  processing_time_ms: number;
  summary: {
    mean_delta_sonics_probability: number | null;
    max_abs_delta_sonics_probability: number | null;
    mean_delta_fakeprint_probability: number | null;
    mean_segment_snr_db: number | null;
    mean_segment_lsd_db: number | null;
    quality_warning_count: number;
  };
  segments: TemporalComparisonSegment[];
  quality_warnings: string[];
  scientific_disclaimer: string;
}

export interface AnalysisResults {
  job: AnalysisJob;
  asset: AudioAsset;
  provenance: ProvenanceResult | null;
  forensics: {
    time_domain?: Record<string, any>;
    frequency_domain?: Record<string, any>;
    fakeprint?: Record<string, any>;
    observations?: string[];
  };
  detectors: DetectorResult[];
  experiments: ProcessingExperiment[];
  comparisons: ComparisonResult[];
  software_versions: Record<string, string>;
}

export interface WaveformData {
  asset_id: string;
  sample_rate: number;
  duration_seconds: number;
  channels: number;
  num_points: number;
  times: number[];
  peaks_min: number[];
  peaks_max: number[];
  rms_envelope: number[];
}

export interface SpectrogramData {
  asset_id: string;
  sample_rate: number;
  n_fft: number;
  hop_length: number;
  times: number[];
  frequencies: number[];
  magnitude_db: number[][];
  mean_spectrum_freqs: number[];
  mean_spectrum_db: number[];
  fakeprint_freqs: number[];
  fakeprint_spectrum_db: number[];
  fakeprint_hull_db: number[];
  fakeprint_residual_norm: number[];
}

export interface HealthStatus {
  status: string;
  version: string;
  environment: string;
  database_ok: boolean;
  storage_writable: boolean;
  ffmpeg_available: boolean;
  ffmpeg_path: string | null;
  pyav_available: boolean;
  c2pa_sdk_version: string;
  cpu_count: number;
  cuda_available: boolean;
  gpu_info: string | null;
  detectors_ready: number;
  detectors_total: number;
  comfyui_reachable: boolean;
}

export interface DspPreset {
  id: string;
  name: string;
  operation_type: string;
  engine: string;
  description: string;
  parameters: Record<string, any>;
}

export interface GeneratedReport {
  id: string;
  job_id: string;
  experiment_id: string | null;
  audio_asset_id: string;
  report_format: string;
  title: string;
  sha256_checksum: string;
  file_size_bytes: number;
  summary: Record<string, any>;
  created_at: string;
}
