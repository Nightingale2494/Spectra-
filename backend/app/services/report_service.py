"""Technical Forensic Report Generator (JSON & PDF) for SPECTRA."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import uuid
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sqlalchemy.orm import Session

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
from app.services.ingestion_service import decode_audio_pcm, resolve_asset_path
from app.services.temporal_service import compute_temporal_forensic_analysis


def _safe_json_loads(raw: Optional[str], default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default


def build_report_payload(
    db: Session,
    job: AnalysisJob,
    asset: AudioAsset,
    experiment_id: Optional[str] = None,
    include_raw_manifest: bool = True,
) -> Dict[str, Any]:
    """Assemble a complete, reproducible forensic report dictionary."""
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
    experiments_q = db.query(ProcessingExperiment).filter(ProcessingExperiment.source_asset_id == asset.id)
    if experiment_id:
        experiments_q = experiments_q.filter(ProcessingExperiment.id == experiment_id)
    experiments = experiments_q.order_by(ProcessingExperiment.created_at.desc()).all()

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

    forensics = _safe_json_loads(job.forensics_json, {})
    sw_versions = _safe_json_loads(job.software_versions_json, {})

    prov_dict = None
    if prov:
        prov_dict = {
            "c2pa_status": prov.c2pa_status,
            "credential_present": prov.credential_present,
            "signature_valid": prov.signature_valid,
            "active_manifest_id": prov.active_manifest_id,
            "claim_generator": prov.claim_generator,
            "assertions": _safe_json_loads(prov.assertions_json, []),
            "ingredients": _safe_json_loads(prov.ingredients_json, []),
            "certificate_info": _safe_json_loads(prov.certificate_info_json, {}),
            "validation_errors": _safe_json_loads(prov.validation_errors_json, []),
            "validation_warnings": _safe_json_loads(prov.validation_warnings_json, []),
            "conventional_metadata": _safe_json_loads(prov.conventional_metadata_json, {}),
            "supported_fields": _safe_json_loads(prov.supported_fields_json, []),
            "unsupported_fields": _safe_json_loads(prov.unsupported_fields_json, []),
        }
        if include_raw_manifest:
            prov_dict["raw_manifest"] = _safe_json_loads(prov.raw_manifest_json, None)

    detector_list = []
    for d in detectors:
        detector_list.append({
            "detector_id": d.detector_id,
            "detector_name": d.detector_name,
            "detector_version": d.detector_version,
            "model_source": d.model_source,
            "model_revision": d.model_revision,
            "status": d.status,
            "probability": d.probability,
            "raw_score": d.raw_score,
            "threshold": d.threshold,
            "predicted_label": d.predicted_label,
            "confidence": d.confidence,
            "input_sample_rate": d.input_sample_rate,
            "input_duration_sec": d.input_duration_sec,
            "processing_time_ms": d.processing_time_ms,
            "device_used": d.device_used,
            "feature_dim": d.feature_dim,
            "error_message": d.error_message,
            "details": _safe_json_loads(d.details_json, {}),
        })

    exp_list = []
    for e in experiments:
        exp_list.append({
            "id": e.id,
            "operation_type": e.operation_type,
            "preset_name": e.preset_name,
            "engine_used": e.engine_used,
            "parameters": _safe_json_loads(e.parameters_json, {}),
            "original_sha256": e.original_sha256,
            "processed_sha256": e.processed_sha256,
            "output_asset_id": e.output_asset_id,
            "duration_preserved": e.duration_preserved,
            "channels_preserved": e.channels_preserved,
            "clipping_detected": e.clipping_detected,
            "nan_inf_detected": e.nan_inf_detected,
            "reconstruction_valid": e.reconstruction_valid,
            "processing_time_ms": e.processing_time_ms,
            "audit_log": _safe_json_loads(e.audit_log_json, []),
            "warnings": _safe_json_loads(e.warnings_json, []),
            "created_at": e.created_at.isoformat() if e.created_at else None,
        })

    comp_list = []
    for c in comparisons:
        res_comp = _safe_json_loads(c.residual_comparison_json, {})
        comp_list.append({
            "id": c.id,
            "experiment_id": c.experiment_id,
            "source_asset_id": c.source_asset_id,
            "processed_asset_id": c.processed_asset_id,
            "delta_rms_db": c.delta_rms_db,
            "delta_peak_db": c.delta_peak_db,
            "spectral_distance_l2": c.spectral_distance_l2,
            "spectral_convergence": c.spectral_convergence,
            "snr_db": c.snr_db,
            "psnr_db": c.psnr_db,
            "loudness_delta_db": c.loudness_delta_db,
            "duration_delta_sec": c.duration_delta_sec,
            "alignment_offset_samples": c.alignment_offset_samples,
            "original_clipping_ratio": c.original_clipping_ratio,
            "processed_clipping_ratio": c.processed_clipping_ratio,
            "detector_deltas": _safe_json_loads(c.detector_deltas_json, []),
            "temporal_comparison": res_comp.get("temporal_comparison"),
            "observations": _safe_json_loads(c.observations_json, []),
            "warnings": _safe_json_loads(c.warnings_json, []),
        })

    # Load or compute Phase 5 temporal forensic analysis (uses SHA-256 deterministic cache)
    temporal_analysis: Optional[Dict[str, Any]] = None
    try:
        file_path = resolve_asset_path(asset)
        samples, sr = decode_audio_pcm(file_path, target_sr=None, mono=False)
        prov_ctx = {
            "c2pa_status": prov.c2pa_status if prov else "no_c2pa_found",
            "c2pa_verified": bool(prov and prov.credential_present and prov.signature_valid),
            "claim_generator": prov.claim_generator if prov else None,
        }
        temporal_analysis = compute_temporal_forensic_analysis(
            samples=samples,
            sr=sr,
            asset_sha256=asset.sha256_checksum,
            hop_sec=2.5,
            fakeprint_window_sec=5.0,
            include_exploratory_fakeprint=True,
            provenance_context=prov_ctx,
            force_recompute=False,
            job_id=job.id,
        )
    except Exception:
        temporal_analysis = (forensics or {}).get("temporal_summary")

    # Collect model & checkpoint revisions and input validity warnings across detectors
    model_revisions = []
    input_validity_warnings = list(_safe_json_loads(job.warnings_json, []))
    for d_item in detector_list:
        det_details = d_item.get("details") or {}
        model_revisions.append({
            "detector_id": d_item["detector_id"],
            "detector_name": d_item["detector_name"],
            "detector_version": d_item["detector_version"],
            "model_source": d_item["model_source"],
            "model_revision": d_item["model_revision"],
            "weights_sha256": det_details.get("weights_sha256") or det_details.get("model_sha256"),
            "input_sample_rate_hz": d_item["input_sample_rate"],
            "normalization": det_details.get("normalization", "spectral_hull_min_filter"),
            "padding_mode": det_details.get("padding_mode", "none"),
            "crop_policy": det_details.get("crop_policy", "full_track_300s"),
        })
        for w_msg in det_details.get("input_validity_warnings", []):
            if w_msg not in input_validity_warnings:
                input_validity_warnings.append(w_msg)
        if det_details.get("ood_warning") and det_details["ood_warning"] not in input_validity_warnings:
            input_validity_warnings.append(det_details["ood_warning"])

    # Separate conventional metadata into encoder tags vs content/origin tags (Task 3)
    conv_tags = ((prov_dict or {}).get("conventional_metadata") or {}).get("tags", {}) if prov_dict else {}
    encoder_tags = {
        k: v for k, v in conv_tags.items()
        if any(tok in k.upper() for tok in ("TSSE", "TENC", "ENCODER", "ENCODING", "SOFTWARE", "COMMENT"))
    }
    content_assertion_tags = {k: v for k, v in conv_tags.items() if k not in encoder_tags}

    # Build 5-Tier Epistemic Forensic Attribution Hierarchy (Task 3)
    td_metrics = (forensics or {}).get("time_domain", {})
    fd_metrics = (forensics or {}).get("frequency_domain", {})
    fp_metrics = (forensics or {}).get("fakeprint", {})

    active_non_abstained = [
        d_item for d_item in detector_list
        if d_item["status"] == "success"
        and d_item["probability"] is not None
        and not bool((d_item.get("details") or {}).get("abstained", False))
    ]
    research_hypotheses: List[str] = []
    if prov_dict and prov_dict.get("c2pa_status") == "manifest_valid" and prov_dict.get("signature_valid"):
        research_hypotheses.append(
            f"Cryptographic Provenance Hypothesis: Asset contains a valid, tamper-evident C2PA manifest signed by "
            f"'{prov_dict.get('claim_generator') or 'verified signer'}'. This provides cryptographic proof of the "
            "signed asset's provenance chain unless a signing key compromise occurred."
        )
    if len(active_non_abstained) >= 2:
        pos_dets = [d_item["detector_id"] for d_item in active_non_abstained if d_item["probability"] >= d_item["threshold"]]
        neg_dets = [d_item["detector_id"] for d_item in active_non_abstained if d_item["probability"] < d_item["threshold"]]
        if pos_dets and neg_dets:
            research_hypotheses.append(
                f"Detector Disagreement Hypothesis: Positive classification on {pos_dets} vs negative on {neg_dets}. "
                "This divergence can occur on (a) newer generator versions (e.g., Suno v4.5) that retain 1–8 kHz "
                "spectral comb peaks while evading 2024 SONICS spectro-temporal tokens, (b) hybrid DAW-remixed tracks, "
                "or (c) distorted/clipped human recordings."
            )
        elif pos_dets:
            research_hypotheses.append(
                f"Convergent Model Hypothesis: Multiple independent detectors ({pos_dets}) scored above threshold. "
                "Consistent with neural vocoder / synthetic song patterns represented in training distributions, "
                "though statistical agreement does not replace cryptographic provenance."
            )
        else:
            research_hypotheses.append(
                "Low-Signature Hypothesis: All active non-abstained detectors scored below decision thresholds. "
                "Note that continuous latent diffusion generators (e.g., Stable Audio, ACE-Step) frequently evade "
                "both 1–8 kHz spectral comb detectors and 2024 Suno/Udio-trained spectro-temporal classifiers."
            )

    forensic_attribution_hierarchy = {
        "tier_1_cryptographically_verified_c2pa_facts": {
            "epistemic_status": "Cryptographically Verified Fact (when signature_valid=True)",
            "c2pa_status": (prov_dict or {}).get("c2pa_status", "not_inspected"),
            "credential_present": bool((prov_dict or {}).get("credential_present", False)),
            "signature_valid": (prov_dict or {}).get("signature_valid"),
            "claim_generator": (prov_dict or {}).get("claim_generator"),
            "active_manifest_id": (prov_dict or {}).get("active_manifest_id"),
            "strength": (
                "Cryptographic JUMBF hash binding and X.509 signature verification prove the exact byte stream "
                "was signed by the stated claim generator without post-signing modification."
            ),
            "limitation": (
                "C2PA manifests are stripped by standard WAV/MP3 re-encoding or DAW export; absence of a C2PA "
                "manifest is neutral and never proves human origin."
            ),
        },
        "tier_2_conventional_metadata_assertions": {
            "epistemic_status": "Unverified Container Metadata Assertion (Mutable)",
            "encoder_software_tags": encoder_tags,
            "content_and_origin_tags": content_assertion_tags,
            "strength": "Preserves workflow context, export software (e.g. Lavf, LAME, FL Studio), or source URLs when intact.",
            "limitation": (
                "Conventional ID3/Vorbis/RIFF tags are unsigned and trivially editable. Encoder metadata (TSSE/TENC) "
                "identifies only the last muxing/encoding tool—never the original generation source."
            ),
        },
        "tier_3_audio_signal_observations": {
            "epistemic_status": "Objective Mathematical Signal Measurement",
            "measurements": {
                "peak_dbfs": td_metrics.get("peak_dbfs"),
                "rms_dbfs": td_metrics.get("rms_dbfs"),
                "crest_factor_db": td_metrics.get("crest_factor_db"),
                "clipping_ratio": td_metrics.get("clipping_ratio"),
                "hf_energy_ratio_above_12khz": fd_metrics.get("hf_energy_ratio_above_12khz"),
                "fakeprint_1_8khz_periodicity_score": fp_metrics.get("periodicity_score"),
                "fakeprint_200hz_comb_contrast": fp_metrics.get("comb_grid_200hz_contrast"),
                "fakeprint_temporal_stationarity": fp_metrics.get("temporal_stationarity"),
            },
            "observations": (forensics or {}).get("observations", []),
            "strength": "Deterministic, reproducible time-domain and STFT measurements of the physical waveform.",
            "limitation": (
                "A periodic spectral comb pattern in 1–8 kHz is consistent with periodic upsampling structures "
                "described in open-weights literature, but cannot prove an undocumented proprietary Suno/Udio "
                "neural architecture and can be confounded by lossy codecs, clipping, or harmonic distortion."
            ),
        },
        "tier_4_model_predictions": {
            "epistemic_status": "Statistical Classifier Prediction (Domain-Bounded)",
            "detectors": [
                {
                    "detector_id": d_item["detector_id"],
                    "raw_score": d_item["raw_score"],
                    "probability": d_item["probability"],
                    "calibrated_probability": (d_item.get("details") or {}).get("calibrated_probability", d_item["probability"]),
                    "threshold": d_item["threshold"],
                    "predicted_label": d_item["predicted_label"],
                    "abstained": bool((d_item.get("details") or {}).get("abstained", False)),
                }
                for d_item in detector_list
            ],
            "ensemble_policy": (
                "Independent detector reporting (No forced linear ensemble). Empirical evaluation shows that a "
                "linear blend (0.45*Fakeprint + 0.55*SONICS) degrades test F1/AUROC relative to Fakeprint ONNX "
                "alone when SONICS misses newer Suno v4.5 or Udio tracks, while failing to rescue diffusion tracks."
            ),
            "strength": "Quantifies similarity to learned spectral-hull or spectro-temporal training distributions.",
            "limitation": (
                "Pretrained detectors suffer domain shift on newer generator versions (Suno v4.5) and fail to detect "
                "continuous latent diffusion VAEs (Stable Audio, ACE-Step)."
            ),
        },
        "tier_5_research_hypotheses": {
            "epistemic_status": "Non-Definitive Analytical Hypothesis",
            "hypotheses": research_hypotheses,
            "strength": "Synthesizes cross-tier agreements and conflicts for human forensic review.",
            "limitation": "Must be treated as a testable working hypothesis, never as standalone forensic proof.",
        },
    }

    repro_seed_str = f"{job.id}:{asset.sha256_checksum}:" + ",".join(
        f"{m['detector_id']}={m['model_revision']}" for m in model_revisions
    )
    reproducible_experiment_id = hashlib.sha256(repro_seed_str.encode("utf-8")).hexdigest()[:24]

    # Extract selected/notable time windows from temporal_analysis for Priority 7 auditability
    t_steps = (temporal_analysis or {}).get("timeline_steps", [])
    selected_time_windows = {
        "elevated_score_windows": [
            {
                "window_index": s["window_index"],
                "start_sec": s["start_sec"],
                "end_sec": s["end_sec"],
                "sonics_gamma_probability": s.get("sonics_gamma_probability"),
                "sonics_gamma_logit": s.get("sonics_gamma_logit"),
                "fakeprint_exploratory_probability": s.get("fakeprint_exploratory_probability"),
                "region_label": s.get("region_label"),
            }
            for s in t_steps
            if s.get("region_label") == "Elevated AI-associated features"
        ],
        "disagreement_windows": [
            {
                "window_index": s["window_index"],
                "start_sec": s["start_sec"],
                "end_sec": s["end_sec"],
                "sonics_gamma_probability": s.get("sonics_gamma_probability"),
                "fakeprint_exploratory_probability": s.get("fakeprint_exploratory_probability"),
                "disagreement_spread": s.get("disagreement_spread"),
                "rationale": s.get("rationale"),
            }
            for s in t_steps
            if s.get("disagreement")
        ],
        "abstained_or_ood_windows": [
            {
                "window_index": s["window_index"],
                "start_sec": s["start_sec"],
                "end_sec": s["end_sec"],
                "rms_dbfs": s.get("rms_dbfs"),
                "unpadded_duration_sec": s.get("unpadded_duration_sec"),
                "warnings": s.get("warnings", []),
            }
            for s in t_steps
            if s.get("abstained") or s.get("ood_detected")
        ],
        "section_transitions": (temporal_analysis or {}).get("section_transitions", []),
        "contiguous_region_summaries": (temporal_analysis or {}).get("region_summaries", []),
    }

    processed_sha256_latest = exp_list[0]["processed_sha256"] if exp_list else None
    all_processed_sha256s = [e["processed_sha256"] for e in exp_list if e.get("processed_sha256")]

    return {
        "report_metadata": {
            "laboratory": "SPECTRA | AI Audio Forensics & Fingerprint Analysis Laboratory",
            "version": settings.SPECTRA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "job_id": job.id,
            "reproducible_experiment_id": f"spectra-exp-{reproducible_experiment_id}",
            "scientific_methodology_statement": (
                "SPECTRA distinguishes five separate forensic mechanisms and five epistemic attribution tiers: "
                "(1) Cryptographically verified C2PA facts, (2) Unverified conventional metadata assertions, "
                "(3) Objective audio signal observations, (4) Statistical model predictions, and "
                "(5) Non-definitive research hypotheses. No single detector score or encoder tag constitutes "
                "definitive proof of content origin."
            ),
        },
        "original_file_sha256": asset.sha256_checksum,
        "processed_file_sha256": processed_sha256_latest,
        "all_processed_file_sha256s": all_processed_sha256s,
        "c2pa_validation_status": {
            "c2pa_status": (prov_dict or {}).get("c2pa_status", "not_inspected"),
            "credential_present": bool((prov_dict or {}).get("credential_present", False)),
            "signature_valid": (prov_dict or {}).get("signature_valid"),
            "claim_generator": (prov_dict or {}).get("claim_generator"),
            "active_manifest_id": (prov_dict or {}).get("active_manifest_id"),
            "validation_errors": (prov_dict or {}).get("validation_errors", []),
            "validation_warnings": (prov_dict or {}).get("validation_warnings", []),
        },
        "conventional_metadata": (prov_dict or {}).get("conventional_metadata", {}),
        "detector_revisions": {m["detector_id"]: m["model_revision"] for m in model_revisions},
        "model_preprocessing": model_revisions,
        "raw_scores": [
            {
                "detector_id": d_item["detector_id"],
                "raw_score": d_item["raw_score"],
                "probability": d_item["probability"],
                "calibrated_probability": (d_item.get("details") or {}).get(
                    "calibrated_probability", d_item["probability"]
                ),
                "threshold": d_item["threshold"],
                "predicted_label": d_item["predicted_label"],
                "abstained": bool((d_item.get("details") or {}).get("abstained", False)),
            }
            for d_item in detector_list
        ],
        "ood_warnings": input_validity_warnings,
        "temporal_scores": (temporal_analysis or {}).get("summary_statistics", {}),
        "selected_time_windows": selected_time_windows,
        "audio_comparison_metrics": [
            {
                "experiment_id": c["experiment_id"],
                "delta_rms_db": c["delta_rms_db"],
                "delta_peak_db": c["delta_peak_db"],
                "spectral_distance_l2": c["spectral_distance_l2"],
                "spectral_convergence": c["spectral_convergence"],
                "snr_db": c["snr_db"],
                "psnr_db": c["psnr_db"],
                "loudness_delta_db": c["loudness_delta_db"],
                "alignment_offset_samples": c["alignment_offset_samples"],
                "detector_deltas": c["detector_deltas"],
                "temporal_comparison_summary": (c.get("temporal_comparison") or {}).get("summary"),
            }
            for c in comp_list
        ],
        "experiment_parameters": [
            {
                "experiment_id": e["id"],
                "operation_type": e["operation_type"],
                "preset_name": e["preset_name"],
                "engine_used": e["engine_used"],
                "parameters": e["parameters"],
                "original_sha256": e["original_sha256"],
                "processed_sha256": e["processed_sha256"],
            }
            for e in exp_list
        ],
        "reproducibility_information": {
            "reproducible_experiment_id": f"spectra-exp-{reproducible_experiment_id}",
            "job_id": job.id,
            "audio_asset_id": asset.id,
            "original_file_sha256": asset.sha256_checksum,
            "processed_file_sha256": processed_sha256_latest,
            "temporal_cache_key": (temporal_analysis or {}).get("cache_key"),
            "temporal_cache_key_components": (temporal_analysis or {}).get("cache_key_components"),
            "software_versions": sw_versions,
        },
        "limitations_and_attribution_confidence": {
            "attribution_confidence_statement": (
                "High cryptographic confidence ONLY when C2PA signature_valid=True. When C2PA is absent, "
                "all detector outputs represent domain-bounded statistical similarity scores and never "
                "constitute standalone proof of generator origin or human authorship."
            ),
            "known_domain_limitations": [
                "Continuous latent diffusion models (Stable Audio, ACE-Step 1.0/1.5) are out-of-domain for both detectors (0%–33% recall).",
                "SONICS SpecTTTra-α (120s) requires >= 115s unpadded audio; shorter clips trigger transparent abstention.",
                "Segment-level Fakeprint ONNX scores are exploratory due to higher STFT estimation variance on 5s windows (19 frames) vs full-track averages.",
            ],
            "calibration_and_ensemble_policy": (
                "Detectors are reported independently alongside raw logits/probabilities and OOD abstention flags."
            ),
        },
        "reproducible_experiment_identifiers": {
            "reproducible_experiment_id": f"spectra-exp-{reproducible_experiment_id}",
            "job_id": job.id,
            "audio_asset_id": asset.id,
            "audio_asset_sha256": asset.sha256_checksum,
            "processing_experiment_ids": [e["id"] for e in exp_list],
            "comparison_ids": [c["id"] for c in comp_list],
        },
        "model_and_checkpoint_revisions": model_revisions,
        "dataset_provenance_context": {
            "reference_manifest_path": "research/benchmarks/dataset_provenance_manifest.json",
            "upstream_training_domains": {
                "fakeprint_onnx": "lofcz/ai-music-detector (Suno/Udio 1-8 kHz spectral hull residuals vs human audio)",
                "sonics_gamma_5s": "awsaf49/sonics (Suno v2/v3/v3.5 & Udio v1 vs human songs, 5s @ 16 kHz, CC BY-NC 4.0 dataset)",
                "sonics_alpha_120s": "awsaf49/sonics (Suno v2/v3/v3.5 & Udio v1 vs human songs, 120s @ 16 kHz, CC BY-NC 4.0 dataset)",
            },
            "known_domain_limitations": [
                "Continuous latent diffusion models (Stable Audio, ACE-Step 1.0/1.5) are out-of-domain for both detectors (0%–17% recall).",
                "SONICS SpecTTTra-α (120s) requires >= 115s unpadded audio; shorter clips trigger transparent abstention because upstream zero-padding introduces -80 dB spectrogram floor bias.",
            ],
        },
        "input_validity_warnings": input_validity_warnings,
        "confidence_and_calibration_limitations": {
            "fakeprint_onnx_calibration": (
                "Raw ONNX logistic regression outputs saturate near 0.0 or 1.0; temperature scaling (T=3.25) is "
                "provided in details.calibrated_probability to soften extreme logits without altering raw scores."
            ),
            "sonics_window_uncertainty": (
                "SONICS Gamma 5s evaluates multiple 5s windows across the track and reports inter-window IQR and "
                "binary entropy; sparse instrumental intros frequently score < 0.10 even on synthetic tracks."
            ),
            "ensemble_limitation": (
                "SPECTRA reports each detector independently rather than recommending a fixed linear ensemble, "
                "because linear blending degrades sensitivity when one detector experiences generator-version domain shift."
            ),
        },
        "forensic_attribution_hierarchy": forensic_attribution_hierarchy,
        "observed_measurements_vs_conclusions": {
            "observed_measurements": {
                "asset_sha256": asset.sha256_checksum,
                "processed_file_sha256": processed_sha256_latest,
                "duration_seconds": asset.duration_seconds,
                "sample_rate_hz": asset.sample_rate,
                "c2pa_signature_valid": (prov_dict or {}).get("signature_valid"),
                "time_domain_metrics": td_metrics,
                "frequency_domain_summary": {
                    "spectral_centroid_hz": fd_metrics.get("spectral_centroid_hz"),
                    "spectral_rolloff_95_hz": fd_metrics.get("spectral_rolloff_95_hz"),
                    "hf_energy_ratio_above_12khz": fd_metrics.get("hf_energy_ratio_above_12khz"),
                },
                "fakeprint_1_8khz_metrics": {
                    "periodicity_score": fp_metrics.get("periodicity_score"),
                    "comb_grid_200hz_contrast": fp_metrics.get("comb_grid_200hz_contrast"),
                    "temporal_stationarity": fp_metrics.get("temporal_stationarity"),
                },
                "raw_detector_outputs": [
                    {
                        "detector_id": d_item["detector_id"],
                        "raw_score": d_item["raw_score"],
                        "probability": d_item["probability"],
                        "abstained": bool((d_item.get("details") or {}).get("abstained", False)),
                    }
                    for d_item in detector_list
                ],
                "temporal_window_summary": (temporal_analysis or {}).get("summary_statistics", {}),
            },
            "analytical_conclusions_and_hypotheses": research_hypotheses,
        },
        "audio_asset": {
            "id": asset.id,
            "original_filename": asset.original_filename,
            "sanitized_filename": asset.sanitized_filename,
            "format": asset.original_format,
            "mime_type": asset.mime_type,
            "codec": asset.codec,
            "container_format": asset.container_format,
            "duration_seconds": asset.duration_seconds,
            "sample_rate_hz": asset.sample_rate,
            "channels": asset.channels,
            "bit_depth": asset.bit_depth,
            "bitrate_bps": asset.bitrate,
            "file_size_bytes": asset.file_size_bytes,
            "sha256_checksum": asset.sha256_checksum,
            "created_at": asset.created_at.isoformat() if asset.created_at else None,
        },
        "provenance_and_metadata": prov_dict,
        "signal_forensics": forensics,
        "ai_detectors": detector_list,
        "temporal_forensic_analysis": temporal_analysis,
        "processing_experiments": exp_list,
        "before_after_comparisons": comp_list,
        "software_environment": sw_versions,
    }


def render_pdf_bytes(payload: Dict[str, Any]) -> bytes:
    """Render a professional multi-section PDF technical report using ReportLab."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=f"SPECTRA Forensic Report - {payload['audio_asset']['original_filename']}",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "SpectraTitle",
        parent=styles["Heading1"],
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#1E1B4B"),
        spaceAfter=6,
    )
    h2_style = ParagraphStyle(
        "SpectraH2",
        parent=styles["Heading2"],
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#4338CA"),
        spaceBefore=10,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "SpectraBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=12,
    )
    small_style = ParagraphStyle(
        "SpectraSmall",
        parent=styles["BodyText"],
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#475569"),
    )

    story = []
    meta = payload["report_metadata"]
    asset = payload["audio_asset"]

    story.append(Paragraph("SPECTRA | AI Audio Forensics & Signal Processing Laboratory", title_style))
    story.append(Paragraph(f"Generated: {meta['generated_at_utc']} | Job ID: {meta['job_id']}", small_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph(meta["scientific_methodology_statement"], small_style))
    story.append(Spacer(1, 8))

    # Section 1: Asset Technical Profile
    story.append(Paragraph("1. Audio Asset & Cryptographic Integrity", h2_style))
    asset_rows = [
        ["Property", "Value"],
        ["Original Filename", str(asset["original_filename"])],
        ["Asset UUID", str(asset["id"])],
        ["SHA-256 Checksum", str(asset["sha256_checksum"])],
        ["Format / Codec", f"{asset['format']} ({asset['codec']}, {asset['container_format']})"],
        ["Duration / Sample Rate", f"{asset['duration_seconds']:.3f} s @ {asset['sample_rate_hz']} Hz"],
        ["Channels / Bit Depth", f"{asset['channels']} ch | {asset['bit_depth'] or 'N/A'} bit"],
        ["File Size", f"{asset['file_size_bytes']:,} bytes"],
    ]
    t_asset = Table(asset_rows, colWidths=[48 * mm, 124 * mm])
    t_asset.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#312E81")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("PADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(t_asset)

    # Section 2: C2PA & Conventional Metadata
    story.append(Paragraph("2. C2PA Content Credentials & Conventional Metadata", h2_style))
    prov = payload.get("provenance_and_metadata") or {}
    c2pa_rows = [
        ["Field", "Details"],
        ["C2PA Status", str(prov.get("c2pa_status", "N/A"))],
        ["Credential Present", str(prov.get("credential_present", False))],
        ["Signature Valid", str(prov.get("signature_valid", "N/A"))],
        ["Claim Generator", str(prov.get("claim_generator") or "None")],
        ["Conventional Tags Count", str(len(prov.get("conventional_metadata", {})))],
    ]
    t_c2pa = Table(c2pa_rows, colWidths=[48 * mm, 124 * mm])
    t_c2pa.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
            ("PADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(t_c2pa)

    # Section 3: Signal Forensics
    story.append(Paragraph("3. Time-Domain & Spectral Forensics", h2_style))
    sf_data = payload.get("signal_forensics") or {}
    td = sf_data.get("time_domain", {})
    fd = sf_data.get("frequency_domain", {})
    fp = sf_data.get("fakeprint", {})
    sig_rows = [
        ["Metric", "Measured Value"],
        ["Peak / RMS Level", f"{td.get('peak_dbfs', 'N/A')} dBFS / {td.get('rms_dbfs', 'N/A')} dBFS"],
        ["Crest Factor", f"{td.get('crest_factor_db', 'N/A')} dB"],
        ["Clipping Detected", f"{td.get('clipping_detected', False)} ({td.get('clipping_samples', 0)} samples)"],
        ["Spectral Centroid / Bandwidth", f"{fd.get('spectral_centroid_hz_mean', 'N/A')} Hz / {fd.get('spectral_bandwidth_hz_mean', 'N/A')} Hz"],
        ["Spectral Rolloff (85% / 95%)", f"{fd.get('spectral_rolloff_85_hz', 'N/A')} Hz / {fd.get('spectral_rolloff_95_hz', 'N/A')} Hz"],
        ["1-8 kHz Residual Periodicity", f"{fp.get('periodicity_score', 'N/A')} (Max residual: {fp.get('residual_max_raw_db', 'N/A')} dB)"],
    ]
    t_sig = Table(sig_rows, colWidths=[60 * mm, 112 * mm])
    t_sig.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
            ("PADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(t_sig)

    # Section 4: AI Detector Runs
    story.append(Paragraph("4. AI-Music Detector Results", h2_style))
    det_rows = [["Detector", "Status", "P(AI)", "Threshold", "Classification", "Revision"]]
    for d in payload.get("ai_detectors", []):
        p_str = f"{d['probability']*100:.2f}%" if d.get("probability") is not None else "N/A"
        det_rows.append([
            str(d.get("detector_id", "")),
            str(d.get("status", "")),
            p_str,
            f"{d.get('threshold', 0.5):.2f}",
            str(d.get("predicted_label", "")),
            str(d.get("model_revision", ""))[:10],
        ])
    t_det = Table(det_rows, colWidths=[32 * mm, 22 * mm, 20 * mm, 20 * mm, 52 * mm, 26 * mm])
    t_det.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#312E81")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
            ("PADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(t_det)

    # Section 4B: Temporal Forensic Timeline & Region Summaries (Phase 5 Part C)
    temp_data = payload.get("temporal_forensic_analysis") or {}
    if temp_data and temp_data.get("region_summaries"):
        story.append(Paragraph("4B. Temporal Sliding-Window Analysis & Neutral Region Summaries", h2_style))
        cfg = temp_data.get("configuration", {})
        stats_g = (temp_data.get("summary_statistics") or {}).get("sonics_gamma_5s", {})
        story.append(
            Paragraph(
                f"Window: {cfg.get('sonics_gamma_window_sec', 5.0)}s | Hop: {cfg.get('hop_sec', 2.5)}s | "
                f"Valid Windows: {stats_g.get('valid_window_count', 0)}/{stats_g.get('total_window_count', 0)} | "
                f"Disagreement Windows: {(temp_data.get('summary_statistics') or {}).get('disagreement_window_count', 0)}",
                small_style,
            )
        )
        reg_rows = [["Region Span", "Neutral Classification", "Windows", "Mean SONICS-γ", "Fakeprint (Expl.)"]]
        for r_item in temp_data.get("region_summaries", [])[:12]:
            ms_str = (
                f"{r_item['mean_sonics_probability']*100:.1f}%"
                if r_item.get("mean_sonics_probability") is not None
                else "N/A"
            )
            fp_str = (
                f"{r_item['mean_fakeprint_exploratory_probability']*100:.1f}%"
                if r_item.get("mean_fakeprint_exploratory_probability") is not None
                else "N/A"
            )
            reg_rows.append([
                f"{r_item.get('start_sec', 0):.2f}s - {r_item.get('end_sec', 0):.2f}s",
                str(r_item.get("region_label", "")),
                str(r_item.get("window_count", 0)),
                ms_str,
                fp_str,
            ])
        t_reg = Table(reg_rows, colWidths=[34 * mm, 64 * mm, 18 * mm, 28 * mm, 28 * mm])
        t_reg.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                ("PADDING", (0, 0), (-1, -1), 3.5),
            ])
        )
        story.append(t_reg)

    # Section 5: Processing Experiments & Comparisons
    comps = payload.get("before_after_comparisons", [])
    if comps:
        story.append(Paragraph("5. DSP Processing & Before/After Comparison", h2_style))
        for c in comps:
            story.append(
                Paragraph(
                    f"Experiment {c['experiment_id']}: ΔRMS={c['delta_rms_db']:+.2f} dB, "
                    f"ΔPeak={c['delta_peak_db']:+.2f} dB, LSD={c['spectral_distance_l2']:.3f} dB, "
                    f"SNR={c['snr_db']} dB",
                    body_style,
                )
            )
            for dd in c.get("detector_deltas", []):
                story.append(
                    Paragraph(
                        f"• {dd['detector_id']}: Original P(AI)={dd['original_probability']} -> "
                        f"Processed P(AI)={dd['processed_probability']} (Δ={dd['delta_probability']})",
                        small_style,
                    )
                )

    # Section 6: Epistemic Attribution Hierarchy & Calibration Limitations
    story.append(Paragraph("6. Forensic Attribution Hierarchy & Limitations", h2_style))
    story.append(
        Paragraph(
            f"Reproducible Experiment ID: {meta.get('reproducible_experiment_id', 'N/A')}",
            small_style,
        )
    )
    in_warns = payload.get("input_validity_warnings", [])
    if in_warns:
        for w in in_warns[:5]:
            story.append(Paragraph(f"• Warning: {w}", small_style))
    cal_lim = payload.get("confidence_and_calibration_limitations", {})
    if cal_lim:
        story.append(Paragraph(f"• Calibration & Ensemble Policy: {cal_lim.get('ensemble_limitation', '')}", small_style))
    hyps = (payload.get("forensic_attribution_hierarchy", {}).get("tier_5_research_hypotheses", {}) or {}).get("hypotheses", [])
    for h_txt in hyps:
        story.append(Paragraph(f"• Hypothesis: {h_txt}", small_style))

    doc.build(story)
    return buf.getvalue()


def generate_and_store_report(
    db: Session,
    job: AnalysisJob,
    asset: AudioAsset,
    report_format: str = "json",
    experiment_id: Optional[str] = None,
    include_raw_manifest: bool = True,
) -> Tuple[GeneratedReport, bytes]:
    """Generate a JSON or PDF technical report, save it to storage/reports, and persist GeneratedReport."""
    fmt_clean = report_format.lower().strip()
    if fmt_clean not in {"json", "pdf"}:
        raise ValueError("Report format must be 'json' or 'pdf'.")

    payload = build_report_payload(
        db=db,
        job=job,
        asset=asset,
        experiment_id=experiment_id,
        include_raw_manifest=include_raw_manifest,
    )

    if fmt_clean == "json":
        raw_bytes = json.dumps(payload, indent=2).encode("utf-8")
        ext = ".json"
    else:
        raw_bytes = render_pdf_bytes(payload)
        ext = ".pdf"

    rep_id = str(uuid.uuid4())
    rel_path = f"reports/{rep_id}{ext}"
    abs_path = settings.storage_path / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_bytes(raw_bytes)

    sha256_hex = hashlib.sha256(raw_bytes).hexdigest()
    summary = {
        "asset_filename": asset.original_filename,
        "asset_sha256": asset.sha256_checksum,
        "c2pa_status": (payload.get("provenance_and_metadata") or {}).get("c2pa_status"),
        "detectors_evaluated": len(payload.get("ai_detectors", [])),
        "experiments_included": len(payload.get("processing_experiments", [])),
    }

    report_row = GeneratedReport(
        id=rep_id,
        job_id=job.id,
        experiment_id=experiment_id,
        audio_asset_id=asset.id,
        report_format=fmt_clean,
        title=f"SPECTRA Forensic Report - {asset.original_filename} ({fmt_clean.upper()})",
        storage_rel_path=rel_path,
        sha256_checksum=sha256_hex,
        file_size_bytes=len(raw_bytes),
        summary_json=json.dumps(summary),
    )
    db.add(report_row)
    db.commit()
    db.refresh(report_row)

    return report_row, raw_bytes
