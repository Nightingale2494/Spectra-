"""Temporal Audio Forensics & Sliding-Window Inference Engine for SPECTRA (Phase 5).

Implements:
- Sliding-window temporal evaluation for SONICS SpecTTTra-gamma (5s windows, configurable hop)
  with exact upstream preprocessing (16 kHz mono, mode='constant' zero-padding on incomplete
  final windows, independent per-window unit-variance normalization and (1, T) inference).
- Configurable exploratory segment-level evaluation for Fakeprint ONNX (exact 8192-point STFT
  3,585-bin 1-8 kHz lower-hull residual extraction per segment, marked exploratory=True).
- Strict separation of SONICS SpecTTTra-alpha (120s) long-context predictions so 120s spans
  are never misrepresented as localized 5s predictions.
- Contiguous forensic region aggregation using strictly neutral scientific labels:
  * "Elevated AI-associated features"
  * "Low AI-associated features"
  * "Model disagreement"
  * "Insufficient evidence"
- Section transition detection and hybrid recording heterogeneity analysis separating
  known provenance facts from localized model predictions.
- Aligned before-versus-after temporal DSP comparison across segments.
- Single-decode in-memory window slicing and SHA-256 deterministic caching.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import librosa
import numpy as np
import soxr
import torch

from app.core.config import settings
from app.detectors.registry import detector_registry
from app.services.forensics_service import extract_neural_vocoder_fakeprint

EPS = 1e-10
DEFAULT_PREPROCESSING_VERSION = "spectra_preproc_v2_soxr_hq_16k_unit_var_const_pad"

# Neutral scientific labels required by Phase 5 Part A.3
LABEL_ELEVATED = "Elevated AI-associated features"
LABEL_LOW = "Low AI-associated features"
LABEL_DISAGREEMENT = "Model disagreement"
LABEL_INSUFFICIENT = "Insufficient evidence"

# Active job cancellation registry for long-running temporal jobs
_CANCELLED_TEMPORAL_JOBS: set[str] = set()
_CANCEL_LOCK = threading.Lock()


def request_temporal_job_cancel(job_id: str) -> None:
    with _CANCEL_LOCK:
        _CANCELLED_TEMPORAL_JOBS.add(job_id)


def clear_temporal_job_cancel(job_id: str) -> None:
    with _CANCEL_LOCK:
        _CANCELLED_TEMPORAL_JOBS.discard(job_id)


def is_temporal_job_cancelled(job_id: Optional[str]) -> bool:
    if not job_id:
        return False
    with _CANCEL_LOCK:
        return job_id in _CANCELLED_TEMPORAL_JOBS


def _get_cache_dir() -> Path:
    base_temp = getattr(settings, "temp_dir", settings.storage_path / "temp")
    d = Path(base_temp) / "temporal_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_temporal_cache_key_components(
    asset_sha256: str,
    hop_sec: float,
    sonics_window_sec: float,
    fakeprint_window_sec: float,
    include_exploratory_fakeprint: bool,
    detector_revisions: Dict[str, str],
    preprocessing_version: str = DEFAULT_PREPROCESSING_VERSION,
    inference_settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the canonical dictionary of all 7 required cache key components (Phase 6 Priority 3):
    1. Source audio hash (source_audio_sha256)
    2. Model identity (model_identities)
    3. Model revision (model_revisions)
    4. Window length (window_length_sec)
    5. Hop length (hop_length_sec)
    6. Preprocessing version (preprocessing_version)
    7. Relevant inference settings (inference_settings)
    """
    effective_settings = {
        "include_exploratory_fakeprint": bool(include_exploratory_fakeprint),
        "sonics_threshold": 0.5,
        "fakeprint_threshold": 0.5,
        "min_unpadded_sec": 3.0,
        "silence_gate_dbfs": -60.0,
        "fakeprint_temperature": 3.25,
    }
    if inference_settings:
        effective_settings.update(inference_settings)

    return {
        "cache_schema": "phase6_temporal_cache_v2",
        "source_audio_sha256": str(asset_sha256),
        "model_identities": sorted(list(detector_revisions.keys())),
        "model_revisions": dict(sorted(detector_revisions.items())),
        "window_length_sec": {
            "sonics_gamma_5s": round(float(sonics_window_sec), 4),
            "fakeprint_onnx": round(float(fakeprint_window_sec), 4),
            "sonics_alpha_120s": 120.0,
        },
        "hop_length_sec": round(float(hop_sec), 4),
        "preprocessing_version": str(preprocessing_version),
        "inference_settings": effective_settings,
    }


def compute_temporal_cache_key(
    asset_sha256: str,
    hop_sec: float,
    fakeprint_window_sec: float,
    include_exploratory_fakeprint: bool,
    detector_revisions: Dict[str, str],
    sonics_window_sec: float = 5.0,
    preprocessing_version: str = DEFAULT_PREPROCESSING_VERSION,
    inference_settings: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    components = build_temporal_cache_key_components(
        asset_sha256=asset_sha256,
        hop_sec=hop_sec,
        sonics_window_sec=sonics_window_sec,
        fakeprint_window_sec=fakeprint_window_sec,
        include_exploratory_fakeprint=include_exploratory_fakeprint,
        detector_revisions=detector_revisions,
        preprocessing_version=preprocessing_version,
        inference_settings=inference_settings,
    )
    payload = json.dumps(components, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), components


def _compute_cache_key(
    asset_sha256: str,
    hop_sec: float,
    fakeprint_window_sec: float,
    include_exploratory_fakeprint: bool,
    detector_revisions: Dict[str, str],
    sonics_window_sec: float = 5.0,
    preprocessing_version: str = DEFAULT_PREPROCESSING_VERSION,
    inference_settings: Optional[Dict[str, Any]] = None,
) -> str:
    key, _ = compute_temporal_cache_key(
        asset_sha256=asset_sha256,
        hop_sec=hop_sec,
        fakeprint_window_sec=fakeprint_window_sec,
        include_exploratory_fakeprint=include_exploratory_fakeprint,
        detector_revisions=detector_revisions,
        sonics_window_sec=sonics_window_sec,
        preprocessing_version=preprocessing_version,
        inference_settings=inference_settings,
    )
    return key


def generate_sliding_window_bounds(
    duration_sec: float,
    window_sec: float = 5.0,
    hop_sec: float = 2.5,
) -> List[Tuple[float, float, bool]]:
    """
    Generate ordered (start_sec, end_sec, is_incomplete_final) tuples covering [0, duration_sec].

    Rules:
    - If duration_sec <= 0: returns [].
    - If duration_sec < window_sec: returns a single incomplete window (0.0, duration_sec, True).
    - Otherwise steps by hop_sec while start_sec + window_sec <= duration_sec + 1e-6.
    - If an unanalyzed tail remains at the end (duration_sec - last_end_sec > 0.25s),
      appends one final incomplete window (next_start_sec, duration_sec, True) so the
      tail is evaluated with documented upstream mode='constant' zero-padding (and
      abstained if unpadded length < 3.0s).
    """
    if duration_sec <= 0.0:
        return []
    window_sec = max(0.5, float(window_sec))
    hop_sec = max(0.25, float(hop_sec))

    if duration_sec < window_sec - 1e-6:
        return [(0.0, round(duration_sec, 4), True)]

    bounds: List[Tuple[float, float, bool]] = []
    idx = 0
    while True:
        start = round(idx * hop_sec, 4)
        end = round(start + window_sec, 4)
        if end <= duration_sec + 1e-6:
            bounds.append((start, min(round(duration_sec, 4), end), False))
            idx += 1
        else:
            # Check if there is a meaningful unanalyzed tail beyond the last full window end
            last_end = bounds[-1][1] if bounds else 0.0
            remaining_tail = round(duration_sec - last_end, 4)
            if remaining_tail > 0.25 and start < duration_sec - 0.1:
                bounds.append((start, round(duration_sec, 4), True))
            break
    return bounds


def _evaluate_sonics_gamma_window(
    sonics_gamma: Any,
    audio_16k: np.ndarray,
    sr_16k: int,
    start_sec: float,
    end_sec: float,
    is_incomplete_final: bool,
) -> Dict[str, Any]:
    """
    Evaluate a single 5.0s window with SONICS SpecTTTra-gamma (5s) using exact upstream
    preprocessing and independent (1, T) inference.
    """
    expected_samples = int(sr_16k * 5.0)
    start_idx = max(0, int(round(start_sec * sr_16k)))
    end_idx = min(len(audio_16k), int(round(end_sec * sr_16k)))
    raw_slice = audio_16k[start_idx:end_idx]
    unpadded_len = len(raw_slice)
    unpadded_sec = float(unpadded_len) / float(sr_16k)

    rms_lin = float(np.sqrt(np.mean(raw_slice ** 2))) if unpadded_len > 0 else 0.0
    rms_dbfs = 20.0 * np.log10(max(rms_lin, 1e-10))
    peak_lin = float(np.max(np.abs(raw_slice))) if unpadded_len > 0 else 0.0
    crest_factor_db = 20.0 * np.log10(max(peak_lin, 1e-10) / max(rms_lin, 1e-10))
    clipping_ratio = float(np.mean(np.abs(raw_slice) >= 0.999)) if unpadded_len > 0 else 0.0

    warnings: List[str] = []
    abstained = False
    ood_detected = False

    # Prepare exact 80,000-sample window with upstream mode="constant" zero-padding if shorter
    if unpadded_len < expected_samples:
        pad_len = expected_samples - unpadded_len
        win_padded = np.pad(raw_slice, (0, pad_len), mode="constant")
        zero_pad_sec = float(pad_len) / float(sr_16k)
    else:
        win_padded = raw_slice[:expected_samples]
        zero_pad_sec = 0.0

    # Independent per-window standard-deviation normalization (no cross-window contamination)
    std_val = float(np.std(win_padded))
    if std_val > 1e-6:
        win_norm = (win_padded / std_val).astype(np.float32)
    else:
        win_norm = win_padded.astype(np.float32)

    # Check OOD / validity conditions before interpreting the score
    if rms_dbfs < -60.0:
        abstained = True
        ood_detected = True
        warnings.append(
            f"Segment is near-silent (RMS={rms_dbfs:.1f} dBFS < -60.0 dBFS); classification abstained."
        )

    if unpadded_sec < 3.0:
        abstained = True
        ood_detected = True
        warnings.append(
            f"Incomplete segment duration ({unpadded_sec:.2f}s < 3.0s minimum); "
            f"upstream mode='constant' zero-pads {zero_pad_sec:.2f}s with silence; classification abstained."
        )
    elif is_incomplete_final or zero_pad_sec > 0.05:
        warnings.append(
            f"Incomplete final window ({unpadded_sec:.2f}s unpadded audio zero-padded by {zero_pad_sec:.2f}s "
            "to 5.0s using upstream mode='constant')."
        )

    if clipping_ratio >= 0.005:
        warnings.append(
            f"Digital clipping in segment ({clipping_ratio * 100:.2f}% samples >= 0.999 FS)."
        )

    prob: Optional[float] = None
    logit: Optional[float] = None
    status = "unavailable"
    device_used = getattr(sonics_gamma, "active_device", "cpu")

    if sonics_gamma is not None:
        if sonics_gamma.model is None:
            sonics_gamma.load()
        if sonics_gamma.model is not None:
            try:
                prob_val, logit_val = sonics_gamma._score_single_window(win_norm)
                prob = round(float(prob_val), 6)
                logit = round(float(logit_val), 6)
                status = "success"
                device_used = sonics_gamma.active_device
            except RuntimeError as exc:
                # Graceful CPU fallback on CUDA OOM
                if "out of memory" in str(exc).lower() and sonics_gamma.active_device == "cuda":
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if sonics_gamma.load(device="cpu"):
                        prob_val, logit_val = sonics_gamma._score_single_window(win_norm)
                        prob = round(float(prob_val), 6)
                        logit = round(float(logit_val), 6)
                        status = "success"
                        device_used = "cpu"
                        warnings.append("CUDA out-of-memory encountered; fell back to CPU cleanly.")
                    else:
                        status = "error"
                        warnings.append(f"Inference error: {exc}")
                else:
                    status = "error"
                    warnings.append(f"Inference error: {exc}")
            except Exception as exc:
                status = "error"
                warnings.append(f"Inference error: {exc}")

    return {
        "detector_id": "sonics_gamma_5s",
        "model_version": getattr(sonics_gamma, "version", "0.2.0-iclr25"),
        "model_revision": getattr(sonics_gamma, "model_revision", "unknown"),
        "start_sec": round(start_sec, 3),
        "end_sec": round(end_sec, 3),
        "window_sec": 5.0,
        "unpadded_duration_sec": round(unpadded_sec, 3),
        "zero_pad_sec": round(zero_pad_sec, 3),
        "is_incomplete_final_window": bool(is_incomplete_final),
        "probability": prob,
        "raw_score": logit,
        "threshold": getattr(sonics_gamma, "threshold", 0.5),
        "rms_dbfs": round(rms_dbfs, 2),
        "crest_factor_db": round(crest_factor_db, 2),
        "clipping_ratio": round(clipping_ratio, 6),
        "abstained": abstained,
        "ood_detected": ood_detected,
        "exploratory": False,
        "device_used": device_used,
        "status": status,
        "warnings": warnings,
    }


def _evaluate_fakeprint_segment(
    fakeprint_det: Any,
    audio_16k: np.ndarray,
    sr_16k: int,
    start_sec: float,
    end_sec: float,
    window_sec: float,
) -> Dict[str, Any]:
    """
    Evaluate a single segment with Fakeprint ONNX in exploratory segment-level mode.
    Preserves exact 8192-point STFT @ 16 kHz and 3,585-bin (1,000-8,000 Hz) feature dimension.
    """
    start_idx = max(0, int(round(start_sec * sr_16k)))
    end_idx = min(len(audio_16k), int(round(end_sec * sr_16k)))
    seg_slice = audio_16k[start_idx:end_idx]
    seg_dur = float(len(seg_slice)) / float(sr_16k)

    warnings: List[str] = [
        "Exploratory segment-level Fakeprint score: 8192-point STFT lower-hull residual is averaged "
        f"over a {seg_dur:.2f}s window rather than full-track duration."
    ]
    abstained = False
    ood_detected = False

    if seg_dur < 1.5 or len(seg_slice) < 8192:
        abstained = True
        ood_detected = True
        warnings.append(
            f"Segment duration ({seg_dur:.2f}s) is shorter than 1.5s minimum required for 8192-point STFT averaging."
        )

    samples_2d = seg_slice.reshape(1, -1) if len(seg_slice) > 0 else np.zeros((1, 1600), dtype=np.float32)
    fp_res = extract_neural_vocoder_fakeprint(
        samples=samples_2d,
        sr=sr_16k,
        target_sr=16000,
        n_fft=8192,
        freq_min=1000,
        freq_max=8000,
        hull_area=10,
        max_db=5.0,
        min_db=-45.0,
        max_duration=300,
    )
    vec = fp_res["feature_vector"]
    rms_dbfs = float(fp_res.get("signal_rms_dbfs", -100.0))
    res_max_db = float(fp_res.get("residual_max_raw_db", 0.0))
    band_ratio = float(fp_res.get("band_1k_8k_energy_ratio", 0.0))
    active_ratio = float(fp_res.get("active_bin_ratio", 0.0))
    periodicity = float(fp_res.get("periodicity_score", 0.0))
    stationarity = float(fp_res.get("temporal_stationarity", 0.0))

    if rms_dbfs < -65.0 or res_max_db < 0.25:
        abstained = True
        ood_detected = True
        warnings.append(f"Near-silence or negligible 1-8 kHz energy in segment (RMS={rms_dbfs:.1f} dBFS).")
    if band_ratio < 0.005:
        abstained = True
        ood_detected = True
        warnings.append(f"Insufficient 1-8 kHz energy ratio ({band_ratio:.5f} < 0.005).")
    if active_ratio < 0.025:
        abstained = True
        ood_detected = True
        warnings.append(f"Sparse narrowband residual ({active_ratio * 100:.2f}% active bins < 2.5%).")

    prob: Optional[float] = None
    calibrated_prob: Optional[float] = None
    raw_logit: Optional[float] = None
    status = "unavailable"

    if fakeprint_det is not None:
        if fakeprint_det.session is None:
            fakeprint_det.load()
        if fakeprint_det.session is not None and fakeprint_det.input_name is not None:
            try:
                inp = vec.reshape(1, fakeprint_det.expected_features).astype(np.float32)
                outputs = fakeprint_det.session.run(None, {fakeprint_det.input_name: inp})
                p_raw = float(np.clip(np.asarray(outputs[0]).reshape(-1)[0], 0.0, 1.0))
                p_clamped = float(np.clip(p_raw, 1e-6, 1.0 - 1e-6))
                logit_val = float(np.log(p_clamped / (1.0 - p_clamped)))
                cal_val = float(1.0 / (1.0 + np.exp(-logit_val / 3.25)))
                prob = round(p_raw, 6)
                calibrated_prob = round(cal_val, 6)
                raw_logit = round(logit_val, 4)
                status = "success"
                if p_raw >= 0.5 and stationarity < 0.25 and not abstained:
                    warnings.append(
                        f"Low temporal stationarity ({stationarity:.2f} < 0.25) in segment; transient broadband "
                        "distortion or overdriven guitar harmonics may elevate 1-8 kHz hull residuals."
                    )
            except Exception as exc:
                status = "error"
                warnings.append(f"ONNX segment inference error: {exc}")

    stft_frames = max(1, 1 + max(0, len(seg_slice) - 8192) // 4096) if len(seg_slice) > 0 else 0
    return {
        "detector_id": "fakeprint_onnx",
        "model_version": getattr(fakeprint_det, "version", "1.0.0-ismir25"),
        "model_revision": getattr(fakeprint_det, "model_revision", "unknown"),
        "start_sec": round(start_sec, 3),
        "end_sec": round(end_sec, 3),
        "window_sec": round(window_sec, 3),
        "unpadded_duration_sec": round(seg_dur, 3),
        "stft_frames_averaged": int(stft_frames),
        "feature_dim": int(len(vec)),
        "probability": prob,
        "calibrated_probability": calibrated_prob,
        "raw_score": raw_logit,
        "threshold": getattr(fakeprint_det, "threshold", 0.5),
        "periodicity_score": round(periodicity, 4),
        "temporal_stationarity": round(stationarity, 4),
        "active_bin_ratio": round(active_ratio, 4),
        "rms_dbfs": round(rms_dbfs, 2),
        "abstained": abstained,
        "ood_detected": ood_detected,
        "exploratory": True,
        "exploratory_notice": (
            "Segment-level Fakeprint is marked exploratory until validated on segment-level benchmarks; "
            "short-window STFT averaging has higher variance than full-track averaging."
        ),
        "status": status,
        "warnings": warnings,
    }


def _evaluate_long_context_alpha(
    sonics_alpha: Any,
    audio_16k: np.ndarray,
    sr_16k: int,
    duration_sec: float,
) -> Dict[str, Any]:
    """
    Evaluate SONICS SpecTTTra-alpha (120s) strictly as a long-context span prediction.
    Never misrepresents a 120s prediction as a 5s localized window.
    """
    if sonics_alpha is None or not sonics_alpha.is_available():
        return {
            "detector_id": "sonics_alpha_120s",
            "available": False,
            "abstained": True,
            "context_window_sec": 120.0,
            "minimum_required_duration_sec": 115.0,
            "recording_duration_sec": round(duration_sec, 3),
            "status_notice": "SONICS SpecTTTra-alpha (120s) weights not enabled or unavailable.",
            "long_context_windows": [],
        }

    samples_2d = audio_16k.reshape(1, -1)
    pred = sonics_alpha.predict(samples_2d, sr_16k)
    details = pred.details or {}
    abstained = bool(details.get("abstained", duration_sec < 115.0))
    win_scores = details.get("window_scores", {})

    long_windows: List[Dict[str, Any]] = []
    expected_samples = int(sr_16k * 120.0)
    total_len = len(audio_16k)

    if total_len <= int(expected_samples * 1.05):
        long_windows.append({
            "window_label": "full_or_zero_padded_120s_span",
            "start_sec": 0.0,
            "end_sec": round(duration_sec, 3),
            "context_span_sec": 120.0,
            "zero_pad_sec": round(max(0.0, 120.0 - duration_sec), 3),
            "probability": pred.probability,
            "raw_score": pred.raw_score,
            "abstained": abstained,
            "warning": details.get("ood_warning"),
        })
    else:
        rem = total_len - expected_samples
        center_st = max(0, rem // 2) / float(sr_16k)
        crop34_st = int(rem * 0.75) / float(sr_16k)
        crop14_st = int(rem * 0.25) / float(sr_16k)
        span_map = {
            "center": (center_st, center_st + 120.0),
            "crop_3_4": (crop34_st, crop34_st + 120.0),
            "crop_1_4": (crop14_st, crop14_st + 120.0),
        }
        for lbl, p_val in win_scores.items():
            s_t, e_t = span_map.get(lbl, (0.0, min(duration_sec, 120.0)))
            long_windows.append({
                "window_label": lbl,
                "start_sec": round(s_t, 3),
                "end_sec": round(min(duration_sec, e_t), 3),
                "context_span_sec": 120.0,
                "zero_pad_sec": 0.0,
                "probability": p_val,
                "abstained": abstained,
                "warning": None,
            })

    return {
        "detector_id": "sonics_alpha_120s",
        "detector_name": sonics_alpha.name,
        "model_version": sonics_alpha.version,
        "model_revision": sonics_alpha.model_revision,
        "available": True,
        "status": pred.status,
        "abstained": abstained,
        "predicted_label": pred.predicted_label,
        "aggregate_probability": pred.probability,
        "aggregate_raw_score": pred.raw_score,
        "context_window_sec": 120.0,
        "minimum_required_duration_sec": 115.0,
        "recording_duration_sec": round(duration_sec, 3),
        "representation_notice": (
            "SONICS SpecTTTra-alpha (120s) operates on a 120-second spectro-temporal context window. "
            "It is reported separately as a long-context span and is NEVER plotted as a localized 5-second trace."
        ),
        "ood_warning": details.get("ood_warning"),
        "long_context_windows": long_windows,
    }


def _classify_timeline_step(
    gamma_win: Dict[str, Any],
    fp_win: Optional[Dict[str, Any]],
    extra_win: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Assign one of the four neutral scientific labels to a timeline window step:
      - "Insufficient evidence"
      - "Model disagreement"
      - "Elevated AI-associated features"
      - "Low AI-associated features"
    """
    g_abstained = bool(gamma_win.get("abstained", True))
    g_prob = gamma_win.get("probability")
    g_thresh = float(gamma_win.get("threshold", 0.5))

    fp_abstained = True if fp_win is None else bool(fp_win.get("abstained", True))
    fp_prob = None if fp_win is None else fp_win.get("probability")
    fp_thresh = 0.5 if fp_win is None else float(fp_win.get("threshold", 0.5))

    ex_abstained = True if extra_win is None else bool(extra_win.get("abstained", True))
    ex_prob = None if extra_win is None else extra_win.get("probability")
    ex_thresh = 0.5 if extra_win is None else float(extra_win.get("threshold", 0.5))

    if g_abstained or g_prob is None:
        return {
            "region_label": LABEL_INSUFFICIENT,
            "disagreement": False,
            "disagreement_spread": None,
            "rationale": (
                "; ".join(gamma_win.get("warnings", []))
                or "Segment lacks valid signal energy or minimum unpadded duration."
            ),
        }

    # Check detector disagreement against exploratory Fakeprint or secondary detector when both are non-abstained
    secondary_probs: List[Tuple[str, float, float]] = []
    if not fp_abstained and fp_prob is not None:
        secondary_probs.append(("fakeprint_onnx (exploratory)", float(fp_prob), fp_thresh))
    if not ex_abstained and ex_prob is not None:
        secondary_probs.append((str(extra_win.get("detector_id", "secondary")), float(ex_prob), ex_thresh))

    g_high = float(g_prob) >= g_thresh
    disagreements: List[str] = []
    max_spread = 0.0

    for sec_name, s_prob, s_thresh in secondary_probs:
        s_high = s_prob >= s_thresh
        spread = abs(float(g_prob) - s_prob)
        max_spread = max(max_spread, spread)
        if (g_high != s_high and spread >= 0.35) or spread >= 0.50:
            disagreements.append(
                f"sonics_gamma_5s={float(g_prob):.3f} vs {sec_name}={s_prob:.3f} (spread={spread:.3f})"
            )

    fp_cal = None if fp_win is None else fp_win.get("calibrated_probability")
    fp_per = 0.0 if fp_win is None else float(fp_win.get("periodicity_score") or 0.0)
    fp_elevated = (fp_prob is not None and float(fp_prob) >= fp_thresh) or (
        fp_cal is not None and float(fp_cal) >= 0.42
    )

    # Because 5s segment-level Fakeprint is exploratory (trained on full-track averages with negative
    # broadband weights that depress 5s probabilities), strong SONICS-gamma 5s detections (>= 0.65, or
    # >= 0.50 with supporting comb periodicity) are labeled "Elevated AI-associated features" while
    # preserving the exact disagreement flag and spread in telemetry.
    if g_high and (float(g_prob) >= 0.65 or fp_elevated or fp_per >= 0.12 or not secondary_probs):
        return {
            "region_label": LABEL_ELEVATED,
            "disagreement": bool(disagreements),
            "disagreement_spread": round(max_spread, 4) if secondary_probs else None,
            "rationale": (
                f"SONICS SpecTTTra-gamma 5s window score ({float(g_prob):.3f}) is at or above threshold ({g_thresh:.2f})"
                + (f" ({'; '.join(disagreements)})." if disagreements else ".")
            ),
        }

    if disagreements or (not g_high and fp_elevated):
        return {
            "region_label": LABEL_DISAGREEMENT,
            "disagreement": True,
            "disagreement_spread": round(max_spread, 4),
            "rationale": f"Detectors diverge on this time window: {'; '.join(disagreements) or 'exploratory Fakeprint elevated while SONICS-gamma is low'}.",
        }

    return {
        "region_label": LABEL_LOW,
        "disagreement": False,
        "disagreement_spread": round(max_spread, 4) if secondary_probs else None,
        "rationale": (
            f"SONICS SpecTTTra-gamma 5s window score ({float(g_prob):.3f}) is below threshold ({g_thresh:.2f})."
        ),
    }


def _build_contiguous_region_summaries(
    timeline_steps: List[Dict[str, Any]],
    duration_sec: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Merge adjacent/overlapping windows with identical neutral classification labels into
    contiguous forensic regions and detect section transitions where detector scores shift.
    """
    if not timeline_steps:
        return [], []

    regions: List[Dict[str, Any]] = []
    transitions: List[Dict[str, Any]] = []

    current_group: List[Dict[str, Any]] = [timeline_steps[0]]

    for idx in range(1, len(timeline_steps)):
        prev_step = timeline_steps[idx - 1]
        curr_step = timeline_steps[idx]

        # Check for sharp score transition between consecutive valid windows
        p_prev = prev_step.get("sonics_gamma_probability")
        p_curr = curr_step.get("sonics_gamma_probability")
        if (
            p_prev is not None
            and p_curr is not None
            and not prev_step.get("abstained")
            and not curr_step.get("abstained")
        ):
            delta = float(p_curr) - float(p_prev)
            if abs(delta) >= 0.30 or prev_step["region_label"] != curr_step["region_label"]:
                transition_time = round(
                    0.5 * (float(prev_step["end_sec"]) + float(curr_step["start_sec"])), 3
                )
                direction = "increase" if delta > 0 else "decrease"
                transitions.append({
                    "transition_index": len(transitions) + 1,
                    "timestamp_sec": transition_time,
                    "previous_window": [prev_step["start_sec"], prev_step["end_sec"]],
                    "next_window": [curr_step["start_sec"], curr_step["end_sec"]],
                    "from_label": prev_step["region_label"],
                    "to_label": curr_step["region_label"],
                    "sonics_probability_before": round(float(p_prev), 4),
                    "sonics_probability_after": round(float(p_curr), 4),
                    "probability_delta": round(delta, 4),
                    "description": (
                        f"Localized detector score {direction} (Δ={delta:+.3f}) around {transition_time:.2f}s "
                        f"({prev_step['region_label']} → {curr_step['region_label']})."
                    ),
                })

        if curr_step["region_label"] == current_group[-1]["region_label"]:
            current_group.append(curr_step)
        else:
            regions.append(_summarize_region_group(current_group, len(regions) + 1))
            current_group = [curr_step]

    if current_group:
        regions.append(_summarize_region_group(current_group, len(regions) + 1))

    return regions, transitions


def _summarize_region_group(group: List[Dict[str, Any]], region_idx: int) -> Dict[str, Any]:
    start_sec = float(group[0]["start_sec"])
    end_sec = float(group[-1]["end_sec"])
    label = str(group[0]["region_label"])

    g_probs = [
        float(s["sonics_gamma_probability"])
        for s in group
        if s.get("sonics_gamma_probability") is not None
    ]
    fp_probs = [
        float(s["fakeprint_exploratory_probability"])
        for s in group
        if s.get("fakeprint_exploratory_probability") is not None
    ]
    rms_vals = [float(s["rms_dbfs"]) for s in group if s.get("rms_dbfs") is not None]
    all_warnings: List[str] = []
    for s in group:
        for w in s.get("warnings", []):
            if w not in all_warnings:
                all_warnings.append(w)

    return {
        "region_id": f"region_{region_idx}",
        "region_index": region_idx,
        "start_sec": round(start_sec, 3),
        "end_sec": round(end_sec, 3),
        "duration_sec": round(max(0.0, end_sec - start_sec), 3),
        "window_count": len(group),
        "region_label": label,
        "mean_sonics_probability": round(float(np.mean(g_probs)), 4) if g_probs else None,
        "max_sonics_probability": round(float(np.max(g_probs)), 4) if g_probs else None,
        "min_sonics_probability": round(float(np.min(g_probs)), 4) if g_probs else None,
        "mean_fakeprint_exploratory_probability": round(float(np.mean(fp_probs)), 4) if fp_probs else None,
        "mean_rms_dbfs": round(float(np.mean(rms_vals)), 2) if rms_vals else None,
        "disagreement_present": any(bool(s.get("disagreement")) for s in group),
        "abstained": all(bool(s.get("abstained")) for s in group),
        "scientific_interpretation": _neutral_region_explanation(label),
        "warnings": all_warnings,
    }


def _neutral_region_explanation(label: str) -> str:
    if label == LABEL_ELEVATED:
        return (
            "Windows in this time span exhibit elevated AI-associated spectro-temporal features above "
            "the detector threshold. This does not automatically identify specific vocal or instrument stems."
        )
    if label == LABEL_LOW:
        return (
            "Windows in this time span exhibit low AI-associated spectro-temporal features below the "
            "detector threshold. Note that continuous latent diffusion generators (ACE-Step, Stable Audio) "
            "may also score low on Suno/Udio-trained detectors."
        )
    if label == LABEL_DISAGREEMENT:
        return (
            "Evaluated detectors produce divergent scores in this region (e.g., high 1-8 kHz spectral hull "
            "residual alongside low spectro-temporal transformer score, or vice versa). Inspect segment audio "
            "for broadband distortion, mastering compression, or hybrid stem mixing."
        )
    return (
        "Signal conditions in this span (near-silence, fade-in/out, or unpadded duration < 3.0s) are "
        "insufficient for reliable classification."
    )


def compute_temporal_forensic_analysis(
    samples: np.ndarray,
    sr: int,
    asset_sha256: str,
    hop_sec: float = 2.5,
    fakeprint_window_sec: float = 5.0,
    include_exploratory_fakeprint: bool = True,
    provenance_context: Optional[Dict[str, Any]] = None,
    force_recompute: bool = False,
    job_id: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    sonics_window_sec: float = 5.0,
    preprocessing_version: str = DEFAULT_PREPROCESSING_VERSION,
    inference_settings: Optional[Dict[str, Any]] = None,
    step_delay_ms: float = 0.0,
) -> Dict[str, Any]:
    """
    Execute sliding-window temporal forensic analysis across the recording.

    - Decodes/resamples to 16,000 Hz mono once in memory.
    - Evaluates SONICS SpecTTTra-gamma (5.0s windows, configurable hop_sec) independently per window.
    - Evaluates Fakeprint ONNX in configurable segment-level mode (marked exploratory=True).
    - Evaluates any additional registered segment-capable detector (e.g., cross-generator spectral/phase detector).
    - Separately evaluates SONICS SpecTTTra-alpha (120s) long-context span predictions.
    - Produces contiguous neutral region summaries, transition events, and hybrid recording analysis.
    """
    t0 = time.perf_counter()
    hop_sec = float(np.clip(hop_sec, 0.5, 10.0))
    sonics_window_sec = float(np.clip(sonics_window_sec, 1.0, 30.0))
    fakeprint_window_sec = float(np.clip(fakeprint_window_sec, 2.5, 30.0))

    sonics_gamma = detector_registry.get("sonics_gamma_5s")
    fakeprint_det = detector_registry.get("fakeprint_onnx")
    sonics_alpha = detector_registry.get("sonics_alpha_120s")
    diffusion_det = detector_registry.get("latent_diffusion_forensic_v1")

    detector_revisions = {
        "sonics_gamma_5s": getattr(sonics_gamma, "model_revision", "none"),
        "fakeprint_onnx": getattr(fakeprint_det, "model_revision", "none"),
        "sonics_alpha_120s": getattr(sonics_alpha, "model_revision", "none"),
        "latent_diffusion_forensic_v1": getattr(diffusion_det, "model_revision", "none") if diffusion_det else "none",
    }

    cache_key, cache_key_components = compute_temporal_cache_key(
        asset_sha256=asset_sha256,
        hop_sec=hop_sec,
        fakeprint_window_sec=fakeprint_window_sec,
        include_exploratory_fakeprint=include_exploratory_fakeprint,
        detector_revisions=detector_revisions,
        sonics_window_sec=sonics_window_sec,
        preprocessing_version=preprocessing_version,
        inference_settings=inference_settings,
    )
    cache_file = _get_cache_dir() / f"{cache_key}.json"

    if not force_recompute and cache_file.exists():
        try:
            cached_data = json.loads(cache_file.read_text(encoding="utf-8"))
            cached_data["cache_hit"] = True
            cached_data["cache_key"] = cache_key
            cached_data["cache_key_components"] = cache_key_components
            return cached_data
        except Exception:
            pass

    if samples is None or samples.ndim != 2 or samples.shape[1] == 0 or sr <= 0:
        raise ValueError("Invalid audio PCM array or sample rate for temporal analysis.")

    duration_sec = float(samples.shape[1]) / float(sr)

    # Single conversion & resample to 16,000 Hz mono for 5s SONICS and 8192-FFT Fakeprint windows
    mono = (np.mean(samples, axis=0) if samples.shape[0] > 1 else samples[0]).astype(np.float32)
    sr_16k = 16000
    if sr != sr_16k:
        audio_16k = soxr.resample(mono, sr, sr_16k, quality="HQ").astype(np.float32)
    else:
        audio_16k = mono

    window_bounds = generate_sliding_window_bounds(
        duration_sec=duration_sec,
        window_sec=sonics_window_sec,
        hop_sec=hop_sec,
    )

    sonics_gamma_windows: List[Dict[str, Any]] = []
    fakeprint_exploratory_windows: List[Dict[str, Any]] = []
    cross_gen_windows: List[Dict[str, Any]] = []
    timeline_steps: List[Dict[str, Any]] = []

    total_wins = max(1, len(window_bounds))
    for idx, (w_start, w_end, is_incomplete) in enumerate(window_bounds):
        if is_temporal_job_cancelled(job_id):
            clear_temporal_job_cancel(job_id or "")
            raise RuntimeError(f"Temporal analysis job {job_id} was cancelled.")

        if step_delay_ms > 0.0:
            time.sleep(min(0.5, float(step_delay_ms) / 1000.0))
            if is_temporal_job_cancelled(job_id):
                clear_temporal_job_cancel(job_id or "")
                raise RuntimeError(f"Temporal analysis job {job_id} was cancelled.")

        if progress_callback:
            progress_callback(round((idx / total_wins) * 85.0, 1), f"Scoring window {idx + 1}/{total_wins}")

        g_res = _evaluate_sonics_gamma_window(
            sonics_gamma=sonics_gamma,
            audio_16k=audio_16k,
            sr_16k=sr_16k,
            start_sec=w_start,
            end_sec=w_end,
            is_incomplete_final=is_incomplete,
        )
        sonics_gamma_windows.append(g_res)

        if is_temporal_job_cancelled(job_id):
            clear_temporal_job_cancel(job_id or "")
            raise RuntimeError(f"Temporal analysis job {job_id} was cancelled.")

        fp_res: Optional[Dict[str, Any]] = None
        if include_exploratory_fakeprint:
            # Center or align the fakeprint segment around [w_start, w_end]
            if abs(fakeprint_window_sec - sonics_window_sec) <= 1e-3:
                fp_start, fp_end = w_start, w_end
            else:
                mid = 0.5 * (w_start + w_end)
                fp_start = max(0.0, round(mid - 0.5 * fakeprint_window_sec, 3))
                fp_end = min(round(duration_sec, 3), round(fp_start + fakeprint_window_sec, 3))
            fp_res = _evaluate_fakeprint_segment(
                fakeprint_det=fakeprint_det,
                audio_16k=audio_16k,
                sr_16k=sr_16k,
                start_sec=fp_start,
                end_sec=fp_end,
                window_sec=fakeprint_window_sec,
            )
            fakeprint_exploratory_windows.append(fp_res)

        cg_res: Optional[Dict[str, Any]] = None
        if diffusion_det is not None and hasattr(diffusion_det, "score_segment"):
            s_idx = max(0, int(round(w_start * sr)))
            e_idx = min(samples.shape[1], int(round(w_end * sr)))
            cg_res = diffusion_det.score_segment(samples[:, s_idx:e_idx], sr, w_start, w_end)
            cross_gen_windows.append(cg_res)

        step_cls = _classify_timeline_step(g_res, fp_res, cg_res)
        step_warnings = list(g_res.get("warnings", []))
        if fp_res:
            for w in fp_res.get("warnings", []):
                if "Exploratory segment-level" not in w and w not in step_warnings:
                    step_warnings.append(w)

        timeline_steps.append({
            "window_index": idx,
            "start_sec": g_res["start_sec"],
            "end_sec": g_res["end_sec"],
            "center_sec": round(0.5 * (g_res["start_sec"] + g_res["end_sec"]), 3),
            "unpadded_duration_sec": g_res["unpadded_duration_sec"],
            "zero_pad_sec": g_res["zero_pad_sec"],
            "is_incomplete_final_window": g_res["is_incomplete_final_window"],
            "sonics_gamma_probability": g_res["probability"],
            "sonics_gamma_logit": g_res["raw_score"],
            "fakeprint_exploratory_probability": fp_res["probability"] if fp_res else None,
            "fakeprint_calibrated_probability": fp_res["calibrated_probability"] if fp_res else None,
            "fakeprint_periodicity_score": fp_res["periodicity_score"] if fp_res else None,
            "cross_generator_probability": cg_res["probability"] if cg_res else None,
            "rms_dbfs": g_res["rms_dbfs"],
            "crest_factor_db": g_res["crest_factor_db"],
            "clipping_ratio": g_res["clipping_ratio"],
            "abstained": g_res["abstained"],
            "ood_detected": g_res["ood_detected"],
            "region_label": step_cls["region_label"],
            "disagreement": step_cls["disagreement"],
            "disagreement_spread": step_cls["disagreement_spread"],
            "rationale": step_cls["rationale"],
            "warnings": step_warnings,
        })

    if is_temporal_job_cancelled(job_id):
        clear_temporal_job_cancel(job_id or "")
        raise RuntimeError(f"Temporal analysis job {job_id} was cancelled.")

    # Long-context SONICS Alpha 120s evaluation (kept strictly separate from 5s timeline steps)
    long_context_alpha = _evaluate_long_context_alpha(
        sonics_alpha=sonics_alpha,
        audio_16k=audio_16k,
        sr_16k=sr_16k,
        duration_sec=duration_sec,
    )

    regions, transitions = _build_contiguous_region_summaries(timeline_steps, duration_sec)

    # Compute summary statistics across valid (non-abstained) 5s windows
    valid_g_probs = [
        float(w["probability"])
        for w in sonics_gamma_windows
        if w.get("probability") is not None and not w.get("abstained")
    ]
    valid_fp_probs = [
        float(w["probability"])
        for w in fakeprint_exploratory_windows
        if w.get("probability") is not None and not w.get("abstained")
    ]

    if valid_g_probs:
        g_arr = np.array(valid_g_probs, dtype=np.float64)
        top_k = min(3, len(valid_g_probs))
        top3_mean_val = float(np.mean(np.sort(g_arr)[-top_k:]))
        g_stats = {
            "valid_window_count": len(valid_g_probs),
            "total_window_count": len(sonics_gamma_windows),
            "top3_mean_probability": round(top3_mean_val, 6),
            "mean_probability": round(float(np.mean(g_arr)), 6),
            "median_probability": round(float(np.median(g_arr)), 6),
            "max_probability": round(float(np.max(g_arr)), 6),
            "min_probability": round(float(np.min(g_arr)), 6),
            "std_probability": round(float(np.std(g_arr)), 6),
            "iqr_probability": round(float(np.percentile(g_arr, 75) - np.percentile(g_arr, 25)), 6),
            "elevated_window_fraction": round(float(np.mean(g_arr >= 0.5)), 4),
        }
    else:
        g_stats = {
            "valid_window_count": 0,
            "total_window_count": len(sonics_gamma_windows),
            "top3_mean_probability": None,
            "mean_probability": None,
            "median_probability": None,
            "max_probability": None,
            "min_probability": None,
            "std_probability": None,
            "iqr_probability": None,
            "elevated_window_fraction": 0.0,
        }

    disagreement_windows = [s for s in timeline_steps if s.get("disagreement")]
    ood_windows = [s for s in timeline_steps if s.get("abstained") or s.get("ood_detected")]

    # Hybrid / heterogeneous recording analysis (Part A.4)
    std_val = g_stats["std_probability"] or 0.0
    elevated_count = sum(1 for s in timeline_steps if s["region_label"] == LABEL_ELEVATED)
    low_count = sum(1 for s in timeline_steps if s["region_label"] == LABEL_LOW)
    has_mixed_regimes = elevated_count > 0 and low_count > 0

    hybrid_analysis = {
        "known_provenance_facts": provenance_context or {
            "c2pa_verified": False,
            "metadata_origin_note": (
                "No cryptographically signed C2PA provenance manifest present; origin cannot be established "
                "from container tags alone."
            ),
        },
        "temporal_heterogeneity_metrics": {
            "inter_window_std": round(float(std_val), 4),
            "section_transition_count": len(transitions),
            "elevated_windows": elevated_count,
            "low_windows": low_count,
            "disagreement_windows": len(disagreement_windows),
            "insufficient_evidence_windows": len(ood_windows),
            "contains_mixed_feature_regimes": has_mixed_regimes,
        },
        "forensic_interpretation": (
            "Temporal trajectory exhibits mixed high-score and low-score segments across the timeline "
            f"(inter-window σ={std_val:.3f}, {len(transitions)} transition(s)). This pattern can arise in "
            "hybrid DAW productions/mashups combining human and synthetic sections, or in full-AI songs "
            "where sparse instrumental intros/outros carry weaker generator cues than dense vocal/chorus sections. "
            "Model predictions are probabilistic signal observations and do not identify individual stems."
            if has_mixed_regimes
            else (
                f"Temporal trajectory is relatively uniform across valid windows (inter-window σ={std_val:.3f}, "
                f"elevated={elevated_count}, low={low_count})."
            )
        ),
    }

    fakeprint_validation_info = {
        "exploratory": True,
        "stft_n_fft": 8192,
        "sample_rate_hz": 16000,
        "fft_frame_duration_sec": 0.512,
        "fft_hop_duration_sec": 0.256,
        "configured_segment_window_sec": fakeprint_window_sec,
        "frames_per_configured_window": max(1, 1 + max(0, int(fakeprint_window_sec * 16000) - 8192) // 4096),
        "feature_dimension": 3585,
        "validation_note": (
            "Fakeprint ONNX was trained on full-track time-averaged 8192-point STFT log-power spectra "
            "(up to 300s). While 5s-10s windows preserve the exact 3,585 frequency bins in 1-8 kHz, "
            "short-window averaging over 19-38 STFT frames exhibits higher variance to transient percussion "
            "and guitar distortion than full-track averaging. Segment-level Fakeprint scores are therefore "
            "labeled exploratory."
        ),
    }

    score_computation_methodology = {
        "sonics_gamma_window_probability": (
            "p_i = sigmoid(z_i) = 1 / (1 + exp(-z_i)), where z_i is the scalar logit output of "
            "SONICS SpecTTTra-gamma (hf_hub:awsaf49/sonics-spectttra-gamma-5s) evaluated on an independently "
            "unit-variance normalized 5.0s (80,000-sample @ 16 kHz mono) window x_norm = x / max(std(x), 1e-6)."
        ),
        "sonics_top3_mean_probability": (
            "Mean of the top k = min(3, N_valid) non-abstained window probabilities p_i across the track."
        ),
        "fakeprint_exploratory_segment_probability": (
            "p_fp = sigmoid(w^T r + b) from the ONNX logistic regression head (lofcz/ai-music-detector) "
            "evaluated on the 3,585-bin (1,000-8,000 Hz) 8192-point STFT lower-hull morphological residual r "
            "of the segment. Calibrated probability is sigmoid(logit(p_fp) / 3.25). Marked exploratory=True."
        ),
        "rms_and_crest_factor_dbfs": (
            "RMS_dBFS = 20 * log10(max(sqrt(mean(x^2)), 1e-10)); "
            "Crest_dB = 20 * log10(max(|x|) / max(sqrt(mean(x^2)), 1e-10))."
        ),
        "segment_snr_and_lsd_db": (
            "SNR_dB = 10 * log10(mean(x^2) / mean((y - x)^2)); "
            "LSD_dB = sqrt(mean((20*log10(|STFT(x)|) - 20*log10(|STFT(y)|))^2))."
        ),
    }

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    result: Dict[str, Any] = {
        "schema_version": "phase6_temporal_v2",
        "asset_sha256": asset_sha256,
        "recording_duration_sec": round(duration_sec, 3),
        "processing_time_ms": elapsed_ms,
        "cache_hit": False,
        "cache_key": cache_key,
        "cache_key_components": cache_key_components,
        "configuration": {
            "sonics_gamma_window_sec": sonics_window_sec,
            "hop_sec": hop_sec,
            "fakeprint_segment_window_sec": fakeprint_window_sec,
            "include_exploratory_fakeprint": include_exploratory_fakeprint,
            "sonics_alpha_context_sec": 120.0,
            "preprocessing_version": preprocessing_version,
            "incomplete_final_window_policy": (
                "Upstream mode='constant' zero-padding to 5.0s; segments with < 3.0s unpadded audio abstain."
            ),
            "per_window_normalization": (
                "Independent unit-variance normalization (window /= max(std(window), 1e-6)) and (1, T) "
                "inference per window to prevent cross-window top_db mel contamination."
            ),
        },
        "score_computation_methodology": score_computation_methodology,
        "detector_revisions": detector_revisions,
        "summary_statistics": {
            "sonics_gamma_5s": g_stats,
            "fakeprint_exploratory_mean_prob": (
                round(float(np.mean(valid_fp_probs)), 6) if valid_fp_probs else None
            ),
            "disagreement_window_count": len(disagreement_windows),
            "ood_or_abstained_window_count": len(ood_windows),
            "region_count": len(regions),
            "transition_count": len(transitions),
        },
        "timeline_steps": timeline_steps,
        "per_detector_windows": {
            "sonics_gamma_5s": sonics_gamma_windows,
            "fakeprint_onnx_exploratory": fakeprint_exploratory_windows,
            "cross_generator_detector": cross_gen_windows,
        },
        "long_context_predictions": long_context_alpha,
        "fakeprint_segment_validation": fakeprint_validation_info,
        "region_summaries": regions,
        "section_transitions": transitions,
        "hybrid_forensic_analysis": hybrid_analysis,
        "scientific_notices": [
            "Region labels ('Elevated AI-associated features', 'Low AI-associated features', 'Model disagreement', "
            "'Insufficient evidence') describe localized model responses and do not automatically identify "
            "synthetic vocals, human vocals, or specific generator stems.",
            "SONICS SpecTTTra-alpha (120s) requires >= 115s of audio context and is reported separately from "
            "the 5-second sliding-window timeline.",
            "Segment-level Fakeprint ONNX traces are marked exploratory because the upstream logistic regression "
            "weights were fitted on full-track STFT averages.",
        ],
    }

    if is_temporal_job_cancelled(job_id):
        clear_temporal_job_cancel(job_id or "")
        raise RuntimeError(f"Temporal analysis job {job_id} was cancelled before cache commit.")

    tmp_file = cache_file.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp_file.write_text(json.dumps(result), encoding="utf-8")
        if is_temporal_job_cancelled(job_id):
            tmp_file.unlink(missing_ok=True)
            clear_temporal_job_cancel(job_id or "")
            raise RuntimeError(f"Temporal analysis job {job_id} was cancelled before cache commit.")
        tmp_file.replace(cache_file)
    except RuntimeError:
        raise
    except Exception:
        pass
    finally:
        try:
            if tmp_file.exists():
                tmp_file.unlink(missing_ok=True)
        except Exception:
            pass

    if progress_callback:
        progress_callback(100.0, "Temporal analysis complete")

    return result


def compute_temporal_dsp_comparison(
    orig_samples: np.ndarray,
    orig_sr: int,
    orig_sha256: str,
    proc_samples: np.ndarray,
    proc_sr: int,
    proc_sha256: str,
    alignment_offset_samples: int = 0,
    hop_sec: float = 2.5,
    include_exploratory_fakeprint: bool = True,
) -> Dict[str, Any]:
    """
    Compare original and DSP-processed audio across aligned time segments (Phase 5 Part A.5).

    Computes per-segment:
    - Original and processed detector scores (SONICS Gamma 5s and exploratory Fakeprint ONNX)
    - Score deltas (processed - original)
    - Segment SNR (dB), Log-Spectral Distance (LSD dB), RMS delta (dB), and 1-8 kHz periodicity delta
    - Segment-level audio quality warnings (e.g. SNR < 15 dB, clipping introduced)
    """
    t0 = time.perf_counter()

    # Align mono waveforms at orig_sr for per-segment spectral and SNR comparison
    orig_mono = (np.mean(orig_samples, axis=0) if orig_samples.shape[0] > 1 else orig_samples[0]).astype(np.float32)
    proc_mono = (np.mean(proc_samples, axis=0) if proc_samples.shape[0] > 1 else proc_samples[0]).astype(np.float32)
    if proc_sr != orig_sr:
        proc_mono = soxr.resample(proc_mono, proc_sr, orig_sr, quality="HQ").astype(np.float32)

    if alignment_offset_samples > 0 and alignment_offset_samples < len(proc_mono):
        proc_aligned = proc_mono[alignment_offset_samples:]
        m = min(len(orig_mono), len(proc_aligned))
        orig_al, proc_al = orig_mono[:m], proc_aligned[:m]
    elif alignment_offset_samples < 0 and abs(alignment_offset_samples) < len(orig_mono):
        lag = abs(alignment_offset_samples)
        orig_aligned = orig_mono[lag:]
        m = min(len(orig_aligned), len(proc_mono))
        orig_al, proc_al = orig_aligned[:m], proc_mono[:m]
    else:
        m = min(len(orig_mono), len(proc_mono))
        orig_al, proc_al = orig_mono[:m], proc_mono[:m]

    aligned_duration_sec = float(len(orig_al)) / float(orig_sr) if orig_sr > 0 else 0.0

    # Run or load cached temporal forensic analysis on aligned signals
    orig_temp = compute_temporal_forensic_analysis(
        samples=orig_al.reshape(1, -1),
        sr=orig_sr,
        asset_sha256=f"{orig_sha256}_al_{alignment_offset_samples}",
        hop_sec=hop_sec,
        fakeprint_window_sec=5.0,
        include_exploratory_fakeprint=include_exploratory_fakeprint,
    )
    proc_temp = compute_temporal_forensic_analysis(
        samples=proc_al.reshape(1, -1),
        sr=orig_sr,
        asset_sha256=f"{proc_sha256}_al_{alignment_offset_samples}",
        hop_sec=hop_sec,
        fakeprint_window_sec=5.0,
        include_exploratory_fakeprint=include_exploratory_fakeprint,
    )

    orig_steps = orig_temp.get("timeline_steps", [])
    proc_steps = proc_temp.get("timeline_steps", [])
    n_steps = min(len(orig_steps), len(proc_steps))

    comparison_segments: List[Dict[str, Any]] = []
    sonics_deltas: List[float] = []
    fp_deltas: List[float] = []
    segment_lsds: List[float] = []
    segment_snrs: List[float] = []
    quality_warnings: List[str] = []

    for idx in range(n_steps):
        o_s = orig_steps[idx]
        p_s = proc_steps[idx]
        w_start = float(o_s["start_sec"])
        w_end = float(o_s["end_sec"])

        s_idx = max(0, int(round(w_start * orig_sr)))
        e_idx = min(len(orig_al), int(round(w_end * orig_sr)))
        x_seg = orig_al[s_idx:e_idx]
        y_seg = proc_al[s_idx:e_idx]

        # Segment SNR & Log-Spectral Distance
        if len(x_seg) >= 512:
            noise = y_seg - x_seg
            sig_pwr = float(np.mean(x_seg ** 2)) + EPS
            nse_pwr = float(np.mean(noise ** 2)) + EPS
            seg_snr_db = round(float(10.0 * np.log10(sig_pwr / nse_pwr)), 2)

            S_x = np.abs(librosa.stft(x_seg, n_fft=1024, hop_length=256))
            S_y = np.abs(librosa.stft(y_seg, n_fft=1024, hop_length=256))
            log_x = 20.0 * np.log10(np.clip(S_x, 1e-7, None))
            log_y = 20.0 * np.log10(np.clip(S_y, 1e-7, None))
            seg_lsd_db = round(float(np.sqrt(np.mean((log_x - log_y) ** 2))), 3)
        else:
            seg_snr_db = 0.0
            seg_lsd_db = 0.0

        segment_snrs.append(seg_snr_db)
        segment_lsds.append(seg_lsd_db)

        o_g = o_s.get("sonics_gamma_probability")
        p_g = p_s.get("sonics_gamma_probability")
        delta_g = round(float(p_g - o_g), 6) if (o_g is not None and p_g is not None) else None
        if delta_g is not None and not o_s.get("abstained"):
            sonics_deltas.append(delta_g)

        o_fp = o_s.get("fakeprint_exploratory_probability")
        p_fp = p_s.get("fakeprint_exploratory_probability")
        delta_fp = round(float(p_fp - o_fp), 6) if (o_fp is not None and p_fp is not None) else None
        if delta_fp is not None and not o_s.get("abstained"):
            fp_deltas.append(delta_fp)

        o_per = o_s.get("fakeprint_periodicity_score")
        p_per = p_s.get("fakeprint_periodicity_score")
        delta_per = round(float(p_per - o_per), 4) if (o_per is not None and p_per is not None) else None

        rms_delta = round(float(p_s["rms_dbfs"] - o_s["rms_dbfs"]), 2)

        seg_warnings: List[str] = []
        if seg_snr_db < 15.0 and o_s["rms_dbfs"] > -50.0:
            w_msg = f"Window [{w_start:.1f}s-{w_end:.1f}s]: Low segment SNR ({seg_snr_db:.1f} dB < 15.0 dB) indicates audible signal degradation."
            seg_warnings.append(w_msg)
            if len(quality_warnings) < 5:
                quality_warnings.append(w_msg)
        if p_s["clipping_ratio"] >= 0.005 and o_s["clipping_ratio"] < 0.005:
            w_msg = f"Window [{w_start:.1f}s-{w_end:.1f}s]: Processing introduced digital clipping ({p_s['clipping_ratio']*100:.2f}%)."
            seg_warnings.append(w_msg)
            if w_msg not in quality_warnings:
                quality_warnings.append(w_msg)

        comparison_segments.append({
            "window_index": idx,
            "start_sec": w_start,
            "end_sec": w_end,
            "center_sec": o_s["center_sec"],
            "original_sonics_probability": o_g,
            "processed_sonics_probability": p_g,
            "delta_sonics_probability": delta_g,
            "original_fakeprint_probability": o_fp,
            "processed_fakeprint_probability": p_fp,
            "delta_fakeprint_probability": delta_fp,
            "original_periodicity_score": o_per,
            "processed_periodicity_score": p_per,
            "delta_periodicity_score": delta_per,
            "original_region_label": o_s["region_label"],
            "processed_region_label": p_s["region_label"],
            "segment_snr_db": seg_snr_db,
            "segment_lsd_db": seg_lsd_db,
            "segment_rms_delta_db": rms_delta,
            "abstained": bool(o_s.get("abstained") or p_s.get("abstained")),
            "quality_warnings": seg_warnings,
        })

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {
        "schema_version": "phase5_temporal_comparison_v1",
        "aligned_duration_sec": round(aligned_duration_sec, 3),
        "alignment_offset_samples": int(alignment_offset_samples),
        "alignment_offset_ms": round((alignment_offset_samples / float(orig_sr)) * 1000.0, 3) if orig_sr > 0 else 0.0,
        "hop_sec": hop_sec,
        "window_sec": 5.0,
        "segment_count": len(comparison_segments),
        "processing_time_ms": elapsed_ms,
        "summary": {
            "mean_delta_sonics_probability": round(float(np.mean(sonics_deltas)), 6) if sonics_deltas else None,
            "max_abs_delta_sonics_probability": round(float(np.max(np.abs(sonics_deltas))), 6) if sonics_deltas else None,
            "mean_delta_fakeprint_probability": round(float(np.mean(fp_deltas)), 6) if fp_deltas else None,
            "mean_segment_snr_db": round(float(np.mean(segment_snrs)), 2) if segment_snrs else None,
            "mean_segment_lsd_db": round(float(np.mean(segment_lsds)), 3) if segment_lsds else None,
            "quality_warning_count": len(quality_warnings),
        },
        "segments": comparison_segments,
        "quality_warnings": quality_warnings,
        "scientific_disclaimer": (
            "Detector-score changes across time segments measure the sensitivity of specific model "
            "feature representations to the applied DSP perturbation. Reductions in detector probability "
            "do NOT demonstrate removal of cryptographic provenance or content-origin evidence."
        ),
    }
