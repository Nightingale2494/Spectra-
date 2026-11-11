"""Original-versus-Processed Audio Comparison Engine for SPECTRA."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
from scipy import signal
import soxr
from sqlalchemy.orm import Session

from app.database.models import AudioAsset, ComparisonResult, DetectorRun, ProcessingExperiment
from app.detectors.registry import detector_registry
from app.services.forensics_service import (
    compute_frequency_domain_metrics,
    compute_time_domain_metrics,
    extract_neural_vocoder_fakeprint,
)
from app.services.temporal_service import compute_temporal_dsp_comparison

EPS = 1e-10


def _align_signals(orig_mono: np.ndarray, proc_mono: np.ndarray, max_lag: int = 2048) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Estimate and compensate for filter/codec latency between original and processed mono signals
    using cross-correlation on a central segment before computing sample-level SNR/spectral distance.
    """
    n = min(len(orig_mono), len(proc_mono))
    if n < 256:
        return orig_mono[:n], proc_mono[:n], 0

    # Use up to 65,536 central samples for fast cross-correlation lag estimation
    seg_len = min(n, 65536)
    start = (n - seg_len) // 2
    x = orig_mono[start : start + seg_len]
    y = proc_mono[start : start + seg_len]

    corr = signal.correlate(y, x, mode="full", method="fft")
    lags = signal.correlation_lags(len(y), len(x), mode="full")
    valid_mask = np.abs(lags) <= max_lag
    if np.any(valid_mask):
        sub_corr = corr[valid_mask]
        sub_lags = lags[valid_mask]
        best_lag = int(sub_lags[int(np.argmax(np.abs(sub_corr)))])
    else:
        best_lag = 0

    if best_lag > 0:
        y_aligned = proc_mono[best_lag:]
        m = min(len(orig_mono), len(y_aligned))
        return orig_mono[:m], y_aligned[:m], best_lag
    elif best_lag < 0:
        lag_abs = abs(best_lag)
        x_aligned = orig_mono[lag_abs:]
        m = min(len(x_aligned), len(proc_mono))
        return x_aligned[:m], proc_mono[:m], best_lag
    else:
        return orig_mono[:n], proc_mono[:n], 0


def compute_before_after_comparison(
    db: Session,
    experiment: ProcessingExperiment,
    source_asset: AudioAsset,
    processed_asset: AudioAsset,
    orig_samples: np.ndarray,
    orig_sr: int,
    proc_samples: np.ndarray,
    proc_sr: int,
    run_detectors: bool = True,
) -> ComparisonResult:
    """
    Compute objective acoustic, spectral, fakeprint residual, and detector score deltas
    between original and processed audio, accounting for latency and gain alignment.
    """
    orig_td = compute_time_domain_metrics(orig_samples, orig_sr)
    proc_td = compute_time_domain_metrics(proc_samples, proc_sr)

    delta_rms_db = round(float(proc_td["rms_dbfs"] - orig_td["rms_dbfs"]), 3)
    delta_peak_db = round(float(proc_td["peak_dbfs"] - orig_td["peak_dbfs"]), 3)
    loudness_delta_db = delta_rms_db
    duration_delta_sec = round(float(proc_td["duration_seconds"] - orig_td["duration_seconds"]), 4)

    # Convert both to mono at orig_sr for aligned comparison
    orig_mono = (np.mean(orig_samples, axis=0) if orig_samples.shape[0] > 1 else orig_samples[0]).astype(np.float32)
    proc_mono = (np.mean(proc_samples, axis=0) if proc_samples.shape[0] > 1 else proc_samples[0]).astype(np.float32)
    if proc_sr != orig_sr:
        proc_mono = soxr.resample(proc_mono, proc_sr, orig_sr, quality="HQ").astype(np.float32)

    x_al, y_al, offset_samples = _align_signals(orig_mono, proc_mono)

    # Gain-compensated error for true structural SNR / PSNR alongside raw SNR
    noise = y_al - x_al
    sig_power = float(np.mean(x_al ** 2)) + EPS
    noise_power = float(np.mean(noise ** 2)) + EPS
    snr_db = round(float(10.0 * np.log10(sig_power / noise_power)), 2)
    peak_power = float(np.max(x_al ** 2)) + EPS
    psnr_db = round(float(10.0 * np.log10(peak_power / noise_power)), 2)

    # STFT spectral convergence & log-spectral L2 distance
    n_fft = 2048
    hop = 512
    S_orig = np.abs(librosa.stft(x_al, n_fft=n_fft, hop_length=hop))
    S_proc = np.abs(librosa.stft(y_al, n_fft=n_fft, hop_length=hop))
    min_frames = min(S_orig.shape[1], S_proc.shape[1])
    S_orig = S_orig[:, :min_frames]
    S_proc = S_proc[:, :min_frames]

    spec_conv = float(np.linalg.norm(S_orig - S_proc, ord="fro") / (np.linalg.norm(S_orig, ord="fro") + EPS))
    log_S_orig = 20.0 * np.log10(np.clip(S_orig, 1e-7, None))
    log_S_proc = 20.0 * np.log10(np.clip(S_proc, 1e-7, None))
    lsd_db = float(np.sqrt(np.mean((log_S_orig - log_S_proc) ** 2)))

    # Full-band spectrum and 1-8 kHz fakeprint residual comparison curves
    orig_fd = compute_frequency_domain_metrics(orig_samples, orig_sr)
    proc_fd = compute_frequency_domain_metrics(proc_samples, proc_sr)
    orig_fp = extract_neural_vocoder_fakeprint(orig_samples, orig_sr)
    proc_fp = extract_neural_vocoder_fakeprint(proc_samples, proc_sr)

    spectrum_comparison = {
        "frequencies_hz": orig_fd["spectrum_curve"]["frequencies_hz"],
        "original_power_db": orig_fd["spectrum_curve"]["mean_power_db"],
        "processed_power_db": proc_fd["spectrum_curve"]["mean_power_db"][: len(orig_fd["spectrum_curve"]["frequencies_hz"])],
        "original_centroid_hz": orig_fd["spectral_centroid_hz_mean"],
        "processed_centroid_hz": proc_fd["spectral_centroid_hz_mean"],
        "original_flatness_db": orig_fd["spectral_flatness_db"],
        "processed_flatness_db": proc_fd["spectral_flatness_db"],
    }

    # Temporal segment-by-segment score & spectral delta comparison (Phase 5 Part A.5)
    default_hop = 2.5 if orig_td["duration_seconds"] <= 90.0 else 5.0
    temporal_comparison = compute_temporal_dsp_comparison(
        orig_samples=orig_samples,
        orig_sr=orig_sr,
        orig_sha256=source_asset.sha256_checksum,
        proc_samples=proc_samples,
        proc_sr=proc_sr,
        proc_sha256=processed_asset.sha256_checksum,
        alignment_offset_samples=offset_samples,
        hop_sec=default_hop,
        include_exploratory_fakeprint=True,
    )

    residual_comparison = {
        "frequencies_hz": orig_fp["chart_data"]["frequencies_hz"],
        "original_residual_norm": orig_fp["chart_data"]["residual_norm"],
        "processed_residual_norm": proc_fp["chart_data"]["residual_norm"],
        "original_periodicity_score": orig_fp["periodicity_score"],
        "processed_periodicity_score": proc_fp["periodicity_score"],
        "temporal_comparison": temporal_comparison,
    }

    # Detector score deltas (Original vs Processed)
    detector_deltas: List[Dict[str, Any]] = []
    if run_detectors:
        orig_preds = detector_registry.run_all(orig_samples, orig_sr, precomputed_fakeprint=orig_fp["feature_vector"])
        proc_preds = detector_registry.run_all(proc_samples, proc_sr, precomputed_fakeprint=proc_fp["feature_vector"])

        for o_p, p_p in zip(orig_preds, proc_preds):
            # Persist DetectorRun for processed asset as well
            db_run = DetectorRun(
                job_id=experiment.job_id,
                audio_asset_id=processed_asset.id,
                detector_id=p_p.detector_id,
                detector_name=p_p.detector_name,
                detector_version=p_p.detector_version,
                model_source=p_p.model_source,
                model_revision=p_p.model_revision,
                probability=p_p.probability,
                raw_score=p_p.raw_score,
                threshold=p_p.threshold,
                predicted_label=p_p.predicted_label,
                confidence=p_p.confidence,
                input_sample_rate=p_p.input_sample_rate,
                input_duration_sec=p_p.input_duration_sec,
                processing_time_ms=p_p.processing_time_ms,
                device_used=p_p.device_used,
                feature_dim=p_p.feature_dim,
                status=p_p.status,
                error_message=p_p.error_message,
                details_json=json.dumps(p_p.details),
            )
            db.add(db_run)

            delta_prob = (
                round(float(p_p.probability - o_p.probability), 6)
                if (o_p.probability is not None and p_p.probability is not None)
                else None
            )
            detector_deltas.append({
                "detector_id": o_p.detector_id,
                "detector_name": o_p.detector_name,
                "status": p_p.status,
                "original_probability": o_p.probability,
                "processed_probability": p_p.probability,
                "before_probability": o_p.probability,
                "after_probability": p_p.probability,
                "delta_probability": delta_prob,
                "original_label": o_p.predicted_label,
                "processed_label": p_p.predicted_label,
                "before_label": o_p.predicted_label,
                "after_label": p_p.predicted_label,
                "threshold": o_p.threshold,
            })

    observations: List[str] = [
        f"Aligned timing offset: {offset_samples} samples ({offset_samples / orig_sr * 1000:.2f} ms).",
        f"Log-spectral distance (LSD): {lsd_db:.2f} dB; Spectral convergence: {spec_conv:.4f}; SNR: {snr_db:.2f} dB.",
        f"1-8 kHz spectral residual periodicity changed from {orig_fp['periodicity_score']:.3f} to {proc_fp['periodicity_score']:.3f}.",
    ]
    warnings: List[str] = [
        "Scientific caveat: A reduction in a detector's probability score indicates sensitivity of that specific model's feature representation to the applied DSP transformation, NOT universal removal of generative provenance."
    ]
    if proc_td["clipping_detected"] and not orig_td["clipping_detected"]:
        warnings.append("Processing introduced digital clipping that was not present in the original recording.")

    comp = ComparisonResult(
        experiment_id=experiment.id,
        source_asset_id=source_asset.id,
        processed_asset_id=processed_asset.id,
        delta_rms_db=delta_rms_db,
        delta_peak_db=delta_peak_db,
        spectral_distance_l2=round(lsd_db, 4),
        spectral_convergence=round(spec_conv, 5),
        snr_db=snr_db,
        psnr_db=psnr_db,
        loudness_delta_db=loudness_delta_db,
        duration_delta_sec=duration_delta_sec,
        alignment_offset_samples=offset_samples,
        original_clipping_ratio=orig_td["clipping_ratio"],
        processed_clipping_ratio=proc_td["clipping_ratio"],
        detector_deltas_json=json.dumps(detector_deltas),
        spectrum_comparison_json=json.dumps(spectrum_comparison),
        residual_comparison_json=json.dumps(residual_comparison),
        observations_json=json.dumps(observations),
        warnings_json=json.dumps(warnings),
    )
    db.add(comp)
    db.commit()
    db.refresh(comp)
    return comp
