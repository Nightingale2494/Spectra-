"""Scientific DSP Processing Laboratory for SPECTRA using NumPy/SciPy and optional Pedalboard/PyAV."""
from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import av
import librosa
import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d
import soundfile as sf
import soxr
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.models import AnalysisJob, AudioAsset, ProcessingExperiment
from app.services.ingestion_service import decode_audio_pcm, ingest_uploaded_bytes, resolve_asset_path

EPS = 1e-10

DSP_PRESETS: List[Dict[str, Any]] = [
    {
        "id": "vocoder_band_smoothing",
        "name": "Spectral Residual Smoothing (1–8 kHz)",
        "operation_type": "spectral_smoothing",
        "engine": "scipy_dsp",
        "description": "Attenuates sharp fine-grained periodic comb peaks in the 1-8 kHz band via STFT log-magnitude envelope smoothing while preserving phase and broadband timbre.",
        "parameters": {
            "freq_min_hz": 1000.0,
            "freq_max_hz": 8000.0,
            "smoothing_bins": 9,
            "mix": 0.65,
            "n_fft": 4096,
        },
    },
    {
        "id": "hf_rolloff_7500",
        "name": "High-Frequency Rolloff Filter (7.5 kHz Low-Pass)",
        "operation_type": "lowpass",
        "engine": "scipy_dsp",
        "description": "Applies a zero-phase 6th-order Butterworth low-pass filter at 7,500 Hz to evaluate detector sensitivity to high-frequency bandwidth truncation.",
        "parameters": {"cutoff_hz": 7500.0, "order": 6},
    },
    {
        "id": "sub_rumble_highpass",
        "name": "Sub-Bass Rumble Removal (80 Hz High-Pass)",
        "operation_type": "highpass",
        "engine": "scipy_dsp",
        "description": "Applies a 4th-order Butterworth high-pass filter at 80 Hz to clean low-frequency DC offset and sub-bass rumble.",
        "parameters": {"cutoff_hz": 80.0, "order": 4},
    },
    {
        "id": "midband_focus_bp",
        "name": "Mid-Band Isolation (300–5,000 Hz Band-Pass)",
        "operation_type": "bandpass",
        "engine": "scipy_dsp",
        "description": "Isolates core vocal and harmonic mid-band energy between 300 Hz and 5,000 Hz.",
        "parameters": {"low_hz": 300.0, "high_hz": 5000.0, "order": 4},
    },
    {
        "id": "parametric_notch_4k",
        "name": "Parametric Upper-Mid Attenuation (4 kHz, -4.5 dB)",
        "operation_type": "parametric_eq",
        "engine": "scipy_dsp",
        "description": "Applies a surgical peaking biquad filter centered at 4,000 Hz to test localized spectral modification.",
        "parameters": {"center_hz": 4000.0, "q_factor": 2.5, "gain_db": -4.5},
    },
    {
        "id": "mastering_3band_eq",
        "name": "Three-Band Mastering Tilt EQ",
        "operation_type": "eq_3band",
        "engine": "scipy_dsp",
        "description": "Applies gentle low-shelf warmth (+1.5 dB @ 120 Hz), neutral mids, and high-shelf air attenuation (-2.0 dB @ 6,500 Hz).",
        "parameters": {
            "low_gain_db": 1.5,
            "low_freq_hz": 120.0,
            "mid_gain_db": -0.5,
            "mid_freq_hz": 1500.0,
            "mid_q": 0.8,
            "high_gain_db": -2.0,
            "high_freq_hz": 6500.0,
        },
    },
    {
        "id": "freq_selective_decomb",
        "name": "Frequency-Selective Comb Peak Damping (1–8 kHz)",
        "operation_type": "frequency_selective",
        "engine": "scipy_dsp",
        "description": "Suppresses narrowband spectral spikes exceeding local median envelope in the 1-8 kHz region via COLA STFT/ISTFT reconstruction.",
        "parameters": {
            "freq_min_hz": 1000.0,
            "freq_max_hz": 8000.0,
            "spike_threshold_db": 2.2,
            "suppression_ratio": 0.6,
            "n_fft": 4096,
        },
    },
    {
        "id": "broadband_denoise",
        "name": "Broadband Spectral Floor Denoising",
        "operation_type": "noise_reduction",
        "engine": "scipy_dsp",
        "description": "Estimates stationary background floor from lowest-energy frames and applies soft Wiener spectral gating.",
        "parameters": {"reduction_db": 10.0, "sensitivity": 1.2, "n_fft": 2048},
    },
    {
        "id": "broadcast_compressor",
        "name": "Dynamic Range Compression (-16 dBFS, 3:1)",
        "operation_type": "dynamic_compression",
        "engine": "scipy_dsp",
        "description": "Applies feed-forward RMS envelope compression to evaluate detector robustness under mastering dynamics processing.",
        "parameters": {
            "threshold_db": -16.0,
            "ratio": 3.0,
            "attack_ms": 12.0,
            "release_ms": 120.0,
            "makeup_gain_db": 2.0,
        },
    },
    {
        "id": "codec_mp3_128k",
        "name": "Lossy Codec Transcode Simulation (MP3 128 kbps)",
        "operation_type": "codec_simulation",
        "engine": "scipy_dsp",
        "description": "Encodes and decodes audio through a real 128 kbps MP3 codec pipeline to measure codec robustness of spectral detectors.",
        "parameters": {"codec": "mp3", "bitrate_kbps": 128},
    },
    {
        "id": "vocoder_comb_notch_suppression",
        "name": "Harmonic Comb Notch Filter (400 Hz Multiples in 1.6–7.6 kHz)",
        "operation_type": "notch_comb",
        "engine": "scipy_dsp",
        "description": "Applies surgical zero-phase IIR notch filters at periodic comb multiples in the 1.6–7.6 kHz region to test narrowband comb peak suppression.",
        "parameters": {"fundamental_hz": 400.0, "min_freq_hz": 1600.0, "max_freq_hz": 7600.0, "q": 30.0},
    },
    {
        "id": "pedalboard_studio_chain",
        "name": "Spotify Pedalboard Mastering Chain (HPF + EQ + Comp + Limiter)",
        "operation_type": "pedalboard_chain",
        "engine": "pedalboard",
        "description": "Executes a multi-stage Spotify Pedalboard C++ chain (80 Hz High-Pass, -2.5 dB 4.5 kHz Peak EQ, 2.5:1 Compressor, -0.5 dBFS Limiter).",
        "parameters": {
            "highpass_hz": 80.0,
            "peak_freq_hz": 4500.0,
            "peak_gain_db": -2.5,
            "threshold_db": -16.0,
            "ratio": 2.5,
            "ceiling_db": -0.5,
        },
    },
]


def _design_biquad_peaking(center_hz: float, sr: int, q: float, gain_db: float) -> Tuple[np.ndarray, np.ndarray]:
    """Audio EQ Cookbook peaking biquad filter coefficients."""
    nyq = 0.5 * sr
    center_hz = float(np.clip(center_hz, 20.0, nyq * 0.95))
    q = float(max(0.1, q))
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * center_hz / float(sr)
    alpha = np.sin(w0) / (2.0 * q)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha / A

    b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return b, a


def _design_biquad_shelf(cutoff_hz: float, sr: int, gain_db: float, mode: str = "low") -> Tuple[np.ndarray, np.ndarray]:
    """Audio EQ Cookbook low-shelf and high-shelf biquad filter coefficients (S=1)."""
    nyq = 0.5 * sr
    cutoff_hz = float(np.clip(cutoff_hz, 20.0, nyq * 0.95))
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * cutoff_hz / float(sr)
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / 2.0 * np.sqrt(2.0)
    two_sqrtA_alpha = 2.0 * np.sqrt(A) * alpha

    if mode == "low":
        b0 = A * ((A + 1.0) - (A - 1.0) * cos_w0 + two_sqrtA_alpha)
        b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cos_w0)
        b2 = A * ((A + 1.0) - (A - 1.0) * cos_w0 - two_sqrtA_alpha)
        a0 = (A + 1.0) + (A - 1.0) * cos_w0 + two_sqrtA_alpha
        a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cos_w0)
        a2 = (A + 1.0) + (A - 1.0) * cos_w0 - two_sqrtA_alpha
    else:
        b0 = A * ((A + 1.0) + (A - 1.0) * cos_w0 + two_sqrtA_alpha)
        b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w0)
        b2 = A * ((A + 1.0) + (A - 1.0) * cos_w0 - two_sqrtA_alpha)
        a0 = (A + 1.0) - (A - 1.0) * cos_w0 + two_sqrtA_alpha
        a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cos_w0)
        a2 = (A + 1.0) - (A - 1.0) * cos_w0 - two_sqrtA_alpha

    b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return b, a


def apply_dsp_operation(
    samples: np.ndarray,
    sr: int,
    operation_type: str,
    parameters: Dict[str, Any],
    engine: str = "scipy_dsp",
) -> Tuple[np.ndarray, int, List[str], List[str]]:
    """
    Execute a reproducible DSP transformation on float32 PCM audio (channels, samples).
    Returns (processed_samples, output_sr, audit_log, warnings).
    """
    if samples.ndim != 2 or samples.shape[1] == 0:
        raise ValueError("Input samples must have shape (channels, samples) with non-zero length.")

    orig_channels, orig_len = samples.shape
    out = samples.astype(np.float64).copy()
    out_sr = int(sr)
    audit_log: List[str] = [
        f"Input signal: {orig_channels} ch, {orig_len} samples @ {sr} Hz ({orig_len / sr:.3f}s)"
    ]
    warnings: List[str] = []

    nyq = 0.5 * sr

    # Optional Pedalboard execution path when requested and installed
    if engine == "pedalboard":
        try:
            import pedalboard as pb

            board_plugins = []
            if operation_type == "highpass":
                cutoff = float(np.clip(parameters.get("cutoff_hz", 100.0), 20.0, nyq * 0.95))
                board_plugins.append(pb.HighpassFilter(cutoff_frequency_hz=cutoff))
            elif operation_type == "lowpass":
                cutoff = float(np.clip(parameters.get("cutoff_hz", 7500.0), 100.0, nyq * 0.95))
                board_plugins.append(pb.LowpassFilter(cutoff_frequency_hz=cutoff))
            elif operation_type == "parametric_eq":
                c_hz = float(np.clip(parameters.get("center_hz", 3000.0), 40.0, nyq * 0.95))
                q_val = float(np.clip(parameters.get("q_factor", 1.5), 0.1, 15.0))
                g_db = float(np.clip(parameters.get("gain_db", -3.0), -24.0, 24.0))
                board_plugins.append(pb.PeakFilter(cutoff_frequency_hz=c_hz, gain_db=g_db, q=q_val))
            elif operation_type == "dynamic_compression":
                t_db = float(parameters.get("threshold_db", -16.0))
                rat = float(max(1.0, parameters.get("ratio", 3.0)))
                att = float(max(0.5, parameters.get("attack_ms", 10.0)))
                rel = float(max(5.0, parameters.get("release_ms", 100.0)))
                board_plugins.append(
                    pb.Compressor(threshold_db=t_db, ratio=rat, attack_ms=att, release_ms=rel)
                )
            elif operation_type == "gain_limiter":
                g_db = float(np.clip(parameters.get("gain_db", 0.0), -36.0, 24.0))
                board_plugins.append(pb.Gain(gain_db=g_db))
                if parameters.get("enable_limiter", True):
                    ceil_db = float(np.clip(parameters.get("ceiling_db", -0.5), -12.0, 0.0))
                    board_plugins.append(pb.Limiter(threshold_db=ceil_db))

            if board_plugins:
                board = pb.Pedalboard(board_plugins)
                out = board(samples.astype(np.float32), out_sr).astype(np.float64)
                audit_log.append(f"Executed Spotify Pedalboard chain ({len(board_plugins)} plugin(s)) for {operation_type}.")
                return out.astype(np.float32), out_sr, audit_log, warnings
            else:
                warnings.append(
                    f"Operation '{operation_type}' is a custom spectral routine not native to Pedalboard; executed via SciPy DSP engine."
                )
        except ImportError:
            warnings.append("Spotify Pedalboard not installed; falling back to SciPy DSP engine.")

    if operation_type == "highpass":
        cutoff_hz = float(np.clip(parameters.get("cutoff_hz", 100.0), 15.0, nyq * 0.95))
        order = int(np.clip(parameters.get("order", 4), 1, 10))
        sos = signal.butter(order, cutoff_hz / nyq, btype="highpass", output="sos")
        out = signal.sosfiltfilt(sos, out, axis=1)
        audit_log.append(f"Applied zero-phase Butterworth high-pass filter: cutoff={cutoff_hz:.1f} Hz, order={order}")

    elif operation_type == "lowpass":
        cutoff_hz = float(np.clip(parameters.get("cutoff_hz", 7500.0), 100.0, nyq * 0.98))
        order = int(np.clip(parameters.get("order", 6), 1, 12))
        sos = signal.butter(order, cutoff_hz / nyq, btype="lowpass", output="sos")
        out = signal.sosfiltfilt(sos, out, axis=1)
        audit_log.append(f"Applied zero-phase Butterworth low-pass filter: cutoff={cutoff_hz:.1f} Hz, order={order}")

    elif operation_type == "bandpass":
        low_hz = float(np.clip(parameters.get("low_hz", 250.0), 20.0, nyq * 0.85))
        high_hz = float(np.clip(parameters.get("high_hz", 6000.0), low_hz + 50.0, nyq * 0.98))
        order = int(np.clip(parameters.get("order", 4), 1, 8))
        sos = signal.butter(order, [low_hz / nyq, high_hz / nyq], btype="bandpass", output="sos")
        out = signal.sosfiltfilt(sos, out, axis=1)
        audit_log.append(f"Applied zero-phase Butterworth band-pass filter: {low_hz:.1f}-{high_hz:.1f} Hz, order={order}")

    elif operation_type == "parametric_eq":
        center_hz = float(np.clip(parameters.get("center_hz", 3500.0), 20.0, nyq * 0.95))
        q_factor = float(np.clip(parameters.get("q_factor", 2.0), 0.1, 20.0))
        gain_db = float(np.clip(parameters.get("gain_db", -4.0), -24.0, 24.0))
        b, a = _design_biquad_peaking(center_hz, out_sr, q_factor, gain_db)
        out = signal.lfilter(b, a, out, axis=1)
        audit_log.append(f"Applied parametric peaking EQ: center={center_hz:.1f} Hz, Q={q_factor:.2f}, gain={gain_db:+.2f} dB")

    elif operation_type == "eq_3band":
        low_gain = float(np.clip(parameters.get("low_gain_db", 0.0), -24.0, 24.0))
        low_freq = float(np.clip(parameters.get("low_freq_hz", 120.0), 20.0, 500.0))
        mid_gain = float(np.clip(parameters.get("mid_gain_db", 0.0), -24.0, 24.0))
        mid_freq = float(np.clip(parameters.get("mid_freq_hz", 1200.0), 200.0, 5000.0))
        mid_q = float(np.clip(parameters.get("mid_q", 0.707), 0.1, 10.0))
        high_gain = float(np.clip(parameters.get("high_gain_db", 0.0), -24.0, 24.0))
        high_freq = float(np.clip(parameters.get("high_freq_hz", 6000.0), 1500.0, nyq * 0.95))

        if abs(low_gain) > 1e-3:
            b, a = _design_biquad_shelf(low_freq, out_sr, low_gain, mode="low")
            out = signal.lfilter(b, a, out, axis=1)
        if abs(mid_gain) > 1e-3:
            b, a = _design_biquad_peaking(mid_freq, out_sr, mid_q, mid_gain)
            out = signal.lfilter(b, a, out, axis=1)
        if abs(high_gain) > 1e-3:
            b, a = _design_biquad_shelf(high_freq, out_sr, high_gain, mode="high")
            out = signal.lfilter(b, a, out, axis=1)
        audit_log.append(
            f"Applied 3-Band EQ: Low({low_gain:+.1f} dB @ {low_freq:.0f} Hz), "
            f"Mid({mid_gain:+.1f} dB @ {mid_freq:.0f} Hz, Q={mid_q:.2f}), "
            f"High({high_gain:+.1f} dB @ {high_freq:.0f} Hz)"
        )

    elif operation_type == "spectral_smoothing":
        freq_min = float(np.clip(parameters.get("freq_min_hz", 1000.0), 0.0, nyq))
        freq_max = float(np.clip(parameters.get("freq_max_hz", min(8000.0, nyq)), freq_min + 100.0, nyq))
        smooth_bins = int(np.clip(parameters.get("smoothing_bins", 9), 3, 45))
        mix = float(np.clip(parameters.get("mix", 0.65), 0.0, 1.0))
        n_fft = int(parameters.get("n_fft", 4096))
        hop_length = n_fft // 4

        freqs = np.linspace(0.0, nyq, (n_fft // 2) + 1)
        band_mask = (freqs >= freq_min) & (freqs <= freq_max)

        reconstructed_channels = []
        for ch in range(orig_channels):
            stft_c = librosa.stft(out[ch].astype(np.float32), n_fft=n_fft, hop_length=hop_length)
            mag = np.abs(stft_c)
            phase = np.exp(1j * np.angle(stft_c))

            log_mag = np.log10(np.clip(mag, EPS, None))
            smoothed_log_mag = uniform_filter1d(log_mag, size=smooth_bins, axis=0, mode="nearest")

            new_log_mag = log_mag.copy()
            new_log_mag[band_mask, :] = (1.0 - mix) * log_mag[band_mask, :] + mix * smoothed_log_mag[band_mask, :]
            new_mag = 10.0 ** new_log_mag

            rec = librosa.istft(new_mag * phase, hop_length=hop_length, length=orig_len)
            reconstructed_channels.append(rec)

        out = np.stack(reconstructed_channels, axis=0).astype(np.float64)
        audit_log.append(
            f"Applied STFT/ISTFT spectral smoothing in {freq_min:.0f}-{freq_max:.0f} Hz "
            f"(n_fft={n_fft}, smoothing_bins={smooth_bins}, mix={mix:.2f})"
        )

    elif operation_type == "frequency_selective":
        freq_min = float(np.clip(parameters.get("freq_min_hz", parameters.get("low_hz", 1000.0)), 0.0, nyq))
        freq_max = float(np.clip(parameters.get("freq_max_hz", parameters.get("high_hz", min(8000.0, nyq))), freq_min + 100.0, nyq))
        spike_thresh_db = float(np.clip(parameters.get("spike_threshold_db", 2.2), 0.5, 12.0))
        default_supp = 0.6
        if "gain_db" in parameters and "suppression_ratio" not in parameters:
            default_supp = float(np.clip(abs(float(parameters["gain_db"])) / 15.0, 0.1, 1.0))
        suppression = float(np.clip(parameters.get("suppression_ratio", default_supp), 0.0, 1.0))
        n_fft = int(parameters.get("n_fft", 4096))
        hop_length = n_fft // 4

        freqs = np.linspace(0.0, nyq, (n_fft // 2) + 1)
        band_mask = (freqs >= freq_min) & (freqs <= freq_max)

        reconstructed_channels = []
        for ch in range(orig_channels):
            stft_c = librosa.stft(out[ch].astype(np.float32), n_fft=n_fft, hop_length=hop_length)
            mag = np.abs(stft_c)
            phase = np.exp(1j * np.angle(stft_c))

            mag_db = 20.0 * np.log10(np.clip(mag, EPS, None))
            mean_profile_db = np.mean(mag_db, axis=1)
            local_baseline_db = uniform_filter1d(mean_profile_db, size=15, mode="nearest")
            excess_db = np.clip(mean_profile_db - local_baseline_db - spike_thresh_db, 0.0, None)

            attenuation_db = np.zeros_like(mean_profile_db)
            attenuation_db[band_mask] = excess_db[band_mask] * suppression
            gain_curve = (10.0 ** (-attenuation_db / 20.0))[:, np.newaxis]

            rec = librosa.istft((mag * gain_curve) * phase, hop_length=hop_length, length=orig_len)
            reconstructed_channels.append(rec)

        out = np.stack(reconstructed_channels, axis=0).astype(np.float64)
        audit_log.append(
            f"Applied frequency-selective comb spike damping in {freq_min:.0f}-{freq_max:.0f} Hz "
            f"(threshold={spike_thresh_db:.1f} dB, suppression={suppression:.2f})"
        )

    elif operation_type == "noise_reduction":
        reduction_db = float(np.clip(parameters.get("reduction_db", 10.0), 1.0, 30.0))
        sensitivity = float(np.clip(parameters.get("sensitivity", 1.2), 0.5, 3.0))
        n_fft = int(parameters.get("n_fft", 2048))
        hop_length = n_fft // 4
        min_gain = 10.0 ** (-reduction_db / 20.0)

        reconstructed_channels = []
        for ch in range(orig_channels):
            stft_c = librosa.stft(out[ch].astype(np.float32), n_fft=n_fft, hop_length=hop_length)
            mag = np.abs(stft_c)
            phase = np.exp(1j * np.angle(stft_c))

            # Estimate noise profile from 15th percentile across time frames per frequency bin
            noise_thresh = np.percentile(mag, 15.0, axis=1, keepdims=True) * sensitivity
            wiener_mask = np.clip(1.0 - (noise_thresh / np.clip(mag, EPS, None)) ** 2, min_gain, 1.0)
            rec = librosa.istft((mag * wiener_mask) * phase, hop_length=hop_length, length=orig_len)
            reconstructed_channels.append(rec)

        out = np.stack(reconstructed_channels, axis=0).astype(np.float64)
        audit_log.append(
            f"Applied broadband Wiener spectral gating noise reduction (reduction={reduction_db:.1f} dB, sensitivity={sensitivity:.2f})"
        )

    elif operation_type == "dynamic_compression":
        thresh_db = float(np.clip(parameters.get("threshold_db", -16.0), -48.0, 0.0))
        ratio = float(np.clip(parameters.get("ratio", 3.0), 1.0, 20.0))
        attack_ms = float(np.clip(parameters.get("attack_ms", 10.0), 1.0, 200.0))
        release_ms = float(np.clip(parameters.get("release_ms", 100.0), 10.0, 1000.0))
        makeup_db = float(np.clip(parameters.get("makeup_gain_db", 1.5), -12.0, 18.0))

        # Envelope follower over block steps for speed & accuracy
        env_abs = np.max(np.abs(out), axis=0)
        win_samples = max(1, int(0.005 * out_sr))
        smooth_env = uniform_filter1d(env_abs, size=win_samples, mode="nearest")
        env_db = 20.0 * np.log10(np.clip(smooth_env, 1e-7, None))

        over_db = np.maximum(0.0, env_db - thresh_db)
        gain_reduction_db = over_db * (1.0 - 1.0 / ratio)

        # Smooth gain curve according to attack/release average
        smooth_gr_samples = max(1, int(((attack_ms + release_ms) * 0.5 / 1000.0) * out_sr))
        smooth_gr_db = uniform_filter1d(gain_reduction_db, size=smooth_gr_samples, mode="nearest")
        total_gain_linear = 10.0 ** ((-smooth_gr_db + makeup_db) / 20.0)

        out = out * total_gain_linear[np.newaxis, :]
        audit_log.append(
            f"Applied dynamic range compression: threshold={thresh_db:.1f} dBFS, ratio={ratio:.1f}:1, makeup={makeup_db:+.1f} dB"
        )

    elif operation_type == "gain_limiter":
        gain_db = float(np.clip(parameters.get("gain_db", 0.0), -36.0, 24.0))
        enable_limiter = bool(parameters.get("enable_limiter", True))
        ceiling_db = float(np.clip(parameters.get("ceiling_db", -0.3), -18.0, 0.0))

        out = out * (10.0 ** (gain_db / 20.0))
        if enable_limiter:
            ceiling_lin = 10.0 ** (ceiling_db / 20.0)
            # Soft-knee tanh saturation near ceiling
            out = ceiling_lin * np.tanh(out / max(ceiling_lin, EPS))
        audit_log.append(
            f"Applied gain adjustment ({gain_db:+.2f} dB) with soft-knee limiter={enable_limiter} (ceiling={ceiling_db:.2f} dBFS)"
        )

    elif operation_type == "resample":
        intermediate_sr = int(np.clip(parameters.get("target_sr", 16000), 8000, 96000))
        preserve_original_sr = bool(parameters.get("roundtrip_to_original_sr", True))

        down_chs = [soxr.resample(out[ch], out_sr, intermediate_sr, quality="HQ") for ch in range(orig_channels)]
        down_arr = np.stack(down_chs, axis=0)
        if preserve_original_sr and intermediate_sr != out_sr:
            up_chs = [soxr.resample(down_arr[ch], intermediate_sr, out_sr, quality="HQ") for ch in range(orig_channels)]
            up_arr = np.stack(up_chs, axis=0)
            # Match exact original sample count
            if up_arr.shape[1] >= orig_len:
                out = up_arr[:, :orig_len]
            else:
                out = np.pad(up_arr, ((0, 0), (0, orig_len - up_arr.shape[1])), mode="constant")
            audit_log.append(f"Applied sample-rate roundtrip: {out_sr} Hz -> {intermediate_sr} Hz -> {out_sr} Hz")
        else:
            out = down_arr
            out_sr = intermediate_sr
            audit_log.append(f"Converted sample rate from {sr} Hz to {out_sr} Hz")

    elif operation_type == "codec_simulation":
        codec_name = str(parameters.get("codec", "mp3")).lower()
        bitrate_kbps = int(np.clip(parameters.get("bitrate_kbps", 128), 64, 320))
        encoded_bytes = encode_audio_bytes(out.astype(np.float32), out_sr, fmt=codec_name, bitrate_kbps=bitrate_kbps)
        temp_codec_file = settings.temp_dir / f"codec_sim_{int(time.time() * 1000)}.{codec_name}"
        temp_codec_file.write_bytes(encoded_bytes)
        try:
            decoded_pcm, dec_sr = decode_audio_pcm(temp_codec_file, target_sr=out_sr, mono=False)
        finally:
            try:
                temp_codec_file.unlink()
            except OSError:
                pass
        if decoded_pcm.shape[1] >= orig_len:
            out = decoded_pcm[:, :orig_len].astype(np.float64)
        else:
            out = np.pad(
                decoded_pcm.astype(np.float64),
                ((0, 0), (0, orig_len - decoded_pcm.shape[1])),
                mode="constant",
            )
        audit_log.append(f"Executed real codec transcode roundtrip: {codec_name.upper()} @ {bitrate_kbps} kbps")

    elif operation_type == "notch_comb":
        f0 = float(np.clip(parameters.get("fundamental_hz", 400.0), 50.0, 2000.0))
        min_f = float(np.clip(parameters.get("min_freq_hz", 1200.0), 100.0, nyq * 0.9))
        max_f = float(np.clip(parameters.get("max_freq_hz", min(7800.0, nyq * 0.95)), min_f + 100.0, nyq * 0.98))
        q_val = float(np.clip(parameters.get("q", 30.0), 5.0, 80.0))

        notched_freqs: List[float] = []
        f_curr = f0
        while f_curr <= max_f:
            if f_curr >= min_f and f_curr < nyq * 0.98:
                b_n, a_n = signal.iirnotch(f_curr, q_val, fs=out_sr)
                out = signal.filtfilt(b_n, a_n, out, axis=1)
                notched_freqs.append(round(f_curr, 1))
            f_curr += f0
        audit_log.append(
            f"Applied harmonic comb IIR notch filter (f0={f0:.1f} Hz, Q={q_val:.1f}) across {len(notched_freqs)} harmonics in {min_f:.0f}-{max_f:.0f} Hz"
        )

    elif operation_type == "pedalboard_chain":
        hp_hz = float(np.clip(parameters.get("highpass_hz", 80.0), 20.0, 500.0))
        pk_hz = float(np.clip(parameters.get("peak_freq_hz", 4500.0), 200.0, nyq * 0.95))
        pk_gain = float(np.clip(parameters.get("peak_gain_db", -2.5), -18.0, 18.0))
        t_db = float(np.clip(parameters.get("threshold_db", -16.0), -40.0, 0.0))
        rat = float(np.clip(parameters.get("ratio", 2.5), 1.0, 15.0))
        ceil_db = float(np.clip(parameters.get("ceiling_db", -0.5), -12.0, 0.0))
        try:
            import pedalboard as pb
            board = pb.Pedalboard([
                pb.HighpassFilter(cutoff_frequency_hz=hp_hz),
                pb.PeakFilter(cutoff_frequency_hz=pk_hz, gain_db=pk_gain, q=1.8),
                pb.Compressor(threshold_db=t_db, ratio=rat, attack_ms=10.0, release_ms=100.0),
                pb.Limiter(threshold_db=ceil_db),
            ])
            out = board(samples.astype(np.float32), out_sr).astype(np.float64)
            audit_log.append(
                f"Executed Spotify Pedalboard 4-stage mastering chain (HPF {hp_hz:.0f} Hz, Peak {pk_gain:+.1f} dB @ {pk_hz:.0f} Hz, Comp {rat:.1f}:1, Limiter {ceil_db:.1f} dBFS)"
            )
        except ImportError:
            sos = signal.butter(4, hp_hz / nyq, btype="highpass", output="sos")
            out = signal.sosfiltfilt(sos, out, axis=1)
            b_p, a_p = _design_biquad_peaking(pk_hz, out_sr, 1.8, pk_gain)
            out = signal.lfilter(b_p, a_p, out, axis=1)
            warnings.append("Pedalboard not available; executed equivalent SciPy HPF + Peaking EQ chain.")
            audit_log.append("Executed fallback SciPy mastering chain.")

    else:
        raise ValueError(f"Unsupported DSP operation_type: '{operation_type}'")

    # Integrity checks for NaN / Infinity
    if not np.all(np.isfinite(out)):
        warnings.append("NaN or Infinity values detected after filtering; sanitized non-finite samples to 0.0.")
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    # Peak check
    peak_after = float(np.max(np.abs(out))) if out.size > 0 else 0.0
    if peak_after >= 1.0:
        warnings.append(f"Output peak reached {peak_after:.3f} (>= 1.0 FS); digital clipping present in PCM export.")

    return out.astype(np.float32), out_sr, audit_log, warnings


def encode_audio_bytes(
    samples: np.ndarray,
    sr: int,
    fmt: str = "wav",
    bitrate_kbps: int = 192,
) -> bytes:
    """Encode float32 PCM (channels, samples) to WAV, FLAC, MP3, or M4A bytes."""
    fmt_clean = fmt.lstrip(".").lower()
    # Clamp to valid [-1.0, 1.0] for integer PCM formats
    clamped = np.clip(samples, -1.0, 1.0).astype(np.float32)

    if fmt_clean in {"wav", "flac"}:
        buf = io.BytesIO()
        sf.write(buf, clamped.T, sr, format=fmt_clean.upper(), subtype="PCM_16")
        return buf.getvalue()

    if fmt_clean in {"mp3", "m4a"}:
        buf = io.BytesIO()
        container_fmt = "mp3" if fmt_clean == "mp3" else "ipod"
        codec_name = "libmp3lame" if fmt_clean == "mp3" else "aac"
        layout = "stereo" if clamped.shape[0] >= 2 else "mono"

        with av.open(buf, mode="w", format=container_fmt) as container:
            stream = container.add_stream(codec_name, rate=sr, layout=layout)
            stream.bit_rate = bitrate_kbps * 1000

            frame_arr = clamped[:2] if layout == "stereo" else clamped[:1]
            frame = av.AudioFrame.from_ndarray(frame_arr, format="fltp", layout=layout)
            frame.sample_rate = sr
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)
        return buf.getvalue()

    raise ValueError(f"Unsupported export format: {fmt}")


def run_dsp_experiment(
    db: Session,
    asset: AudioAsset,
    job: AnalysisJob,
    operation_type: str,
    parameters: Dict[str, Any],
    preset_name: Optional[str] = None,
    engine: str = "scipy_dsp",
    export_format: str = "wav",
) -> Tuple[AudioAsset, ProcessingExperiment, np.ndarray, int, np.ndarray, int]:
    """
    Decode original asset, apply DSP transformation, validate integrity, archive processed file,
    and persist ProcessingExperiment record.
    """
    t0 = time.perf_counter()
    orig_path = resolve_asset_path(asset)
    orig_samples, orig_sr = decode_audio_pcm(orig_path, target_sr=None, mono=False)

    raw_finite_check = True
    proc_samples, proc_sr, audit_log, warnings = apply_dsp_operation(
        samples=orig_samples,
        sr=orig_sr,
        operation_type=operation_type,
        parameters=parameters,
        engine=engine,
    )

    nan_inf_detected = any("NaN or Infinity" in w for w in warnings)
    clipping_detected = bool(np.any(np.abs(proc_samples) >= 0.999))
    channels_preserved = proc_samples.shape[0] == orig_samples.shape[0]
    orig_dur = orig_samples.shape[1] / float(orig_sr)
    proc_dur = proc_samples.shape[1] / float(proc_sr)
    duration_preserved = abs(orig_dur - proc_dur) <= 0.05
    reconstruction_valid = bool(proc_samples.size > 0 and not nan_inf_detected and proc_sr > 0)

    if not duration_preserved:
        warnings.append(f"Duration changed from {orig_dur:.3f}s to {proc_dur:.3f}s.")
    if not channels_preserved:
        warnings.append(f"Channel count changed from {orig_samples.shape[0]} to {proc_samples.shape[0]}.")

    ext = f".{export_format.lstrip('.').lower()}"
    if ext not in {".wav", ".flac", ".mp3", ".m4a"}:
        ext = ".wav"

    out_bytes = encode_audio_bytes(proc_samples, proc_sr, fmt=ext[1:])
    stem = Path(asset.sanitized_filename).stem
    out_filename = f"{stem}_{operation_type}{ext}"

    out_asset = ingest_uploaded_bytes(
        db=db,
        raw_bytes=out_bytes,
        original_filename=out_filename,
        is_processed_variant=True,
        parent_asset_id=asset.id,
        variant_description=f"DSP: {preset_name or operation_type}",
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    exp = ProcessingExperiment(
        job_id=job.id,
        source_asset_id=asset.id,
        output_asset_id=out_asset.id,
        operation_type=operation_type,
        preset_name=preset_name,
        parameters_json=json.dumps(parameters),
        engine_used=engine,
        original_sha256=asset.sha256_checksum,
        processed_sha256=out_asset.sha256_checksum,
        duration_preserved=duration_preserved,
        channels_preserved=channels_preserved,
        clipping_detected=clipping_detected,
        nan_inf_detected=nan_inf_detected,
        reconstruction_valid=reconstruction_valid,
        processing_time_ms=round(elapsed_ms, 2),
        audit_log_json=json.dumps(audit_log),
        warnings_json=json.dumps(warnings),
        status="completed",
    )
    db.add(exp)
    db.commit()
    db.refresh(exp)

    return out_asset, exp, orig_samples, orig_sr, proc_samples, proc_sr
