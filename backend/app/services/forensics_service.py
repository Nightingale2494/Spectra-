"""Mathematical audio forensics engine: time-domain, frequency-domain, and neural-vocoder fakeprint analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
from scipy import signal
from scipy.ndimage import minimum_filter1d
import soxr
import torch
import torchaudio

from app.services.ingestion_service import decode_audio_pcm

EPS = 1e-10


def _to_db(val: float, floor_db: float = -120.0) -> float:
    if val <= EPS:
        return floor_db
    return float(max(floor_db, 20.0 * np.log10(val)))


def compute_waveform_envelope(
    samples: np.ndarray,
    sr: int,
    num_points: int = 600,
) -> Dict[str, Any]:
    """Compute min/max peak and RMS envelope across time for interactive waveform visualization."""
    mono = np.mean(samples, axis=0) if samples.shape[0] > 1 else samples[0]
    total_samples = len(mono)
    duration = float(total_samples) / float(sr) if sr > 0 else 0.0

    actual_points = min(num_points, max(1, total_samples))
    edges = np.linspace(0, total_samples, actual_points + 1, dtype=int)

    times: List[float] = []
    peaks_min: List[float] = []
    peaks_max: List[float] = []
    rms_env: List[float] = []

    for i in range(actual_points):
        s_idx, e_idx = edges[i], max(edges[i] + 1, edges[i + 1])
        chunk = mono[s_idx:e_idx]
        t_mid = ((s_idx + e_idx) * 0.5) / float(sr)
        times.append(round(float(t_mid), 4))
        peaks_min.append(round(float(np.min(chunk)), 5))
        peaks_max.append(round(float(np.max(chunk)), 5))
        rms_env.append(round(float(np.sqrt(np.mean(chunk ** 2))), 5))

    return {
        "sample_rate": sr,
        "duration_seconds": round(duration, 4),
        "channels": int(samples.shape[0]),
        "num_points": actual_points,
        "times": times,
        "peaks_min": peaks_min,
        "peaks_max": peaks_max,
        "rms_envelope": rms_env,
    }


def compute_time_domain_metrics(samples: np.ndarray, sr: int) -> Dict[str, Any]:
    """Compute comprehensive time-domain audio forensics measurements."""
    channels, n_samples = samples.shape
    duration = float(n_samples) / float(sr) if sr > 0 else 0.0
    mono = np.mean(samples, axis=0) if channels > 1 else samples[0]

    peak_linear = float(np.max(np.abs(samples))) if samples.size > 0 else 0.0
    rms_linear = float(np.sqrt(np.mean(samples ** 2))) if samples.size > 0 else 0.0
    peak_dbfs = _to_db(peak_linear)
    rms_dbfs = _to_db(rms_linear)

    crest_linear = float(peak_linear / max(rms_linear, EPS))
    crest_db = float(20.0 * np.log10(max(crest_linear, EPS)))
    dc_offset = float(np.mean(mono))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=mono.astype(np.float32)))) if len(mono) > 1 else 0.0

    # Clipping detection (|x| >= 0.999)
    clipped_mask = np.abs(samples) >= 0.999
    clipping_samples = int(np.sum(clipped_mask))
    clipping_ratio = float(clipping_samples / max(1, samples.size))
    clipping_detected = clipping_samples > 0

    # Silence region detection (frames below -50 dBFS lasting >= 50 ms)
    frame_len = max(1, int(0.025 * sr))
    hop_len = max(1, int(0.010 * sr))
    silence_threshold_linear = 10.0 ** (-50.0 / 20.0)
    silence_regions: List[Dict[str, float]] = []
    silent_frames = 0
    total_frames = 0

    if len(mono) >= frame_len:
        n_frames = 1 + (len(mono) - frame_len) // hop_len
        total_frames = n_frames
        in_silence = False
        sil_start = 0.0

        for idx in range(n_frames):
            f_start = idx * hop_len
            frame = mono[f_start : f_start + frame_len]
            f_rms = float(np.sqrt(np.mean(frame ** 2)))
            t_sec = f_start / float(sr)
            if f_rms < silence_threshold_linear:
                silent_frames += 1
                if not in_silence:
                    in_silence = True
                    sil_start = t_sec
            else:
                if in_silence:
                    in_silence = False
                    sil_dur = t_sec - sil_start
                    if sil_dur >= 0.05:
                        silence_regions.append({
                            "start_sec": round(sil_start, 3),
                            "end_sec": round(t_sec, 3),
                            "duration_sec": round(sil_dur, 3),
                        })
        if in_silence:
            sil_dur = duration - sil_start
            if sil_dur >= 0.05:
                silence_regions.append({
                    "start_sec": round(sil_start, 3),
                    "end_sec": round(duration, 3),
                    "duration_sec": round(sil_dur, 3),
                })

    silence_ratio = float(silent_frames / max(1, total_frames))

    # Stereo correlation & balance if 2+ channels
    stereo_correlation: Optional[float] = None
    channel_balance_db: Optional[float] = None
    if channels >= 2:
        left, right = samples[0], samples[1]
        l_std = float(np.std(left))
        r_std = float(np.std(right))
        if l_std > EPS and r_std > EPS:
            stereo_correlation = round(float(np.corrcoef(left, right)[0, 1]), 4)
        l_rms = float(np.sqrt(np.mean(left ** 2)))
        r_rms = float(np.sqrt(np.mean(right ** 2)))
        channel_balance_db = round(_to_db(l_rms) - _to_db(r_rms), 3)

    return {
        "duration_seconds": round(duration, 4),
        "channels": int(channels),
        "total_samples_per_channel": int(n_samples),
        "peak_linear": round(peak_linear, 6),
        "peak_dbfs": round(peak_dbfs, 2),
        "peak_db": round(peak_dbfs, 2),
        "rms_linear": round(rms_linear, 6),
        "rms_dbfs": round(rms_dbfs, 2),
        "rms_db": round(rms_dbfs, 2),
        "crest_factor_linear": round(crest_linear, 3),
        "crest_factor_db": round(crest_db, 2),
        "zero_crossing_rate": round(zcr, 5),
        "dc_offset": round(dc_offset, 6),
        "clipping_detected": clipping_detected,
        "clipping_samples": clipping_samples,
        "clipping_ratio": round(clipping_ratio, 6),
        "silence_ratio": round(silence_ratio, 4),
        "silence_regions_count": len(silence_regions),
        "silence_regions": silence_regions[:25],
        "stereo_correlation": stereo_correlation,
        "channel_balance_db": channel_balance_db,
    }


def compute_frequency_domain_metrics(samples: np.ndarray, sr: int) -> Dict[str, Any]:
    """Compute STFT, spectral descriptors, spectral residuals, and cepstral features."""
    mono = np.mean(samples, axis=0) if samples.shape[0] > 1 else samples[0]
    mono = mono.astype(np.float32)

    n_fft = min(2048, max(256, 2 ** int(np.floor(np.log2(max(256, len(mono)))))))
    hop_length = max(64, n_fft // 4)

    stft_complex = librosa.stft(mono, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(stft_complex)
    power = mag ** 2
    log_power_db = 10.0 * np.log10(np.clip(power, EPS, None))

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mean_mag = np.mean(mag, axis=1)
    mean_log_power_db = np.mean(log_power_db, axis=1)

    # Spectral descriptors via librosa
    centroid = librosa.feature.spectral_centroid(S=mag, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=mag, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    flatness = librosa.feature.spectral_flatness(S=mag)[0]
    rolloff_85 = librosa.feature.spectral_rolloff(S=mag, sr=sr, roll_percent=0.85)[0]
    rolloff_95 = librosa.feature.spectral_rolloff(S=mag, sr=sr, roll_percent=0.95)[0]
    rolloff_99 = librosa.feature.spectral_rolloff(S=mag, sr=sr, roll_percent=0.99)[0]

    # Cepstral coefficients (MFCCs)
    mfcc = librosa.feature.mfcc(y=mono, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
    mfcc_means = [round(float(x), 3) for x in np.mean(mfcc, axis=1)]
    mfcc_stds = [round(float(x), 3) for x in np.std(mfcc, axis=1)]

    # Real cepstrum from mean log spectrum
    real_cepstrum = np.fft.irfft(mean_log_power_db)
    cepstral_Low = [round(float(x), 4) for x in real_cepstrum[:16]]

    # High-frequency energy ratio (>8 kHz and >12 kHz)
    total_energy = float(np.sum(mean_mag ** 2)) + EPS
    hf_8k_mask = freqs >= 8000.0
    hf_12k_mask = freqs >= 12000.0
    hf_8k_ratio = float(np.sum(mean_mag[hf_8k_mask] ** 2) / total_energy) if np.any(hf_8k_mask) else 0.0
    hf_12k_ratio = float(np.sum(mean_mag[hf_12k_mask] ** 2) / total_energy) if np.any(hf_12k_mask) else 0.0

    # Spectral residual across full spectrum
    smooth_env = minimum_filter1d(mean_log_power_db, size=15, mode="nearest")
    full_residual = np.clip(mean_log_power_db - smooth_env, 0.0, None)

    # Dominant frequency peak
    peak_bin = int(np.argmax(mean_mag))
    dominant_freq_hz = float(freqs[peak_bin])

    # Downsample mean spectrum to 256 points for JSON summary
    step = max(1, len(freqs) // 256)
    summary_freqs = [round(float(f), 1) for f in freqs[::step]]
    summary_db = [round(float(d), 2) for d in mean_log_power_db[::step]]
    summary_env_db = [round(float(e), 2) for e in smooth_env[::step]]
    summary_res_db = [round(float(r), 3) for r in full_residual[::step]]

    return {
        "n_fft": int(n_fft),
        "hop_length": int(hop_length),
        "nyquist_hz": float(sr / 2.0),
        "dominant_frequency_hz": round(dominant_freq_hz, 2),
        "spectral_centroid_hz": round(float(np.mean(centroid)), 2),
        "spectral_centroid_hz_mean": round(float(np.mean(centroid)), 2),
        "spectral_centroid_hz_std": round(float(np.std(centroid)), 2),
        "spectral_bandwidth_hz_mean": round(float(np.mean(bandwidth)), 2),
        "spectral_bandwidth_hz_std": round(float(np.std(bandwidth)), 2),
        "spectral_flatness_mean": round(float(np.mean(flatness)), 6),
        "spectral_flatness_db": round(float(10.0 * np.log10(max(float(np.mean(flatness)), EPS))), 2),
        "spectral_rolloff_85_hz": round(float(np.mean(rolloff_85)), 2),
        "spectral_rolloff_95_hz": round(float(np.mean(rolloff_95)), 2),
        "effective_bandwidth_cutoff_hz": round(float(np.mean(rolloff_99)), 2),
        "hf_energy_ratio_above_8khz": round(hf_8k_ratio, 6),
        "hf_energy_ratio_above_12khz": round(hf_12k_ratio, 6),
        "residual_mean_db": round(float(np.mean(full_residual)), 3),
        "residual_max_db": round(float(np.max(full_residual)), 3),
        "mfcc_mean": mfcc_means,
        "mfcc_std": mfcc_stds,
        "cepstral_coefficients_low": cepstral_Low,
        "spectrum_curve": {
            "frequencies_hz": summary_freqs,
            "mean_power_db": summary_db,
            "lower_envelope_db": summary_env_db,
            "residual_db": summary_res_db,
        },
    }


def extract_neural_vocoder_fakeprint(
    samples: np.ndarray,
    sr: int,
    target_sr: int = 16000,
    n_fft: int = 8192,
    freq_min: int = 1000,
    freq_max: int = 8000,
    hull_area: int = 10,
    max_db: float = 5.0,
    min_db: float = -45.0,
    max_duration: int = 300,
) -> Dict[str, Any]:
    """
    Extract the exact 16 kHz neural-vocoder spectral residual ("fakeprint")
    matching lofcz/ai-music-detector (based on Afchar et al., ISMIR 2025).
    """
    # Convert to mono Tensor of shape (1, T)
    if samples.shape[0] > 1:
        mono_np = np.mean(samples, axis=0, keepdims=True).astype(np.float32)
    else:
        mono_np = samples[:1, :].astype(np.float32)

    audio_t = torch.from_numpy(mono_np)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        audio_t = resampler(audio_t)

    max_samples = int(max_duration * target_sr)
    if audio_t.shape[1] > max_samples:
        audio_t = audio_t[:, :max_samples]

    # Pad if audio is shorter than n_fft
    if audio_t.shape[1] < n_fft:
        pad_len = n_fft - audio_t.shape[1]
        audio_t = torch.nn.functional.pad(audio_t, (0, pad_len))

    stft_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        power=2,
        normalized=False,
    )
    with torch.no_grad():
        spec = stft_transform(audio_t)

    spec_db = 10.0 * torch.log10(torch.clamp(spec, min=1e-10, max=1e6))
    mean_spectrum = spec_db.mean(dim=(0, 2)).cpu().numpy()

    freq_bins = np.linspace(0.0, target_sr / 2.0, num=(n_fft // 2) + 1)
    freq_mask = (freq_bins >= freq_min) & (freq_bins <= freq_max)
    freq_range = freq_bins[freq_mask]
    freq_spectrum = mean_spectrum[freq_mask]

    # Lower spectral envelope using minimum_filter1d (exact match to upstream inference.py)
    hull = minimum_filter1d(freq_spectrum, size=hull_area, mode="nearest")
    hull = np.clip(hull, min_db, None)

    residue_raw = np.clip(freq_spectrum - hull, 0.0, None)
    residue_clipped = np.clip(residue_raw, 0.0, max_db)
    max_val = float(np.max(residue_clipped)) + 1e-6
    fakeprint_vec = (residue_clipped / max_val).astype(np.float32)

    # Compute broadband spectral support & OOD diagnostics for 1-8 kHz
    signal_rms_lin = float(np.sqrt(np.mean(mono_np ** 2))) if mono_np.size > 0 else 0.0
    signal_rms_dbfs = _to_db(signal_rms_lin)
    mean_lin_power = spec.mean(dim=(0, 2)).cpu().numpy()
    total_spec_power = float(np.sum(mean_lin_power)) + EPS
    band_1k_8k_power = float(np.sum(mean_lin_power[freq_mask]))
    band_1k_8k_energy_ratio = float(band_1k_8k_power / total_spec_power)
    active_bin_ratio = float(np.mean(fakeprint_vec > 0.05)) if len(fakeprint_vec) > 0 else 0.0

    # Identify top spectral residual peaks in 1-8 kHz for forensic inspection
    peak_indices, _ = signal.find_peaks(fakeprint_vec, distance=12, prominence=0.15)
    if len(peak_indices) > 0:
        sorted_peaks = sorted(peak_indices, key=lambda idx: float(fakeprint_vec[idx]), reverse=True)[:10]
        anomalous_peaks = [
            {
                "frequency_hz": round(float(freq_range[idx]), 1),
                "normalized_residual": round(float(fakeprint_vec[idx]), 4),
                "raw_residual_db": round(float(residue_raw[idx]), 3),
            }
            for idx in sorted_peaks
        ]
    else:
        anomalous_peaks = []

    # Autocorrelation of the residual vector to quantify periodic pitch/deconvolution comb patterns
    centered_fp = fakeprint_vec - np.mean(fakeprint_vec)
    norm_denom = float(np.sum(centered_fp ** 2)) + EPS
    autocorr = np.correlate(centered_fp, centered_fp, mode="full")[len(centered_fp) - 1 :] / norm_denom
    # Search lag bins 8..120 (~15 Hz to 230 Hz spacing in 8192-FFT @ 16kHz where 1 bin = 1.953 Hz)
    search_lags = autocorr[8:120] if len(autocorr) > 120 else autocorr[2:]
    periodicity_score = float(np.max(search_lags)) if len(search_lags) > 0 else 0.0

    # Measure 200 Hz / 400 Hz transposed-convolution harmonic grid contrast (1600..8000 Hz)
    grid_freqs = np.arange(1600.0, 8001.0, 200.0)
    grid_bins = [int(np.argmin(np.abs(freq_range - gf))) for gf in grid_freqs]
    grid_vals = [float(np.max(fakeprint_vec[max(0, b - 1) : min(len(fakeprint_vec), b + 2)])) for b in grid_bins]
    off_grid_freqs = np.arange(1700.0, 7901.0, 200.0)
    off_bins = [int(np.argmin(np.abs(freq_range - gf))) for gf in off_grid_freqs]
    off_vals = [float(np.max(fakeprint_vec[max(0, b - 1) : min(len(fakeprint_vec), b + 2)])) for b in off_bins]
    comb_grid_200hz_mean = float(np.mean(grid_vals)) if grid_vals else 0.0
    comb_grid_200hz_contrast = float(comb_grid_200hz_mean - (float(np.mean(off_vals)) if off_vals else 0.0))

    # Measure temporal stationarity of 1-8 kHz spectral residual across 4 time segments
    # Stationary neural-vocoder upsampling combs remain locked in frequency over time, whereas
    # non-stationary musical harmonics or electric guitar distortion shift with chord progressions.
    spec_db_np = spec_db.squeeze(0).cpu().numpy()[freq_mask, :]
    n_stft_frames = spec_db_np.shape[1]
    if n_stft_frames >= 8:
        seg_edges = np.linspace(0, n_stft_frames, 5, dtype=int)
        seg_fps = []
        for idx_seg in range(4):
            seg_mean = np.mean(spec_db_np[:, seg_edges[idx_seg] : seg_edges[idx_seg + 1]], axis=1)
            seg_hull = np.clip(minimum_filter1d(seg_mean, size=hull_area, mode="nearest"), min_db, None)
            seg_r = np.clip(seg_mean - seg_hull, 0.0, max_db)
            seg_fp = seg_r / (float(np.max(seg_r)) + 1e-6)
            seg_fps.append(seg_fp)
        seg_corrs = []
        for i_s in range(4):
            for j_s in range(i_s + 1, 4):
                if np.std(seg_fps[i_s]) > 1e-6 and np.std(seg_fps[j_s]) > 1e-6:
                    seg_corrs.append(float(np.corrcoef(seg_fps[i_s], seg_fps[j_s])[0, 1]))
        temporal_stationarity = float(np.mean(seg_corrs)) if seg_corrs else 0.0
    else:
        temporal_stationarity = 0.0

    # Downsample for chart payload (~360 points across 1000-8000 Hz) while preserving full vector for ONNX
    chart_step = max(1, len(freq_range) // 360)

    return {
        "feature_vector": fakeprint_vec,
        "feature_dim": int(len(fakeprint_vec)),
        "sample_rate": target_sr,
        "n_fft": n_fft,
        "freq_min_hz": freq_min,
        "freq_max_hz": freq_max,
        "hull_area": hull_area,
        "residual_mean_norm": round(float(np.mean(fakeprint_vec)), 4),
        "residual_std_norm": round(float(np.std(fakeprint_vec)), 4),
        "residual_max_raw_db": round(float(np.max(residue_raw)), 3),
        "periodicity_score": round(periodicity_score, 4),
        "comb_periodicity_score": round(periodicity_score, 4),
        "comb_grid_200hz_mean": round(comb_grid_200hz_mean, 4),
        "comb_grid_200hz_contrast": round(comb_grid_200hz_contrast, 4),
        "temporal_stationarity": round(temporal_stationarity, 4),
        "signal_rms_dbfs": round(signal_rms_dbfs, 2),
        "band_1k_8k_energy_ratio": round(band_1k_8k_energy_ratio, 6),
        "active_bin_ratio": round(active_bin_ratio, 4),
        "top_residual_peaks": anomalous_peaks,
        "chart_data": {
            "frequencies_hz": [round(float(f), 1) for f in freq_range[::chart_step]],
            "spectrum_db": [round(float(s), 2) for s in freq_spectrum[::chart_step]],
            "lower_hull_db": [round(float(h), 2) for h in hull[::chart_step]],
            "residual_norm": [round(float(r), 4) for r in fakeprint_vec[::chart_step]],
        },
    }


def run_complete_forensics(
    file_path: Path,
    start_sec: float = 0.0,
    max_duration: Optional[float] = None,
) -> Dict[str, Any]:
    """Execute full time-domain, frequency-domain, and fakeprint forensic analysis."""
    samples, sr = decode_audio_pcm(
        file_path=file_path,
        target_sr=None,
        mono=False,
        max_duration=max_duration,
        start_sec=start_sec,
    )

    time_domain = compute_time_domain_metrics(samples, sr)
    freq_domain = compute_frequency_domain_metrics(samples, sr)
    fp_info = extract_neural_vocoder_fakeprint(samples, sr)

    # Generate technical observations grounded strictly in measured signal metrics (Tier 3: Audio Signal Observations)
    observations: List[str] = []
    if time_domain["clipping_detected"]:
        observations.append(
            f"Signal Observation: Digital clipping detected ({time_domain['clipping_samples']} samples, "
            f"{time_domain['clipping_ratio']*100:.3f}% reached >= 0.999 FS). Non-linear clipping introduces "
            "broadband harmonic overtones that can confound spectral hull detectors."
        )
    if freq_domain["hf_energy_ratio_above_12khz"] < 1e-5 and sr >= 32000:
        observations.append(
            "Signal Observation: Low energy above 12 kHz relative to Nyquist; consistent with either "
            "bandwidth-limited synthesis (16–24 kHz sample rate) or lossy perceptual audio coding (MP3/AAC lowpass)."
        )
    if fp_info["periodicity_score"] >= 0.25:
        observations.append(
            f"Signal Observation: Elevated periodic structure in 1–8 kHz spectral hull residual "
            f"(autocorrelation={fp_info['periodicity_score']:.2f}, 200 Hz grid contrast={fp_info['comb_grid_200hz_contrast']:+.3f}, "
            f"temporal stationarity={fp_info['temporal_stationarity']:.2f}). While consistent with periodic upsampling "
            "artifacts studied in open-weights neural vocoders (Afchar et al., ISMIR 2025), a spectral comb pattern "
            "alone does not identify undocumented proprietary generator architectures."
        )
    else:
        observations.append(
            f"Signal Observation: Moderate/low periodic comb structure in 1–8 kHz spectral residual "
            f"(autocorrelation={fp_info['periodicity_score']:.2f}, 200 Hz grid contrast={fp_info['comb_grid_200hz_contrast']:+.3f})."
        )

    observations.append(
        "Attribution Limitation: Spectral peaks and residuals are physical signal measurements that can also be "
        "excited by harmonic musical instruments, non-linear guitar/DAW distortion, or lossy codec framing."
    )

    # Remove raw ndarray from JSON dict
    fp_serializable = {k: v for k, v in fp_info.items() if k != "feature_vector"}

    return {
        "time_domain": time_domain,
        "frequency_domain": freq_domain,
        "fakeprint": fp_serializable,
        "observations": observations,
        "_fakeprint_vector": fp_info["feature_vector"],
        "_samples": samples,
        "_sr": sr,
    }


def compute_spectrogram_payload(
    file_path: Path,
    max_time_bins: int = 180,
    max_freq_bins: int = 128,
) -> Dict[str, Any]:
    """Generate 2D spectrogram heatmap and spectral residual curves for frontend visualization."""
    samples, sr = decode_audio_pcm(file_path, target_sr=None, mono=True, max_duration=120.0)
    mono = samples[0]

    n_fft = 2048
    hop_length = max(256, len(mono) // max(1, max_time_bins))
    stft_mag = np.abs(librosa.stft(mono, n_fft=n_fft, hop_length=hop_length))
    db_matrix = librosa.amplitude_to_db(stft_mag, ref=np.max)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(db_matrix.shape[1]), sr=sr, hop_length=hop_length)

    f_step = max(1, len(freqs) // max_freq_bins)
    t_step = max(1, len(times) // max_time_bins)

    sub_freqs = [round(float(f), 1) for f in freqs[::f_step]]
    sub_times = [round(float(t), 3) for t in times[::t_step]]
    sub_db = [
        [round(float(val), 1) for val in row[::t_step]]
        for row in db_matrix[::f_step]
    ]

    mean_db = np.mean(db_matrix, axis=1)
    fp_info = extract_neural_vocoder_fakeprint(samples, sr)
    fp_chart = fp_info["chart_data"]

    return {
        "sample_rate": sr,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "times": sub_times,
        "frequencies": sub_freqs,
        "magnitude_db": sub_db,
        "mean_spectrum_freqs": sub_freqs,
        "mean_spectrum_db": [round(float(x), 2) for x in mean_db[::f_step]],
        "fakeprint_freqs": fp_chart["frequencies_hz"],
        "fakeprint_spectrum_db": fp_chart["spectrum_db"],
        "fakeprint_hull_db": fp_chart["lower_hull_db"],
        "fakeprint_residual_norm": fp_chart["residual_norm"],
    }
