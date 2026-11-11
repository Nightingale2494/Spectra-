"""SONICS SpecTTTra deep-learning AI-music classifier adapter."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soxr
import torch

from app.core.config import settings
from app.detectors.base import BaseAudioDetector, DetectorPrediction


class SonicsDetector(BaseAudioDetector):
    def __init__(
        self,
        detector_id: str = "sonics_gamma_5s",
        name: str = "SONICS SpecTTTra-γ (5s Spectro-Temporal Transformer)",
        hf_repo: str = "awsaf49/sonics-spectttra-gamma-5s",
        hf_revision: str = "a349d427b681fa15e3d0e22d1708108322f967b1",
        local_subdir: str = "sonics-spectttra-gamma-5s",
        window_seconds: float = 5.0,
    ):
        self.detector_id = detector_id
        self.name = name
        self.version = "0.2.0-iclr25"
        self.model_source = f"https://huggingface.co/{hf_repo}"
        self.model_revision = hf_revision
        self.license = "MIT (code & pretrained weights); SONICS dataset is CC BY-NC 4.0 (not bundled)"
        self.input_sample_rate = 16000
        self.window_seconds = window_seconds
        self.expected_samples = int(self.input_sample_rate * self.window_seconds)
        self.input_duration_requirements = (
            f"{window_seconds:.0f}s window @ 16,000 Hz mono ({self.expected_samples:,} samples; multi-window evaluation)"
        )
        self.supported_devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        self.active_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.description = (
            f"End-to-end Spectro-Temporal Tokens Transformer (SpecTTTra, ICLR 2025) operating on "
            f"{window_seconds:.0f}-second 128-bin mel-spectrogram windows at 16 kHz with upstream "
            f"standard-deviation normalization and multi-window aggregation."
        )

        self.model_dir = settings.models_path / "sonics" / local_subdir
        self.model: Optional[torch.nn.Module] = None
        self.threshold: float = 0.5
        self.status_message: str = "Not loaded"
        self.weights_sha256: Optional[str] = None
        self.missing_keys: List[str] = []
        self.unexpected_keys: List[str] = []

    def is_available(self) -> bool:
        if not settings.SPECTRA_ENABLE_SONICS:
            return False
        if self.detector_id == "sonics_alpha_120s" and not settings.SPECTRA_ENABLE_SONICS_ALPHA_120S:
            return False
        cfg_file = self.model_dir / "config.json"
        bin_file = self.model_dir / "pytorch_model.bin"
        return cfg_file.exists() and bin_file.exists() and bin_file.stat().st_size > 1024

    def load(self, device: Optional[str] = None) -> bool:
        if not self.is_available():
            self.model = None
            self.status_message = (
                f"Local checkpoint not found or disabled in {self.model_dir.name}. "
                f"Download from {self.model_source} (rev {self.model_revision[:10]}) to enable."
            )
            return False

        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            from sonics.models.hf_model import HFAudioClassifier

            cfg_file = self.model_dir / "config.json"
            bin_file = self.model_dir / "pytorch_model.bin"

            with open(cfg_file, "r", encoding="utf-8") as f:
                cfg_dict = json.load(f)

            # Validate required config keys
            if "model" not in cfg_dict or "audio" not in cfg_dict:
                raise ValueError("Invalid SONICS config.json: missing 'model' or 'audio' section.")

            # Compute SHA-256 of checkpoint weights
            h = hashlib.sha256()
            with open(bin_file, "rb") as bf:
                for chunk in iter(lambda: bf.read(1024 * 1024), b""):
                    h.update(chunk)
            self.weights_sha256 = h.hexdigest()

            model = HFAudioClassifier(cfg_dict)
            state_dict = torch.load(bin_file, map_location=torch.device(target_device), weights_only=True)
            incompatible = model.load_state_dict(state_dict, strict=False)
            self.missing_keys = list(incompatible.missing_keys)
            self.unexpected_keys = list(incompatible.unexpected_keys)

            if self.missing_keys:
                raise RuntimeError(f"Checkpoint validation failed: missing keys {self.missing_keys[:5]}")

            model = model.to(target_device).eval()
            for p in model.parameters():
                p.requires_grad = False

            self.model = model
            self.active_device = target_device
            self.status_message = (
                f"Ready on {target_device} (missing_keys=0, unexpected_keys={len(self.unexpected_keys)})"
            )
            return True
        except Exception as exc:
            self.model = None
            self.status_message = f"SONICS load failed: {exc}"
            return False

    def _prepare_window(
        self, audio_16k: np.ndarray, start_idx: int, pad_mode: str = "constant"
    ) -> Tuple[np.ndarray, float]:
        """
        Extract a fixed-length window and normalize using exact upstream SONICS preprocessing
        (`sonics/utils/dataset.py`):
          1. Crop window `[start_idx : start_idx + max_len]`
          2. If shorter than `max_len`, zero-pad at the end using `mode="constant"`
          3. Normalize by standard deviation (`audio /= np.maximum(np.std(audio), 1e-6)`)
        """
        window = audio_16k[start_idx : start_idx + self.expected_samples]
        rms_lin = float(np.sqrt(np.mean(window ** 2))) if len(window) > 0 else 0.0
        rms_dbfs = 20.0 * np.log10(max(rms_lin, 1e-10))

        if len(window) < self.expected_samples:
            pad_len = self.expected_samples - len(window)
            window = np.pad(window, (0, pad_len), mode=pad_mode)

        std = float(np.std(window))
        if std > 1e-6:
            window = window / std
        return window.astype(np.float32), rms_dbfs

    def _score_single_window(self, win_np: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate a single window `(1, T)` through the model so `torchaudio.functional.amplitude_to_DB`
        computes `top_db = 80.0` independently for this window without cross-window batch contamination.
        """
        assert self.model is not None
        win_t = torch.from_numpy(win_np).unsqueeze(0).to(self.active_device)
        with torch.no_grad():
            logit_t = self.model(win_t).squeeze(-1)
            prob_t = torch.sigmoid(logit_t)
        return float(prob_t.item()), float(logit_t.item())

    def predict(
        self,
        samples: np.ndarray,
        sr: int,
        precomputed_features: Optional[np.ndarray] = None,
    ) -> DetectorPrediction:
        t0 = time.perf_counter()
        duration_sec = float(samples.shape[1]) / float(sr) if (samples.ndim == 2 and sr > 0) else 0.0

        if self.model is None:
            if not self.load():
                return DetectorPrediction(
                    detector_id=self.detector_id,
                    detector_name=self.name,
                    detector_version=self.version,
                    model_source=self.model_source,
                    model_revision=self.model_revision,
                    probability=None,
                    raw_score=None,
                    threshold=self.threshold,
                    predicted_label="Unavailable",
                    confidence=None,
                    input_sample_rate=self.input_sample_rate,
                    input_duration_sec=round(duration_sec, 3),
                    processing_time_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                    device_used=self.active_device,
                    feature_dim=None,
                    status="unavailable",
                    error_message=self.status_message,
                )

        try:
            if samples is None or samples.ndim != 2 or samples.shape[1] == 0 or sr <= 0:
                raise ValueError("Invalid audio PCM input for SONICS detector.")
            if not np.all(np.isfinite(samples)):
                raise ValueError("Audio input contains NaN or Infinity values.")

            mono = np.mean(samples, axis=0) if samples.shape[0] > 1 else samples[0]
            mono = mono.astype(np.float32)

            if sr != self.input_sample_rate:
                audio_16k = soxr.resample(mono, sr, self.input_sample_rate, quality="HQ").astype(np.float32)
            else:
                audio_16k = mono

            total_len = len(audio_16k)
            overall_rms = float(np.sqrt(np.mean(audio_16k ** 2))) if total_len > 0 else 0.0
            overall_rms_dbfs = 20.0 * np.log10(max(overall_rms, 1e-10))
            peak_lin = float(np.max(np.abs(samples))) if samples.size > 0 else 0.0
            crest_factor_db = 20.0 * np.log10(max(peak_lin, 1e-10) / max(overall_rms, 1e-10))
            clipping_ratio = float(np.mean(np.abs(samples) >= 0.999)) if samples.size > 0 else 0.0

            # Exact upstream deterministic validation crop (`sonics/utils/dataset.py` lines 65-69):
            # If audio_len < max_len: mode="constant" zero-pad at end.
            # If audio_len > max_len: idx = int((audio_len - max_len) / 4 * 3) (3/4 crop).
            upstream_3_4_start = int((total_len - self.expected_samples) / 4 * 3) if total_len > self.expected_samples else 0
            up_win_np, _ = self._prepare_window(audio_16k, upstream_3_4_start, pad_mode="constant")
            upstream_3_4_prob, upstream_3_4_logit = self._score_single_window(up_win_np)

            # Construct multi-window sampling schedule
            center_start = max(0, (total_len - self.expected_samples) // 2)
            crop_3_4_start = upstream_3_4_start

            window_specs: List[Tuple[str, int]] = []
            if self.window_seconds <= 15.0:
                # Short-context model (e.g., SONICS SpecTTTra-gamma 5s):
                # Evaluate up to 9 windows across the recording to prevent sparse intro/breakdown bias.
                if total_len <= int(self.expected_samples * 1.2):
                    window_specs.append(("center", center_start))
                else:
                    rem = total_len - self.expected_samples
                    margin = int(min(self.input_sample_rate * 4, rem * 0.08))
                    n_grid = min(8, max(3, int(duration_sec // 12.0)))
                    grid_starts = [int(x) for x in np.linspace(margin, max(margin, rem - margin), n_grid)]
                    seen_starts = set()
                    window_specs.append(("center", center_start))
                    seen_starts.add(center_start)
                    if abs(crop_3_4_start - center_start) >= self.input_sample_rate:
                        window_specs.append(("crop_3_4", crop_3_4_start))
                        seen_starts.add(crop_3_4_start)
                    for idx_g, st in enumerate(grid_starts):
                        if all(abs(st - s) >= int(self.input_sample_rate * 1.5) for s in seen_starts):
                            t_sec = st / float(self.input_sample_rate)
                            window_specs.append((f"win_{idx_g + 1}_{t_sec:.1f}s", st))
                            seen_starts.add(st)
            else:
                # Long-context model (e.g., SONICS SpecTTTra-alpha 120s):
                if total_len <= int(self.expected_samples * 1.05):
                    window_specs.append(("center", 0))
                else:
                    rem = total_len - self.expected_samples
                    window_specs.append(("center", center_start))
                    if abs(crop_3_4_start - center_start) >= self.input_sample_rate * 2:
                        window_specs.append(("crop_3_4", crop_3_4_start))
                    early_1_4 = int(rem * 0.25)
                    if abs(early_1_4 - center_start) >= self.input_sample_rate * 2:
                        window_specs.append(("crop_1_4", early_1_4))

            window_scores: Dict[str, float] = {}
            window_logits: Dict[str, float] = {}
            active_probs: List[float] = []
            active_logits: List[float] = []

            for lbl, st in window_specs:
                win_np, w_rms_db = self._prepare_window(audio_16k, st, pad_mode="constant")
                p_val, l_val = self._score_single_window(win_np)
                window_scores[lbl] = round(p_val, 6)
                window_logits[lbl] = round(l_val, 6)
                # Filter out near-silent fade-in/fade-out windows (<-50 dBFS) when aggregating multi-window songs
                if w_rms_db >= -50.0 or len(window_specs) == 1:
                    active_probs.append(p_val)
                    active_logits.append(l_val)

            if not active_probs:
                active_probs = list(window_scores.values())
                active_logits = list(window_logits.values())

            probs_arr = np.array(active_probs, dtype=np.float64)
            logits_arr = np.array(active_logits, dtype=np.float64)

            center_prob = float(window_scores.get("center", probs_arr[0]))
            center_logit = float(window_logits.get("center", logits_arr[0]))
            crop_3_4_prob = float(window_scores.get("crop_3_4", upstream_3_4_prob))
            mean_prob = float(np.mean(probs_arr))
            median_prob = float(np.median(probs_arr))
            q75_prob = float(np.percentile(probs_arr, 75))
            q90_prob = float(np.percentile(probs_arr, 90))
            max_prob = float(np.max(probs_arr))
            std_prob = float(np.std(probs_arr))
            iqr_prob = float(np.percentile(probs_arr, 75) - np.percentile(probs_arr, 25))
            sorted_desc = sorted(active_probs, reverse=True)
            top3_mean_prob = float(np.mean(sorted_desc[: min(3, len(sorted_desc))]))

            sorted_logits_desc = sorted(active_logits, reverse=True)
            top3_mean_logit = float(np.mean(sorted_logits_desc[: min(3, len(sorted_logits_desc))]))
            if self.window_seconds <= 15.0 and len(probs_arr) >= 3:
                primary_prob = top3_mean_prob
                primary_logit = top3_mean_logit
                agg_policy = "multi_window_top3_mean"
            elif self.window_seconds > 15.0 and len(probs_arr) >= 2:
                primary_prob = mean_prob
                primary_logit = float(np.mean(logits_arr))
                agg_policy = "multi_crop_mean"
            else:
                primary_prob = upstream_3_4_prob if total_len < self.expected_samples else center_prob
                primary_logit = upstream_3_4_logit if total_len < self.expected_samples else center_logit
                agg_policy = "upstream_constant_zero_pad" if total_len < self.expected_samples else "center_window"

            # Uncertainty quantification (binary entropy + inter-window dispersion)
            p_clamped = float(np.clip(primary_prob, 1e-6, 1.0 - 1e-6))
            prediction_entropy = float(
                -(p_clamped * np.log2(p_clamped) + (1.0 - p_clamped) * np.log2(1.0 - p_clamped))
            )

            # Input validity warnings (clipping, extreme brickwall mastering, zero-padding)
            input_validity_warnings: List[str] = []
            if clipping_ratio >= 0.005:
                input_validity_warnings.append(
                    f"Hard digital clipping detected ({clipping_ratio * 100:.2f}% samples >= 0.999 FS); "
                    "non-linear harmonic splatter may alter high-frequency mel-spectrogram tokens."
                )
            if crest_factor_db < 5.0 and overall_rms_dbfs >= -50.0:
                input_validity_warnings.append(
                    f"Extreme dynamic compression / brickwall limiting detected (crest factor={crest_factor_db:.2f} dB < 5.0 dB)."
                )
            if total_len < self.expected_samples:
                pad_pct = (1.0 - (total_len / float(self.expected_samples))) * 100.0
                input_validity_warnings.append(
                    f"Input duration ({duration_sec:.2f}s) is shorter than model context ({self.window_seconds:.0f}s); "
                    f"upstream constant zero-padding fills {pad_pct:.1f}% of the window with -80 dB floor silence."
                )

            # Out-of-distribution / context-length abstention check:
            # 1) Digital silence (RMS < -65 dBFS)
            # 2) Short audio (< 115s for 120s SpecTTTra-alpha, or < 3.0s for 5s SpecTTTra-gamma)
            abstained = False
            ood_detected = False
            ood_warning: Optional[str] = None
            ood_reasons: List[str] = []

            if overall_rms_dbfs < -65.0:
                abstained = True
                ood_detected = True
                ood_reasons.append(f"Signal is near-silent (RMS={overall_rms_dbfs:.1f} dBFS < -65.0 dBFS).")
                label = "Abstained (OOD: Near-Silence)"
                confidence = 0.0
                ood_warning = (
                    f"OUT-OF-DISTRIBUTION ABSTENTION: Input audio is near-silent ({overall_rms_dbfs:.1f} dBFS). "
                    "Raw SONICS score is reported unmodified, but classification is abstained."
                )
            elif self.window_seconds >= 60.0 and duration_sec < 115.0:
                abstained = True
                ood_detected = True
                ood_reasons.append(
                    f"Recording duration ({duration_sec:.1f}s) is shorter than the {self.window_seconds:.0f}s "
                    f"context window required by {self.name} (minimum unpadded threshold: 115.0s)."
                )
                label = f"Abstained (Duration < {self.window_seconds:.0f}s Required for SpecTTTra-α)"
                confidence = 0.0
                ood_warning = (
                    f"CONTEXT-LENGTH ABSTENTION: {self.name} requires a full 120-second spectro-temporal context window. "
                    f"Because this recording is only {duration_sec:.1f}s long, upstream constant zero-padding fills "
                    f"{max(0.0, 120.0 - duration_sec):.1f}s with -80 dB mel floor silence (and reflection-padding would "
                    "fabricate artificial palindrome repetitions). Exact upstream zero-padded score is reported unmodified "
                    "for transparency, but classification is abstained; rely on SONICS SpecTTTra-γ (5s) for tracks < 120s."
                )
            elif self.window_seconds <= 15.0 and duration_sec < 3.0:
                abstained = True
                ood_detected = True
                ood_reasons.append(
                    f"Recording duration ({duration_sec:.2f}s) is shorter than 3.0s minimum for 5.0s window."
                )
                label = "Abstained (Duration < 3.0s Too Short for 5s Window)"
                confidence = 0.0
                ood_warning = (
                    f"CONTEXT-LENGTH ABSTENTION: Input duration ({duration_sec:.2f}s) is too short for {self.name} "
                    "(5.0s window). Exact upstream zero-padded score is reported unmodified, but classification is abstained."
                )
            else:
                is_ai = primary_prob >= self.threshold
                label = "Synthetic Song Pattern" if is_ai else "Human / Acoustic Pattern"
                confidence = float(abs(primary_prob - self.threshold) * 2.0)
                if input_validity_warnings:
                    ood_warning = " | ".join(input_validity_warnings)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return DetectorPrediction(
                detector_id=self.detector_id,
                detector_name=self.name,
                detector_version=self.version,
                model_source=self.model_source,
                model_revision=self.model_revision,
                probability=round(primary_prob, 6),
                raw_score=round(primary_logit, 6),
                threshold=self.threshold,
                predicted_label=label,
                confidence=round(confidence, 4),
                input_sample_rate=self.input_sample_rate,
                input_duration_sec=round(duration_sec, 3),
                processing_time_ms=round(elapsed_ms, 2),
                device_used=self.active_device,
                feature_dim=128 * 128,
                status="success",
                error_message=None,
                details={
                    "weights_sha256": self.weights_sha256,
                    "missing_keys_count": len(self.missing_keys),
                    "unexpected_keys_count": len(self.unexpected_keys),
                    "normalization": "upstream_std_unit_variance",
                    "padding_mode": "constant" if total_len < self.expected_samples else "none",
                    "crop_policy": agg_policy,
                    "window_seconds": self.window_seconds,
                    "windows_evaluated": len(window_scores),
                    "active_windows_aggregated": len(active_probs),
                    "upstream_3_4_crop_prob": round(upstream_3_4_prob, 6),
                    "upstream_3_4_crop_logit": round(upstream_3_4_logit, 6),
                    "center_window_prob": round(center_prob, 6),
                    "crop_3_4_prob": round(crop_3_4_prob, 6),
                    "multi_window_mean_prob": round(mean_prob, 6),
                    "multi_window_median_prob": round(median_prob, 6),
                    "multi_window_q75_prob": round(q75_prob, 6),
                    "multi_window_q90_prob": round(q90_prob, 6),
                    "multi_window_top3_mean_prob": round(top3_mean_prob, 6),
                    "multi_window_max_prob": round(max_prob, 6),
                    "multi_window_std_prob": round(std_prob, 6),
                    "multi_window_iqr_prob": round(iqr_prob, 6),
                    "calibrated_probability": round(primary_prob, 6),
                    "prediction_entropy_bits": round(prediction_entropy, 4),
                    "uncertainty_interval_iqr": [
                        round(float(np.percentile(probs_arr, 25)), 5),
                        round(float(np.percentile(probs_arr, 75)), 5),
                    ],
                    "signal_rms_dbfs": round(overall_rms_dbfs, 2),
                    "crest_factor_db": round(crest_factor_db, 2),
                    "clipping_ratio": round(clipping_ratio, 6),
                    "input_validity_warnings": input_validity_warnings,
                    "window_scores": window_scores,
                    "abstained": abstained,
                    "ood_detected": ood_detected,
                    "ood_reasons": ood_reasons,
                    "ood_warning": ood_warning,
                },
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return DetectorPrediction(
                detector_id=self.detector_id,
                detector_name=self.name,
                detector_version=self.version,
                model_source=self.model_source,
                model_revision=self.model_revision,
                probability=None,
                raw_score=None,
                threshold=self.threshold,
                predicted_label="Error",
                confidence=None,
                input_sample_rate=self.input_sample_rate,
                input_duration_sec=round(duration_sec, 3),
                processing_time_ms=round(elapsed_ms, 2),
                device_used=self.active_device,
                feature_dim=128 * 128,
                status="error",
                error_message=str(exc),
            )

    def info(self) -> Dict[str, Any]:
        avail = self.is_available()
        return {
            "detector_id": self.detector_id,
            "name": self.name,
            "version": self.version,
            "model_source": self.model_source,
            "model_revision": self.model_revision,
            "license": self.license,
            "input_sample_rate": self.input_sample_rate,
            "input_duration_requirements": self.input_duration_requirements,
            "supported_devices": self.supported_devices,
            "active_device": self.active_device,
            "available": avail,
            "loaded": self.model is not None,
            "status_message": self.status_message if self.model is not None else ("Available on disk" if avail else f"Optional weights not downloaded ({self.model_dir.name})"),
            "description": self.description,
        }
