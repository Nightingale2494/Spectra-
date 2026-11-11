"""FakeprintDetector adapter for lofcz/ai-music-detector pretrained ONNX model."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime as ort
import yaml

from app.core.config import settings
from app.detectors.base import BaseAudioDetector, DetectorPrediction
from app.services.forensics_service import extract_neural_vocoder_fakeprint


class FakeprintDetector(BaseAudioDetector):
    detector_id = "fakeprint_onnx"
    name = "Neural-Vocoder Fakeprint Detector (ONNX LR)"
    version = "1.0.0-ismir25"
    model_source = "https://github.com/lofcz/ai-music-detector"
    model_revision = settings.SPECTRA_FAKEPRINT_REVISION
    license = "MIT (code) / Research Evaluation (ISMIR 2025 method)"
    input_sample_rate = 16000
    input_duration_requirements = "Up to 300s @ 16,000 Hz mono (8192-point STFT, 1000-8000 Hz)"
    supported_devices = ["cpu"]
    description = (
        "Extracts log-power spectral residuals above the lower spectral hull in the 1-8 kHz band "
        "to detect periodic deconvolution/vocoder artifacts, scored via a pretrained ONNX logistic regression model."
    )

    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        self.model_path = Path(model_path or settings.SPECTRA_FAKEPRINT_MODEL_PATH)
        self.config_path = Path(config_path or settings.SPECTRA_FAKEPRINT_CONFIG_PATH)
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.expected_features: int = 3585
        self.threshold: float = 0.5
        self.active_device: str = "cpu"
        self.status_message: str = "Not loaded"
        self.model_sha256: Optional[str] = None
        self.config: Dict[str, Any] = {
            "audio": {"sample_rate": 16000, "n_fft": 8192, "max_duration": 300},
            "fakeprint": {
                "freq_min": 1000,
                "freq_max": 8000,
                "hull_area": 10,
                "max_db": 5.0,
                "min_db": -45.0,
            },
        }
        self._load_config()

    def _load_config(self) -> None:
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                    if isinstance(loaded, dict):
                        self.config.update(loaded)
            except Exception:
                pass

    def is_available(self) -> bool:
        return self.model_path.exists() and self.model_path.stat().st_size > 0

    def load(self, device: Optional[str] = None) -> bool:
        if not self.is_available():
            self.session = None
            self.status_message = f"Model weights missing at {self.model_path.name}"
            return False

        try:
            raw_bytes = self.model_path.read_bytes()
            self.model_sha256 = hashlib.sha256(raw_bytes).hexdigest()
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=["CPUExecutionProvider"],
            )
            inp = self.session.get_inputs()[0]
            self.input_name = inp.name
            if len(inp.shape) >= 2 and isinstance(inp.shape[1], int):
                self.expected_features = int(inp.shape[1])
            self.active_device = "cpu"
            self.status_message = f"Ready (expected_features={self.expected_features})"
            return True
        except Exception as exc:
            self.session = None
            self.status_message = f"ONNX load error: {exc}"
            return False

    def predict(
        self,
        samples: np.ndarray,
        sr: int,
        precomputed_features: Optional[np.ndarray] = None,
    ) -> DetectorPrediction:
        t0 = time.perf_counter()
        duration_sec = float(samples.shape[1]) / float(sr) if (samples.ndim == 2 and sr > 0) else 0.0

        if self.session is None:
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
                raise ValueError("Invalid audio PCM array or sample rate supplied to FakeprintDetector.")

            if not np.all(np.isfinite(samples)):
                raise ValueError("Audio input contains NaN or Infinity values.")

            audio_cfg = self.config.get("audio", {})
            fp_cfg = self.config.get("fakeprint", {})

            if precomputed_features is not None:
                fakeprint = np.asarray(precomputed_features, dtype=np.float32)
                if fakeprint.ndim != 1:
                    raise ValueError(f"Precomputed fakeprint must be 1D vector, got shape {fakeprint.shape}")
                sig_rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size > 0 else 0.0
                sig_rms_dbfs = 20.0 * np.log10(max(sig_rms, 1e-10))
                active_bin_ratio = float(np.mean(fakeprint > 0.05)) if len(fakeprint) > 0 else 0.0
                fp_meta: Dict[str, Any] = {
                    "signal_rms_dbfs": round(sig_rms_dbfs, 2),
                    "active_bin_ratio": round(active_bin_ratio, 4),
                    "residual_mean_norm": round(float(np.mean(fakeprint)), 4),
                }
            else:
                fp_res = extract_neural_vocoder_fakeprint(
                    samples=samples,
                    sr=sr,
                    target_sr=int(audio_cfg.get("sample_rate", 16000)),
                    n_fft=int(audio_cfg.get("n_fft", 8192)),
                    freq_min=int(fp_cfg.get("freq_min", 1000)),
                    freq_max=int(fp_cfg.get("freq_max", 8000)),
                    hull_area=int(fp_cfg.get("hull_area", 10)),
                    max_db=float(fp_cfg.get("max_db", 5.0)),
                    min_db=float(fp_cfg.get("min_db", -45.0)),
                    max_duration=int(audio_cfg.get("max_duration", 300)),
                )
                fakeprint = fp_res["feature_vector"]
                fp_meta = {
                    "periodicity_score": fp_res["periodicity_score"],
                    "residual_mean_norm": fp_res["residual_mean_norm"],
                    "residual_max_raw_db": fp_res["residual_max_raw_db"],
                    "signal_rms_dbfs": fp_res.get("signal_rms_dbfs", -100.0),
                    "band_1k_8k_energy_ratio": fp_res.get("band_1k_8k_energy_ratio", 0.0),
                    "active_bin_ratio": fp_res.get("active_bin_ratio", 0.0),
                    "comb_grid_200hz_mean": fp_res.get("comb_grid_200hz_mean", 0.0),
                    "comb_grid_200hz_contrast": fp_res.get("comb_grid_200hz_contrast", 0.0),
                    "temporal_stationarity": fp_res.get("temporal_stationarity", 0.0),
                }

            if len(fakeprint) != self.expected_features:
                raise ValueError(
                    f"Feature dimension mismatch: extracted {len(fakeprint)} bins, model expects {self.expected_features} bins."
                )

            input_tensor = fakeprint.reshape(1, self.expected_features).astype(np.float32)
            assert self.session is not None and self.input_name is not None
            outputs = self.session.run(None, {self.input_name: input_tensor})

            # Handle ONNX output layout safely without manipulating raw model probability
            out_arr = np.asarray(outputs[0])
            prob = float(out_arr.reshape(-1)[0])
            prob = float(np.clip(prob, 0.0, 1.0))

            # Compute logit and temperature-scaled calibrated probability + binary entropy uncertainty
            # Raw ONNX logistic regression produces extreme saturation (0.0000 or 1.0000).
            # Temperature T = 3.25 (fitted on disjoint calibration split) softens extreme logits
            # for calibrated uncertainty reporting while keeping raw probability 100% unmodified.
            p_clamped = float(np.clip(prob, 1e-6, 1.0 - 1e-6))
            raw_logit = float(np.log(p_clamped / (1.0 - p_clamped)))
            calibrated_prob = float(1.0 / (1.0 + np.exp(-raw_logit / 3.25)))
            p_ent = float(np.clip(calibrated_prob, 1e-9, 1.0 - 1e-9))
            prediction_entropy = float(
                -(p_ent * np.log2(p_ent) + (1.0 - p_ent) * np.log2(1.0 - p_ent))
            )

            fp_meta["raw_logit"] = round(raw_logit, 4)
            fp_meta["calibrated_probability"] = round(calibrated_prob, 6)
            fp_meta["temperature_scaling_T"] = 3.25
            fp_meta["prediction_entropy_bits"] = round(prediction_entropy, 4)
            fp_meta["architecture_scope"] = (
                "Targets periodic spectral residual peaks in the 1–8 kHz band (e.g., 200/400 Hz grid spacing) "
                "described in open-weights transposed-convolution vocoder studies (Afchar et al., ISMIR 2025) "
                "and empirically correlated with commercial Suno/Udio recordings. Does not prove undocumented "
                "proprietary generator layer topologies, and does not detect continuous latent diffusion VAEs "
                "(Stable Audio, ACE-Step) that lack periodic spectral hull peaks."
            )

            # Input signal validity measurements (clipping, dynamic range, duration)
            sig_rms_lin = float(np.sqrt(np.mean(samples ** 2))) if samples.size > 0 else 0.0
            sig_peak_lin = float(np.max(np.abs(samples))) if samples.size > 0 else 0.0
            crest_factor_db = float(20.0 * np.log10(max(sig_peak_lin, 1e-10) / max(sig_rms_lin, 1e-10)))
            clipping_ratio = float(np.mean(np.abs(samples) >= 0.999)) if samples.size > 0 else 0.0
            fp_meta["crest_factor_db"] = round(crest_factor_db, 2)
            fp_meta["clipping_ratio"] = round(clipping_ratio, 6)

            input_validity_warnings: List[str] = []
            if clipping_ratio >= 0.005:
                input_validity_warnings.append(
                    f"Hard digital clipping detected ({clipping_ratio * 100:.2f}% samples >= 0.999 FS); "
                    "broadband harmonic distortion can artificially elevate 1–8 kHz spectral hull residuals."
                )
            if crest_factor_db < 5.0 and sig_rms_lin > 1e-3:
                input_validity_warnings.append(
                    f"Extreme dynamic compression / brickwall limiting detected (crest factor={crest_factor_db:.2f} dB < 5.0 dB)."
                )
            if duration_sec < 3.0:
                input_validity_warnings.append(
                    f"Short recording duration ({duration_sec:.2f}s < 3.0s); time-averaged 8192-point STFT has high variance."
                )
            fp_meta["input_validity_warnings"] = input_validity_warnings

            # Evaluate Out-Of-Distribution (OOD) / insufficient broadband 1-8 kHz support
            sig_rms_dbfs = float(fp_meta.get("signal_rms_dbfs", 0.0))
            res_max_db = float(fp_meta.get("residual_max_raw_db", 5.0))
            band_energy_ratio = float(fp_meta.get("band_1k_8k_energy_ratio", 1.0))
            active_ratio = float(fp_meta.get("active_bin_ratio", 1.0))
            stationarity = float(fp_meta.get("temporal_stationarity", 0.5))

            ood_reasons: List[str] = []
            if duration_sec < 1.5:
                ood_reasons.append(
                    f"Recording duration ({duration_sec:.2f}s) is shorter than 1.5s minimum required for stable 8192-point STFT averaging."
                )
            if sig_rms_dbfs < -65.0 or res_max_db < 0.25:
                ood_reasons.append(
                    f"Near-silence or negligible 1–8 kHz spectral prominence (RMS={sig_rms_dbfs:.1f} dBFS, max_residual={res_max_db:.2f} dB)."
                )
            if band_energy_ratio < 0.005:
                ood_reasons.append(
                    f"Less than 0.5% of signal energy falls inside the 1,000–8,000 Hz analysis band (ratio={band_energy_ratio:.5f})."
                )
            if active_ratio < 0.025:
                ood_reasons.append(
                    f"Sparse narrowband residual ({active_ratio * 100:.2f}% active bins > 0.05 in 1–8 kHz; broadband music requires >= 2.5%)."
                )

            ood_detected = len(ood_reasons) > 0
            if ood_detected:
                label = "Abstained (OOD: Silence / Non-Broadband Signal)"
                confidence = 0.0
                ood_warning = (
                    "OUT-OF-DISTRIBUTION ABSTENTION: The input lacks broadband 1–8 kHz musical energy or minimum duration "
                    f"({' '.join(ood_reasons)}). Because the pretrained ONNX logistic regression model "
                    "has a positive bias (b = +4.9827) and negative broadband weight sum (sum(w) = -79.26), "
                    "zero or sparse residual vectors fail to activate negative weights and default to "
                    "sigmoid(+4.9827) ≈ 0.9932. Raw ONNX probability is reported unmodified for transparency, "
                    "but classification is abstained."
                )
                fp_meta["abstained"] = True
                fp_meta["ood_detected"] = True
                fp_meta["ood_reasons"] = ood_reasons
                fp_meta["ood_warning"] = ood_warning
            else:
                is_ai = prob >= self.threshold
                label = "AI-Artifact Signature Detected" if is_ai else "No Strong Vocoder Fakeprint"
                confidence = float(abs(prob - self.threshold) * 2.0)
                fp_meta["abstained"] = False
                fp_meta["ood_detected"] = False
                fp_meta["ood_reasons"] = []
                warn_parts: List[str] = []
                if is_ai and stationarity < 0.25:
                    warn_parts.append(
                        f"Forensic Caution: High 1–8 kHz residual score (P={prob:.4f}) accompanied by low "
                        f"temporal stationarity ({stationarity:.2f} < 0.25). Non-stationary high-frequency distortion "
                        "(such as overdriven electric guitar amplifiers or aggressive DAW clipping/remixes) can "
                        "excite 200/400 Hz harmonic bins; cross-check with SONICS SpecTTTra detectors."
                    )
                if input_validity_warnings:
                    warn_parts.extend(input_validity_warnings)
                fp_meta["ood_warning"] = " | ".join(warn_parts) if warn_parts else None

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return DetectorPrediction(
                detector_id=self.detector_id,
                detector_name=self.name,
                detector_version=self.version,
                model_source=self.model_source,
                model_revision=self.model_revision,
                probability=round(prob, 6),
                raw_score=round(prob, 6),
                threshold=self.threshold,
                predicted_label=label,
                confidence=round(confidence, 4),
                input_sample_rate=self.input_sample_rate,
                input_duration_sec=round(min(duration_sec, 300.0), 3),
                processing_time_ms=round(elapsed_ms, 2),
                device_used=self.active_device,
                feature_dim=self.expected_features,
                status="success",
                error_message=None,
                details={
                    "model_sha256": self.model_sha256,
                    "preprocessing": {
                        "sample_rate": int(audio_cfg.get("sample_rate", 16000)),
                        "n_fft": int(audio_cfg.get("n_fft", 8192)),
                        "freq_min_hz": int(fp_cfg.get("freq_min", 1000)),
                        "freq_max_hz": int(fp_cfg.get("freq_max", 8000)),
                        "hull_area": int(fp_cfg.get("hull_area", 10)),
                    },
                    **fp_meta,
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
                feature_dim=self.expected_features,
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
            "loaded": self.session is not None,
            "status_message": self.status_message if self.session is not None else ("Available on disk" if avail else "Model weights missing"),
            "description": self.description,
        }
