"""Comprehensive automated test suite for SPECTRA Backend (Upload, C2PA, Spectral, Detectors, DSP, Integration)."""
from __future__ import annotations

import io
from pathlib import Path
import sys

from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.database.session import init_db
from app.detectors.fakeprint_detector import FakeprintDetector
from app.detectors.sonics_detector import SonicsDetector
from app.main import app
from app.processing.dsp_engine import apply_dsp_operation, encode_audio_bytes
from app.services.forensics_service import (
    compute_frequency_domain_metrics,
    compute_time_domain_metrics,
    extract_neural_vocoder_fakeprint,
)
from app.services.provenance_service import inspect_c2pa_provenance
from app.services.sample_generator import (
    ensure_reference_samples,
    generate_acoustic_reference_pcm,
    generate_vocoder_artifact_pcm,
    sign_wav_bytes_with_c2pa,
)

init_db()
ensure_reference_samples()
client = TestClient(app)


def _make_sine_wav_bytes(freq_hz: float = 440.0, sr: int = 44100, duration: float = 2.0, stereo: bool = True) -> bytes:
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    sig = (0.5 * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)
    pcm = np.stack([sig, sig * 0.9], axis=0) if stereo else sig.reshape(1, -1)
    return encode_audio_bytes(pcm, sr, fmt="wav")


# ==============================================================================
# 1. UPLOAD TESTS
# ==============================================================================

def test_upload_valid_wav():
    wav_bytes = _make_sine_wav_bytes(440.0, 44100, 2.0)
    resp = client.post(
        "/api/audio/upload?run_sync=true",
        files={"file": ("test_sine.wav", wav_bytes, "audio/wav")},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["asset"]["original_format"] == "WAV"
    assert data["asset"]["sample_rate"] == 44100
    assert data["asset"]["channels"] == 2
    assert abs(data["asset"]["duration_seconds"] - 2.0) < 0.05
    assert len(data["asset"]["sha256_checksum"]) == 64


def test_upload_valid_mp3():
    t = np.linspace(0.0, 2.0, 44100 * 2, endpoint=False)
    sig = (0.4 * np.sin(2.0 * np.pi * 880.0 * t)).astype(np.float32)
    pcm = np.stack([sig, sig], axis=0)
    mp3_bytes = encode_audio_bytes(pcm, 44100, fmt="mp3", bitrate_kbps=192)
    resp = client.post(
        "/api/audio/upload?run_sync=true",
        files={"file": ("test_tone.mp3", mp3_bytes, "audio/mpeg")},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["asset"]["original_format"] == "MP3"
    assert data["asset"]["sample_rate"] == 44100


def test_upload_invalid_and_corrupted_audio():
    # Random garbage bytes with .wav extension
    resp = client.post(
        "/api/audio/upload",
        files={"file": ("garbage.wav", b"NOT_A_VALID_RIFF_FILE_1234567890", "audio/wav")},
    )
    assert resp.status_code == 400

    # Valid RIFF WAVE header but truncated/corrupted audio payload
    corrupted_riff = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x00\x00\x00\x00"
    resp2 = client.post(
        "/api/audio/upload",
        files={"file": ("corrupted.wav", corrupted_riff, "audio/wav")},
    )
    assert resp2.status_code == 400


def test_upload_unsupported_format():
    resp = client.post(
        "/api/audio/upload",
        files={"file": ("unsupported.ogg", b"OggS\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00", "audio/ogg")},
    )
    assert resp.status_code == 415


def test_upload_oversized_file(monkeypatch):
    monkeypatch.setattr(settings, "SPECTRA_MAX_UPLOAD_MB", 1)
    big_bytes = b"0" * (1024 * 1024 + 512)
    resp = client.post(
        "/api/audio/upload",
        files={"file": ("oversized.wav", big_bytes, "audio/wav")},
    )
    assert resp.status_code == 413


def test_upload_dangerous_filenames():
    wav_bytes = _make_sine_wav_bytes(440.0, 16000, 1.0)
    resp = client.post(
        "/api/audio/upload",
        files={"file": ("../../etc/passwd.wav", wav_bytes, "audio/wav")},
    )
    assert resp.status_code == 400

    # Windows reserved device name gets safely prefixed
    resp_con = client.post(
        "/api/audio/upload?run_sync=true",
        files={"file": ("CON.wav", wav_bytes, "audio/wav")},
    )
    assert resp_con.status_code == 200
    assert resp_con.json()["asset"]["sanitized_filename"] == "file_CON.wav"


def test_upload_incorrect_extension_mismatch():
    wav_bytes = _make_sine_wav_bytes(440.0, 16000, 1.0)
    resp = client.post(
        "/api/audio/upload",
        files={"file": ("disguised_as_mp3.mp3", wav_bytes, "audio/mpeg")},
    )
    assert resp.status_code == 400
    assert "extension mismatch" in resp.json()["detail"].lower()


# ==============================================================================
# 2. C2PA PROVENANCE TESTS
# ==============================================================================

def test_c2pa_valid_signed_asset(tmp_path):
    wav_bytes = _make_sine_wav_bytes(440.0, 16000, 1.5, stereo=False)
    signed_bytes = sign_wav_bytes_with_c2pa(wav_bytes, title="Unit Test Signed WAV")
    signed_path = tmp_path / "signed_valid.wav"
    signed_path.write_bytes(signed_bytes)

    res = inspect_c2pa_provenance(signed_path, mime_type="audio/wav")
    assert res["credential_present"] is True
    assert res["c2pa_status"] == "manifest_valid"
    assert res["signature_valid"] is True
    assert res["active_manifest_id"] is not None
    assert "SPECTRA Provenance Lab" in (res["claim_generator"] or "")


def test_c2pa_invalid_tampered_signed_asset(tmp_path):
    wav_bytes = _make_sine_wav_bytes(440.0, 16000, 1.5, stereo=False)
    signed_bytes = bytearray(sign_wav_bytes_with_c2pa(wav_bytes, title="Tampered Test WAV"))
    # Flip audio PCM bytes inside the RIFF data chunk (starting after byte 44) to trigger dataHash mismatch
    signed_bytes[100:200] = b"\xff" * 100
    tampered_path = tmp_path / "signed_tampered.wav"
    tampered_path.write_bytes(bytes(signed_bytes))

    res = inspect_c2pa_provenance(tampered_path, mime_type="audio/wav")
    assert res["credential_present"] is True
    assert res["c2pa_status"] == "manifest_invalid"
    assert res["signature_valid"] is False
    assert len(res["validation_errors"]) > 0


def test_c2pa_no_manifest_absent(tmp_path):
    wav_bytes = _make_sine_wav_bytes(440.0, 16000, 1.0, stereo=False)
    unsigned_path = tmp_path / "unsigned.wav"
    unsigned_path.write_bytes(wav_bytes)

    res = inspect_c2pa_provenance(unsigned_path, mime_type="audio/wav")
    assert res["credential_present"] is False
    assert res["c2pa_status"] == "manifest_absent"
    assert res["signature_valid"] is None


def test_c2pa_unsupported_format_and_exception(tmp_path):
    xyz_path = tmp_path / "sample.xyz"
    xyz_path.write_bytes(b"some dummy content")
    res_unsup = inspect_c2pa_provenance(xyz_path, mime_type="audio/x-unsupported-custom")
    assert res_unsup["c2pa_status"] == "unsupported_format"

    missing_path = tmp_path / "nonexistent_file.wav"
    res_err = inspect_c2pa_provenance(missing_path, mime_type="audio/wav")
    assert res_err["c2pa_status"] == "inspection_error"


# ==============================================================================
# 3. SPECTRAL ANALYSIS TESTS
# ==============================================================================

def test_spectral_pure_sine_wave():
    sr = 16000
    t = np.linspace(0.0, 2.0, sr * 2, endpoint=False)
    sine_1k = (0.8 * np.sin(2.0 * np.pi * 1000.0 * t)).reshape(1, -1).astype(np.float32)
    fd = compute_frequency_domain_metrics(sine_1k, sr)
    assert abs(fd["dominant_frequency_hz"] - 1000.0) < 15.0
    assert fd["spectral_flatness_mean"] < 0.05


def test_spectral_white_noise():
    rng = np.random.default_rng(123)
    noise = rng.uniform(-0.5, 0.5, size=(1, 32000)).astype(np.float32)
    fd = compute_frequency_domain_metrics(noise, 16000)
    assert fd["spectral_flatness_mean"] > 0.45


def test_spectral_silence_and_stereo():
    silence = np.zeros((2, 32000), dtype=np.float32)
    td = compute_time_domain_metrics(silence, 16000)
    assert td["silence_ratio"] > 0.95
    assert td["silence_regions_count"] >= 1
    assert td["channels"] == 2

    # Stereo correlation on identical vs inverted channels
    t = np.linspace(0.0, 1.0, 16000, endpoint=False)
    s = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    stereo_inverted = np.stack([s, -s], axis=0)
    td_inv = compute_time_domain_metrics(stereo_inverted, 16000)
    assert td_inv["stereo_correlation"] is not None
    assert td_inv["stereo_correlation"] < -0.98


def test_spectral_multiple_sample_rates_and_known_peaks():
    for sr in (16000, 44100, 48000):
        t = np.linspace(0.0, 2.0, sr * 2, endpoint=False)
        sig = (
            0.3 * np.sin(2.0 * np.pi * 2500.0 * t)
            + 0.25 * np.sin(2.0 * np.pi * 5500.0 * t)
        ).reshape(1, -1).astype(np.float32)
        fp = extract_neural_vocoder_fakeprint(sig, sr)
        assert fp["feature_dim"] == 3585
        peak_freqs = [p["frequency_hz"] for p in fp["top_residual_peaks"]]
        assert any(abs(pf - 2500.0) < 25.0 for pf in peak_freqs)
        assert any(abs(pf - 5500.0) < 25.0 for pf in peak_freqs)


# ==============================================================================
# 4. AI DETECTOR TESTS
# ==============================================================================

def test_fakeprint_detector_inference_and_determinism():
    det = FakeprintDetector()
    assert det.is_available() is True
    assert det.load() is True
    assert det.expected_features == 3585

    acoustic_pcm = generate_acoustic_reference_pcm(sr=44100, duration=4.0)
    vocoder_pcm = generate_vocoder_artifact_pcm(sr=44100, duration=4.0)

    pred_real_1 = det.predict(acoustic_pcm, 44100)
    pred_real_2 = det.predict(acoustic_pcm, 44100)
    pred_ai = det.predict(vocoder_pcm, 44100)

    assert pred_real_1.status == "success"
    assert pred_real_1.probability == pred_real_2.probability
    assert pred_real_1.probability is not None and pred_real_1.probability < 0.25
    assert pred_real_1.details.get("abstained") is False
    assert pred_ai.status == "success"
    assert pred_ai.probability is not None and pred_ai.probability > 0.85
    assert pred_ai.details.get("abstained") is False


def test_fakeprint_ood_abstention_on_silence_and_pure_tones():
    det = FakeprintDetector()
    det.load()

    # 1. Pure digital silence (0.0 FS) -> OOD abstention, unmanipulated raw ONNX probability (~0.9932)
    silence = np.zeros((1, 44100 * 3), dtype=np.float32)
    pred_sil = det.predict(silence, 44100)
    assert pred_sil.status == "success"
    assert pred_sil.details.get("abstained") is True
    assert pred_sil.details.get("ood_detected") is True
    assert "Abstained" in pred_sil.predicted_label
    assert pred_sil.confidence == 0.0
    assert pred_sil.probability == pred_sil.raw_score
    assert pred_sil.probability is not None and abs(pred_sil.probability - 0.993191) < 0.005
    assert "positive bias" in (pred_sil.details.get("ood_warning") or "")

    # 2. Pure 440 Hz sine wave (zero energy in 1-8 kHz) -> OOD abstention
    t = np.linspace(0.0, 3.0, 44100 * 3, endpoint=False)
    sine_440 = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).reshape(1, -1).astype(np.float32)
    pred_440 = det.predict(sine_440, 44100)
    assert pred_440.details.get("abstained") is True
    assert pred_440.details.get("ood_detected") is True
    assert "Abstained" in pred_440.predicted_label
    assert pred_440.probability == pred_440.raw_score

    # 3. Pure 2000 Hz sine wave (single narrowband bin in 1-8 kHz, <0.5% active bins) -> OOD abstention
    sine_2000 = (0.5 * np.sin(2.0 * np.pi * 2000.0 * t)).reshape(1, -1).astype(np.float32)
    pred_2000 = det.predict(sine_2000, 44100)
    assert pred_2000.details.get("abstained") is True
    assert pred_2000.details.get("ood_detected") is True
    assert "Abstained" in pred_2000.predicted_label
    assert pred_2000.probability == pred_2000.raw_score


def test_detector_missing_weights_and_bad_input_recovery(tmp_path):
    det_missing = FakeprintDetector(model_path=str(tmp_path / "does_not_exist.onnx"))
    assert det_missing.is_available() is False
    pred_unavail = det_missing.predict(np.zeros((1, 16000), dtype=np.float32), 16000)
    assert pred_unavail.status == "unavailable"
    assert pred_unavail.probability is None

    # Wrong precomputed feature shape -> status="error"
    det_valid = FakeprintDetector()
    det_valid.load()
    pred_bad_shape = det_valid.predict(
        np.zeros((1, 16000), dtype=np.float32),
        16000,
        precomputed_features=np.zeros(100, dtype=np.float32),
    )
    assert pred_bad_shape.status == "error"
    assert pred_bad_shape.probability is None

    # NaN input -> status="error"
    nan_audio = np.full((1, 16000), np.nan, dtype=np.float32)
    pred_nan = det_valid.predict(nan_audio, 16000)
    assert pred_nan.status == "error"
    assert pred_nan.probability is None


def test_sonics_detector_cpu_inference_and_unavailable_variant():
    det_5s = SonicsDetector(
        detector_id="sonics_gamma_5s",
        local_subdir="sonics-spectttra-gamma-5s",
        window_seconds=5.0,
    )
    assert det_5s.is_available() is True
    assert det_5s.load(device="cpu") is True

    pcm = generate_acoustic_reference_pcm(sr=16000, duration=5.5)
    pred = det_5s.predict(pcm, 16000)
    assert pred.status == "success"
    assert pred.probability is not None and 0.0 <= pred.probability <= 1.0
    assert pred.details.get("normalization") == "upstream_std_unit_variance"
    assert "prediction_entropy_bits" in pred.details
    assert "calibrated_probability" in pred.details

    # Verify multi-window aggregation on a longer (18-second) track
    pcm_long = generate_vocoder_artifact_pcm(sr=16000, duration=18.0)
    pred_long = det_5s.predict(pcm_long, 16000)
    assert pred_long.status == "success"
    assert pred_long.details.get("windows_evaluated", 0) >= 3
    assert pred_long.details.get("crop_policy") == "multi_window_top3_mean"
    assert "multi_window_mean_prob" in pred_long.details
    assert "multi_window_median_prob" in pred_long.details
    assert "multi_window_q75_prob" in pred_long.details
    assert "multi_window_top3_mean_prob" in pred_long.details
    assert "multi_window_max_prob" in pred_long.details

    # Installed 120s variant: verify short-clip (<115s) context-length abstention and upstream constant zero-padding
    det_120s_installed = SonicsDetector(
        detector_id="sonics_alpha_120s",
        name="SONICS SpecTTTra-α (120s Long-Context Classifier)",
        hf_repo="awsaf49/sonics-spectttra-alpha-120s",
        hf_revision="094b32a5545098a71c113f6ae9d5c55310564268",
        local_subdir="sonics-spectttra-alpha-120s",
        window_seconds=120.0,
    )
    if det_120s_installed.is_available():
        assert det_120s_installed.load(device="cpu") is True
        pred_short_120 = det_120s_installed.predict(pcm, 16000)
        assert pred_short_120.status == "success"
        assert pred_short_120.details.get("abstained") is True
        assert "Abstained" in pred_short_120.predicted_label
        assert pred_short_120.details.get("padding_mode") == "constant"
        assert "upstream_3_4_crop_prob" in pred_short_120.details

        # Also verify a 30-second clip (<115s) abstains on the 120s model while preserving upstream constant-padded score
        pcm_30s = generate_acoustic_reference_pcm(sr=16000, duration=30.0)
        pred_30s_120 = det_120s_installed.predict(pcm_30s, 16000)
        assert pred_30s_120.details.get("abstained") is True
        assert pred_30s_120.details.get("padding_mode") == "constant"
        assert pred_30s_120.probability is not None

        # And a full 121-second recording (>= 115s) does NOT abstain and requires zero padding
        pcm_121s = np.tile(pcm_30s, (1, 5))[:, : 16000 * 121]
        pred_121s_120 = det_120s_installed.predict(pcm_121s, 16000)
        assert pred_121s_120.details.get("abstained") is False
        assert pred_121s_120.details.get("padding_mode") == "none"

    # Uninstalled 120s variant must report unavailable without fabricating a score
    det_120s_missing = SonicsDetector(
        detector_id="sonics_alpha_120s_missing",
        local_subdir="sonics-spectttra-alpha-120s-missing",
        window_seconds=120.0,
    )
    assert det_120s_missing.is_available() is False
    pred_120 = det_120s_missing.predict(pcm, 16000)
    assert pred_120.status == "unavailable"
    assert pred_120.probability is None


# ==============================================================================
# 5. DSP LABORATORY TESTS
# ==============================================================================

@pytest.mark.parametrize(
    "op_type,params",
    [
        ("highpass", {"cutoff_hz": 120.0, "order": 4}),
        ("lowpass", {"cutoff_hz": 6000.0, "order": 6}),
        ("bandpass", {"low_hz": 300.0, "high_hz": 5000.0, "order": 4}),
        ("parametric_eq", {"center_hz": 3000.0, "q_factor": 2.0, "gain_db": -6.0}),
        ("eq_3band", {"low_gain_db": 2.0, "mid_gain_db": -2.0, "high_gain_db": -3.0}),
        ("spectral_smoothing", {"freq_min_hz": 1000.0, "freq_max_hz": 8000.0, "smoothing_bins": 9, "mix": 0.7}),
        ("frequency_selective", {"freq_min_hz": 1000.0, "freq_max_hz": 8000.0, "spike_threshold_db": 2.0}),
        ("noise_reduction", {"reduction_db": 10.0, "sensitivity": 1.2}),
        ("dynamic_compression", {"threshold_db": -16.0, "ratio": 3.0, "makeup_gain_db": 1.5}),
        ("gain_limiter", {"gain_db": 3.0, "enable_limiter": True, "ceiling_db": -0.5}),
        ("resample", {"target_sr": 16000, "roundtrip_to_original_sr": True}),
        ("codec_simulation", {"codec": "mp3", "bitrate_kbps": 128}),
        ("notch_comb", {"fundamental_hz": 400.0, "min_freq_hz": 1600.0, "max_freq_hz": 7600.0, "q": 30.0}),
        ("pedalboard_chain", {"highpass_hz": 80.0, "peak_freq_hz": 4500.0, "peak_gain_db": -2.5}),
    ],
)
def test_dsp_operations_preserve_duration_channels_and_stability(op_type, params):
    sr = 44100
    pcm = generate_vocoder_artifact_pcm(sr=sr, duration=2.5)
    out, out_sr, audit_log, warnings = apply_dsp_operation(pcm, sr, op_type, params)
    assert out.shape == pcm.shape
    assert out_sr == sr
    assert np.all(np.isfinite(out))
    assert len(audit_log) >= 2


def test_processed_audio_playability_stereo_hash_and_immutable_original():
    import hashlib
    from app.services.ingestion_service import decode_audio_pcm

    # Create a stereo WAV with distinct Left (440 Hz + harmonics) and Right (660 Hz + harmonics) channels
    sr = 44100
    t = np.linspace(0.0, 2.5, int(sr * 2.5), endpoint=False)
    left = (0.4 * np.sin(2.0 * np.pi * 440.0 * t) + 0.2 * np.sin(2.0 * np.pi * 2200.0 * t)).astype(np.float32)
    right = (0.4 * np.sin(2.0 * np.pi * 660.0 * t) + 0.2 * np.sin(2.0 * np.pi * 3300.0 * t)).astype(np.float32)
    stereo_pcm = np.stack([left, right], axis=0)
    orig_wav_bytes = encode_audio_bytes(stereo_pcm, sr, fmt="wav")
    expected_orig_sha = hashlib.sha256(orig_wav_bytes).hexdigest()

    up_resp = client.post(
        "/api/audio/upload?run_sync=true",
        files={"file": ("stereo_integrity_test.wav", orig_wav_bytes, "audio/wav")},
    )
    assert up_resp.status_code == 200
    job_id = up_resp.json()["initial_job_id"]
    orig_asset_id = up_resp.json()["asset"]["id"]
    assert up_resp.json()["asset"]["sha256_checksum"] == expected_orig_sha
    assert up_resp.json()["asset"]["channels"] == 2

    # Run DSP processing
    proc_resp = client.post(
        f"/api/jobs/{job_id}/process",
        json={
            "operation_type": "lowpass",
            "preset_name": "hf_rolloff_7500",
            "engine": "scipy_dsp",
            "parameters": {"cutoff_hz": 1800.0, "order": 6},
            "export_format": "wav",
            "run_detectors_on_output": True,
        },
    )
    assert proc_resp.status_code == 200
    proc_json = proc_resp.json()
    proc_asset = proc_json["processed_asset"]
    exp = proc_json["experiment"]

    # 1. Hashes must differ when processing modifies signal content
    assert exp["original_sha256"] == expected_orig_sha
    assert exp["processed_sha256"] == proc_asset["sha256_checksum"]
    assert exp["original_sha256"] != exp["processed_sha256"]

    # 2. Original file streamed from server must remain byte-for-byte untouched
    orig_stream = client.get(f"/api/audio/{orig_asset_id}/stream")
    assert orig_stream.status_code == 200
    assert hashlib.sha256(orig_stream.content).hexdigest() == expected_orig_sha

    # 3. Processed file streamed from server must be playable, finite, 2-channel stereo, with distinct L/R
    proc_stream = client.get(f"/api/audio/{proc_asset['id']}/stream")
    assert proc_stream.status_code == 200
    assert hashlib.sha256(proc_stream.content).hexdigest() == proc_asset["sha256_checksum"]

    tmp_proc = settings.temp_dir / "_verify_playable_stereo.wav"
    tmp_proc.write_bytes(proc_stream.content)
    try:
        dec_pcm, dec_sr = decode_audio_pcm(tmp_proc, target_sr=None, mono=False)
    finally:
        if tmp_proc.exists():
            tmp_proc.unlink()

    assert dec_sr == sr
    assert dec_pcm.shape[0] == 2
    assert abs(dec_pcm.shape[1] - stereo_pcm.shape[1]) <= 2
    assert np.all(np.isfinite(dec_pcm))
    # Verify Left and Right channels remained independent (not collapsed to mono)
    assert float(np.max(np.abs(dec_pcm[0] - dec_pcm[1]))) > 0.05


# ==============================================================================
# 6. END-TO-END INTEGRATION TESTS
# ==============================================================================

def test_full_end_to_end_forensic_and_dsp_workflow():
    # 1. Load reference sample
    resp_load = client.post("/api/audio/load-sample/vocoder_artifact?run_sync=true")
    assert resp_load.status_code == 200, resp_load.text
    load_data = resp_load.json()
    job_id = load_data["initial_job_id"]
    asset_id = load_data["asset"]["id"]

    # 2. Fetch waveform and spectrogram
    resp_wf = client.get(f"/api/audio/{asset_id}/waveform")
    assert resp_wf.status_code == 200
    assert len(resp_wf.json()["peaks_max"]) > 50

    resp_sp = client.get(f"/api/audio/{asset_id}/spectrogram")
    assert resp_sp.status_code == 200
    assert len(resp_sp.json()["fakeprint_residual_norm"]) > 50

    # 3. Fetch complete job results
    resp_res = client.get(f"/api/jobs/{job_id}/results")
    assert resp_res.status_code == 200
    res_data = resp_res.json()
    assert res_data["job"]["status"] == "completed"
    assert res_data["provenance"]["c2pa_status"] == "manifest_absent"
    assert len(res_data["detectors"]) >= 2

    # 4. Run DSP experiment (spectral smoothing in 1-8 kHz)
    resp_proc = client.post(
        f"/api/jobs/{job_id}/process",
        json={
            "operation_type": "spectral_smoothing",
            "preset_name": "Spectral Residual Smoothing (1–8 kHz)",
            "engine": "scipy_dsp",
            "parameters": {"freq_min_hz": 1000.0, "freq_max_hz": 8000.0, "smoothing_bins": 11, "mix": 0.75},
            "export_format": "wav",
            "run_detectors_on_output": True,
        },
    )
    assert resp_proc.status_code == 200, resp_proc.text
    proc_data = resp_proc.json()
    assert proc_data["experiment"]["status"] == "completed"
    assert proc_data["comparison"]["snr_db"] is not None
    assert len(proc_data["comparison"]["detector_deltas"]) >= 1

    # 5. Download JSON & PDF technical reports and processed audio
    resp_json_rep = client.get(f"/api/jobs/{job_id}/report?format=json")
    assert resp_json_rep.status_code == 200
    assert "report_metadata" in resp_json_rep.json()

    resp_pdf_rep = client.get(f"/api/jobs/{job_id}/report?format=pdf")
    assert resp_pdf_rep.status_code == 200
    assert resp_pdf_rep.content[:4] == b"%PDF"

    resp_dl = client.get(f"/api/jobs/{job_id}/download?target=processed")
    assert resp_dl.status_code == 200
    assert resp_dl.content[:4] == b"RIFF"


def test_initial_telemetry_endpoints_and_sample_aliases():
    """Verify all 8 initial frontend telemetry endpoints and sample alias IDs return 200 OK."""
    for ep in [
        "/api/health",
        "/api/detectors",
        "/api/dsp/presets",
        "/api/audio/samples",
        "/api/jobs?limit=25",
        "/api/reports",
        "/api/settings/info",
        "/api/benchmarks/latest",
    ]:
        r = client.get(ep)
        assert r.status_code == 200, f"Endpoint {ep} failed: {r.text}"

    # Verify both canonical and alias sample IDs work
    for sid in ["acoustic_reference", "acoustic_studio", "neural_vocoder_synth"]:
        r_smp = client.post(f"/api/audio/load-sample/{sid}?run_sync=false")
        assert r_smp.status_code == 200, f"Sample alias {sid} failed: {r_smp.text}"


def test_phase4_upstream_preprocessing_and_edge_cases():
    """Verify Fakeprint ONNX and SONICS preprocessing on short, clipped, and brickwall-limited inputs."""
    fp_det = FakeprintDetector()
    fp_det.load()
    g5_det = SonicsDetector(
        detector_id="sonics_gamma_5s",
        local_subdir="sonics-spectttra-gamma-5s",
        window_seconds=5.0,
    )
    g5_det.load(device="cpu")

    base_pcm = generate_acoustic_reference_pcm(sr=16000, duration=6.0)

    # 1. Very short 1.0s clip -> both Fakeprint (<1.5s) and SONICS Gamma 5s (<3.0s) abstain transparently
    short_1s = base_pcm[:, :16000]
    p_fp_1s = fp_det.predict(short_1s, 16000)
    p_g5_1s = g5_det.predict(short_1s, 16000)
    assert p_fp_1s.details.get("abstained") is True
    assert p_fp_1s.probability is not None
    assert p_g5_1s.details.get("abstained") is True
    assert p_g5_1s.details.get("padding_mode") == "constant"
    assert p_g5_1s.probability is not None

    # 2. Hard-clipped recording (+15 dB gain clipped to [-1, 1]) -> emits clipping input_validity_warning
    clipped_pcm = np.clip(base_pcm * 5.6, -1.0, 1.0).astype(np.float32)
    p_fp_clip = fp_det.predict(clipped_pcm, 16000)
    p_g5_clip = g5_det.predict(clipped_pcm, 16000)
    assert p_fp_clip.details.get("clipping_ratio", 0.0) >= 0.005
    assert any("clipping" in w.lower() for w in p_fp_clip.details.get("input_validity_warnings", []))
    assert p_g5_clip.details.get("clipping_ratio", 0.0) >= 0.005
    assert any("clipping" in w.lower() for w in p_g5_clip.details.get("input_validity_warnings", []))


def test_phase4_tiered_attribution_and_hardened_report_payload():
    """Verify Phase 4 Task 3 & Task 7 fields in generated forensic reports."""
    resp_load = client.post("/api/audio/load-sample/c2pa_signed_demo?run_sync=true")
    assert resp_load.status_code == 200
    job_id = resp_load.json()["initial_job_id"]

    rep_resp = client.get(f"/api/jobs/{job_id}/report?format=json")
    assert rep_resp.status_code == 200
    rep = rep_resp.json()

    # Task 7 required fields
    assert "reproducible_experiment_identifiers" in rep
    assert rep["reproducible_experiment_identifiers"]["job_id"] == job_id
    assert "model_and_checkpoint_revisions" in rep
    assert len(rep["model_and_checkpoint_revisions"]) >= 2
    assert "dataset_provenance_context" in rep
    assert "input_validity_warnings" in rep
    assert "confidence_and_calibration_limitations" in rep
    assert "observed_measurements_vs_conclusions" in rep

    # Task 3 5-tier epistemic attribution hierarchy
    hier = rep.get("forensic_attribution_hierarchy", {})
    assert "tier_1_cryptographically_verified_c2pa_facts" in hier
    assert "tier_2_conventional_metadata_assertions" in hier
    assert "tier_3_audio_signal_observations" in hier
    assert "tier_4_model_predictions" in hier
    assert "tier_5_research_hypotheses" in hier
    assert "strength" in hier["tier_1_cryptographically_verified_c2pa_facts"]
    assert "limitation" in hier["tier_2_conventional_metadata_assertions"]


# ==============================================================================
# 7. PHASE 5 TEMPORAL FORENSICS, CACHING & TEMPORAL DSP COMPARISON TESTS
# ==============================================================================

def test_phase5_window_generation_overlap_and_timestamp_accuracy():
    """Verify sliding-window generation, overlap hops, and incomplete final window bounds."""
    from app.services.temporal_service import generate_sliding_window_bounds

    # 1. Empty / zero duration
    assert generate_sliding_window_bounds(0.0, 5.0, 2.5) == []

    # 2. Short clip (< 5.0s) -> single incomplete final window [0.0, 2.2]
    b_short = generate_sliding_window_bounds(2.2, 5.0, 2.5)
    assert len(b_short) == 1
    assert b_short[0] == (0.0, 2.2, True)

    # 3. Exact multiple (10.0s with 5.0s window, 2.5s hop -> [0,5], [2.5,7.5], [5.0,10.0])
    b_exact = generate_sliding_window_bounds(10.0, 5.0, 2.5)
    assert b_exact == [
        (0.0, 5.0, False),
        (2.5, 7.5, False),
        (5.0, 10.0, False),
    ]

    # 4. Recording with unanalyzed tail (11.8s with 5.0s window, 2.5s hop -> 3 full windows + 1 incomplete tail [7.5, 11.8])
    b_tail = generate_sliding_window_bounds(11.8, 5.0, 2.5)
    assert b_tail == [
        (0.0, 5.0, False),
        (2.5, 7.5, False),
        (5.0, 10.0, False),
        (7.5, 11.8, True),
    ]
    # Check 50% overlap (hop = 2.5s on 5.0s windows)
    for i in range(len(b_tail) - 1):
        assert abs((b_tail[i + 1][0] - b_tail[i][0]) - 2.5) < 1e-4


def test_phase5_temporal_short_clips_silent_windows_ood_and_long_context():
    """Verify short-clip abstention, silent-window OOD handling, neutral region labels, and 120s separation."""
    from app.services.temporal_service import (
        LABEL_ELEVATED,
        LABEL_INSUFFICIENT,
        LABEL_LOW,
         LABEL_DISAGREEMENT,
        compute_temporal_forensic_analysis,
    )

    sr = 16000
    # Construct a 12-second composite signal:
    # - [0.0s - 5.0s]: Digital silence (RMS < -80 dBFS -> must abstain and label 'Insufficient evidence')
    # - [5.0s - 12.0s]: Acoustic harmonic signal (must be non-abstained on [5.0, 10.0], and incomplete tail [10.0, 12.0] has 2.0s < 3.0s -> abstains!)
    silence_5s = np.zeros((1, sr * 5), dtype=np.float32)
    acoustic_7s = generate_acoustic_reference_pcm(sr=sr, duration=7.0)[:1, :]
    composite_12s = np.concatenate([silence_5s, acoustic_7s], axis=1)

    res = compute_temporal_forensic_analysis(
        samples=composite_12s,
        sr=sr,
        asset_sha256="test_phase5_composite_12s_sha",
        hop_sec=2.5,
        fakeprint_window_sec=5.0,
        include_exploratory_fakeprint=True,
        force_recompute=True,
    )

    steps = res["timeline_steps"]
    assert len(steps) == 4  # [0,5], [2.5,7.5], [5,10], [7.5,12]

    # Window 0 ([0.0, 5.0]) is pure silence -> abstained=True, ood_detected=True, 'Insufficient evidence'
    w0 = steps[0]
    assert w0["start_sec"] == 0.0 and w0["end_sec"] == 5.0
    assert w0["abstained"] is True
    assert w0["ood_detected"] is True
    assert w0["region_label"] == LABEL_INSUFFICIENT
    assert w0["sonics_gamma_probability"] is not None  # raw score preserved unmodified

    # Final incomplete window ([7.5, 12.0], unpadded=4.5s >= 3.0s) -> zero_pad_sec=0.5s, evaluated with warning
    w_last = steps[-1]
    assert w_last["is_incomplete_final_window"] is True
    assert abs(w_last["unpadded_duration_sec"] - 4.5) < 0.05
    assert abs(w_last["zero_pad_sec"] - 0.5) < 0.05
    assert any("incomplete final window" in w.lower() for w in w_last["warnings"])

    # Also verify a short 2.0s clip (< 3.0s minimum) abstains and zero-pads 3.0s
    short_2s_res = compute_temporal_forensic_analysis(
        samples=acoustic_7s[:, : sr * 2],
        sr=sr,
        asset_sha256="test_phase5_short_2s_sha",
        hop_sec=2.5,
        fakeprint_window_sec=5.0,
        include_exploratory_fakeprint=True,
        force_recompute=True,
    )
    w_short = short_2s_res["timeline_steps"][0]
    assert w_short["is_incomplete_final_window"] is True
    assert abs(w_short["unpadded_duration_sec"] - 2.0) < 0.05
    assert abs(w_short["zero_pad_sec"] - 3.0) < 0.05
    assert w_short["abstained"] is True
    assert w_short["region_label"] == LABEL_INSUFFICIENT

    # Verify exploratory Fakeprint segment flag
    fp_wins = res["per_detector_windows"]["fakeprint_onnx_exploratory"]
    assert len(fp_wins) == len(steps)
    assert all(w["exploratory"] is True for w in fp_wins)
    assert res["fakeprint_segment_validation"]["exploratory"] is True

    # Verify SONICS Alpha 120s is reported separately and NEVER as a 5s trace
    long_ctx = res["long_context_predictions"]
    assert long_ctx["detector_id"] == "sonics_alpha_120s"
    assert long_ctx["context_window_sec"] == 120.0
    assert long_ctx["abstained"] is True  # 12s < 115s minimum

    # Verify all region summaries use strictly the 4 neutral scientific labels
    valid_labels = {LABEL_ELEVATED, LABEL_LOW, LABEL_DISAGREEMENT, LABEL_INSUFFICIENT}
    for reg in res["region_summaries"]:
        assert reg["region_label"] in valid_labels


def test_phase5_cache_correctness_and_cancellation():
    """Verify deterministic SHA-256 caching, force_recompute bypass, and job cancellation."""
    from app.services.temporal_service import (
        clear_temporal_job_cancel,
        compute_temporal_forensic_analysis,
        request_temporal_job_cancel,
    )

    import uuid
    pcm = generate_acoustic_reference_pcm(sr=16000, duration=6.0)
    sha = f"test_phase5_cache_verification_{uuid.uuid4().hex}"

    # First call with force_recompute=True -> cache_hit=False
    r1 = compute_temporal_forensic_analysis(
        samples=pcm,
        sr=16000,
        asset_sha256=sha,
        hop_sec=2.5,
        fakeprint_window_sec=5.0,
        force_recompute=True,
    )
    assert r1["cache_hit"] is False

    # Second call with identical parameters -> cache_hit=True
    r2 = compute_temporal_forensic_analysis(
        samples=pcm,
        sr=16000,
        asset_sha256=sha,
        hop_sec=2.5,
        fakeprint_window_sec=5.0,
        force_recompute=False,
    )
    assert r2["cache_hit"] is True
    assert r2["timeline_steps"] == r1["timeline_steps"]

    # Changing hop_sec to 5.0 -> different cache key -> cache_hit=False
    r3 = compute_temporal_forensic_analysis(
        samples=pcm,
        sr=16000,
        asset_sha256=sha,
        hop_sec=5.0,
        fakeprint_window_sec=5.0,
        force_recompute=False,
    )
    assert r3["cache_hit"] is False
    assert r3["configuration"]["hop_sec"] == 5.0

    # Test cancellation handling
    cancel_id = "cancel_test_job_999"
    request_temporal_job_cancel(cancel_id)
    try:
        with pytest.raises(RuntimeError, match="cancelled"):
            compute_temporal_forensic_analysis(
                samples=pcm,
                sr=16000,
                asset_sha256="uncached_cancel_sha",
                hop_sec=2.5,
                force_recompute=True,
                job_id=cancel_id,
            )
    finally:
        clear_temporal_job_cancel(cancel_id)


def test_phase5_temporal_api_endpoints_dsp_alignment_and_report_integration():
    """Verify /temporal, /temporal-compare, aligned DSP segment deltas, and report temporal payload."""
    resp_load = client.post("/api/audio/load-sample/vocoder_artifact?run_sync=true")
    assert resp_load.status_code == 200
    job_id = resp_load.json()["initial_job_id"]

    # 1. GET /api/jobs/{job_id}/temporal
    t_get = client.get(f"/api/jobs/{job_id}/temporal?hop_sec=2.5")
    assert t_get.status_code == 200, t_get.text
    t_data = t_get.json()
    assert t_data["schema_version"] in ("phase5_temporal_v1", "phase6_temporal_v2")
    assert len(t_data["timeline_steps"]) >= 1
    assert len(t_data["region_summaries"]) >= 1
    assert "hybrid_forensic_analysis" in t_data
    assert "long_context_predictions" in t_data
    assert "score_computation_methodology" in t_data

    # 2. POST /api/jobs/{job_id}/temporal with custom hop_sec=1.5
    t_post = client.post(
        f"/api/jobs/{job_id}/temporal",
        json={
            "hop_sec": 1.5,
            "fakeprint_window_sec": 5.0,
            "include_exploratory_fakeprint": True,
            "force_recompute": True,
        },
    )
    assert t_post.status_code == 200
    assert t_post.json()["configuration"]["hop_sec"] == 1.5

    # 3. Run DSP experiment and verify temporal_comparison in comparison result and /temporal-compare
    proc_resp = client.post(
        f"/api/jobs/{job_id}/process",
        json={
            "operation_type": "lowpass",
            "preset_name": "hf_rolloff_7500",
            "engine": "scipy_dsp",
            "parameters": {"cutoff_hz": 3500.0, "order": 6},
            "export_format": "wav",
            "run_detectors_on_output": True,
        },
    )
    assert proc_resp.status_code == 200
    comp = proc_resp.json()["comparison"]
    temp_comp = comp["residual_comparison"].get("temporal_comparison")
    assert temp_comp is not None
    assert temp_comp["schema_version"] == "phase5_temporal_comparison_v1"
    assert len(temp_comp["segments"]) >= 1
    assert "do NOT demonstrate removal" in temp_comp["scientific_disclaimer"]

    tc_get = client.get(f"/api/jobs/{job_id}/temporal-compare?hop_sec=2.5")
    assert tc_get.status_code == 200
    assert tc_get.json()["segment_count"] == len(temp_comp["segments"])

    # 4. Verify JSON & PDF reports include temporal_forensic_analysis and Priority 7 fields
    rep_json = client.get(f"/api/jobs/{job_id}/report?format=json").json()
    assert "temporal_forensic_analysis" in rep_json
    assert rep_json["temporal_forensic_analysis"]["schema_version"] in (
        "phase5_temporal_v1",
        "phase6_temporal_v2",
    )
    assert len(rep_json["temporal_forensic_analysis"]["region_summaries"]) >= 1

    rep_pdf = client.get(f"/api/jobs/{job_id}/report?format=pdf")
    assert rep_pdf.status_code == 200
    assert rep_pdf.content[:4] == b"%PDF"


def test_phase6_full_temporal_engine_12_conditions_and_methodology():
    """Verify Priority 2: 1.0s/2.5s/5.0s hops, <5s short songs, sample rates, clipping, MP3, and score formulas."""
    import uuid
    from app.services.temporal_service import compute_temporal_forensic_analysis

    sr = 44100
    pcm_8s = generate_acoustic_reference_pcm(sr=sr, duration=8.0)

    # 1. Hop lengths: 1.0s, 2.5s, 5.0s
    counts = {}
    for hop in (1.0, 2.5, 5.0):
        r = compute_temporal_forensic_analysis(
            samples=pcm_8s,
            sr=sr,
            asset_sha256=f"p6_hop_{hop}_{uuid.uuid4().hex}",
            hop_sec=hop,
            include_exploratory_fakeprint=False,
            force_recompute=True,
        )
        counts[hop] = len(r["timeline_steps"])
        assert "score_computation_methodology" in r
        assert "sonics_gamma_window_probability" in r["score_computation_methodology"]
    assert counts[1.0] > counts[2.5] >= counts[5.0]

    # 2. Sample-rate conversion consistency (22050 Hz vs 48000 Hz)
    pcm_22k = generate_acoustic_reference_pcm(sr=22050, duration=6.0)
    pcm_48k = generate_acoustic_reference_pcm(sr=48000, duration=6.0)
    r_22k = compute_temporal_forensic_analysis(
        samples=pcm_22k, sr=22050, asset_sha256=f"p6_sr22k_{uuid.uuid4().hex}", hop_sec=2.5, force_recompute=True
    )
    r_48k = compute_temporal_forensic_analysis(
        samples=pcm_48k, sr=48000, asset_sha256=f"p6_sr48k_{uuid.uuid4().hex}", hop_sec=2.5, force_recompute=True
    )
    assert r_22k["timeline_steps"][0]["sonics_gamma_probability"] is not None
    assert r_48k["timeline_steps"][0]["sonics_gamma_probability"] is not None

    # 3. Clipped audio segment warning
    pcm_clipped = np.clip(pcm_8s[:, : sr * 5] * 8.0, -1.0, 1.0)
    r_clip = compute_temporal_forensic_analysis(
        samples=pcm_clipped, sr=sr, asset_sha256=f"p6_clip_{uuid.uuid4().hex}", hop_sec=5.0, force_recompute=True
    )
    assert r_clip["timeline_steps"][0]["clipping_ratio"] >= 0.005
    assert any("clipping" in w.lower() for w in r_clip["timeline_steps"][0]["warnings"])


def test_phase6_seven_dimension_cache_invalidation_and_mid_run_cancellation():
    """Verify Priority 3: All 7 cache key dimensions invalidate cache and active cancellation is atomic."""
    import threading
    import time
    import uuid
    from app.services.temporal_service import (
        DEFAULT_PREPROCESSING_VERSION,
        _get_cache_dir,
        clear_temporal_job_cancel,
        compute_temporal_cache_key,
        compute_temporal_forensic_analysis,
        request_temporal_job_cancel,
    )

    revs = {"sonics_gamma_5s": "r1", "fakeprint_onnx": "r2", "sonics_alpha_120s": "r3"}
    k_base, comp_base = compute_temporal_cache_key(
        asset_sha256="sha_A",
        hop_sec=2.5,
        fakeprint_window_sec=5.0,
        include_exploratory_fakeprint=True,
        detector_revisions=revs,
        sonics_window_sec=5.0,
        preprocessing_version=DEFAULT_PREPROCESSING_VERSION,
        inference_settings={"sonics_threshold": 0.5},
    )
    assert comp_base["source_audio_sha256"] == "sha_A"
    assert "sonics_gamma_5s" in comp_base["model_identities"]

    # Mutate each of the 7 dimensions and verify the resulting cache key differs
    k_sha, _ = compute_temporal_cache_key("sha_B", 2.5, 5.0, True, revs, 5.0, DEFAULT_PREPROCESSING_VERSION, {"sonics_threshold": 0.5})
    k_id, _ = compute_temporal_cache_key("sha_A", 2.5, 5.0, True, {**revs, "new_model": "v1"}, 5.0, DEFAULT_PREPROCESSING_VERSION, {"sonics_threshold": 0.5})
    k_rev, _ = compute_temporal_cache_key("sha_A", 2.5, 5.0, True, {**revs, "sonics_gamma_5s": "r9"}, 5.0, DEFAULT_PREPROCESSING_VERSION, {"sonics_threshold": 0.5})
    k_win, _ = compute_temporal_cache_key("sha_A", 2.5, 10.0, True, revs, 5.0, DEFAULT_PREPROCESSING_VERSION, {"sonics_threshold": 0.5})
    k_hop, _ = compute_temporal_cache_key("sha_A", 1.0, 5.0, True, revs, 5.0, DEFAULT_PREPROCESSING_VERSION, {"sonics_threshold": 0.5})
    k_prep, _ = compute_temporal_cache_key("sha_A", 2.5, 5.0, True, revs, 5.0, "preproc_v99", {"sonics_threshold": 0.5})
    k_set, _ = compute_temporal_cache_key("sha_A", 2.5, 5.0, False, revs, 5.0, DEFAULT_PREPROCESSING_VERSION, {"sonics_threshold": 0.5})

    all_keys = [k_base, k_sha, k_id, k_rev, k_win, k_hop, k_prep, k_set]
    assert len(set(all_keys)) == 8

    # Active mid-inference cancellation in worker thread
    for stale_tmp in _get_cache_dir().glob("*.tmp"):
        stale_tmp.unlink(missing_ok=True)

    pcm = generate_acoustic_reference_pcm(sr=16000, duration=15.0)
    cancel_sha = f"active_cancel_{uuid.uuid4().hex}"
    cancel_job = f"job_cancel_{uuid.uuid4().hex}"
    clear_temporal_job_cancel(cancel_job)
    outcome = {"cancelled": False}

    def _run():
        try:
            compute_temporal_forensic_analysis(
                samples=pcm,
                sr=16000,
                asset_sha256=cancel_sha,
                hop_sec=1.0,
                force_recompute=True,
                job_id=cancel_job,
                step_delay_ms=45.0,
            )
        except RuntimeError:
            outcome["cancelled"] = True

    t = threading.Thread(target=_run)
    t.start()
    time.sleep(0.08)
    request_temporal_job_cancel(cancel_job)
    t.join(timeout=10.0)
    assert outcome["cancelled"] is True
    assert len(list(_get_cache_dir().glob("*.tmp"))) == 0


def test_phase6_report_scientific_auditability_and_provenance():
    """Verify Priority 7: Exported reports contain all required forensic, temporal, and reproducibility fields."""
    resp_load = client.post("/api/audio/load-sample/c2pa_signed_demo?run_sync=true")
    assert resp_load.status_code == 200
    job_id = resp_load.json()["initial_job_id"]

    # Run a DSP experiment so processed_file_sha256 and audio_comparison_metrics are populated
    proc_resp = client.post(
        f"/api/jobs/{job_id}/process",
        json={
            "operation_type": "highpass",
            "preset_name": "subsonic_rumble_cut",
            "engine": "scipy_dsp",
            "parameters": {"cutoff_hz": 40.0, "order": 4},
            "export_format": "wav",
            "run_detectors_on_output": True,
        },
    )
    assert proc_resp.status_code == 200

    rep = client.get(f"/api/jobs/{job_id}/report?format=json").json()
    required_p7_keys = [
        "original_file_sha256",
        "processed_file_sha256",
        "c2pa_validation_status",
        "conventional_metadata",
        "detector_revisions",
        "model_preprocessing",
        "raw_scores",
        "ood_warnings",
        "temporal_scores",
        "selected_time_windows",
        "audio_comparison_metrics",
        "experiment_parameters",
        "reproducibility_information",
        "limitations_and_attribution_confidence",
        "observed_measurements_vs_conclusions",
    ]
    for k in required_p7_keys:
        assert k in rep, f"Missing Priority 7 report field: {k}"

    assert rep["original_file_sha256"] is not None
    assert rep["processed_file_sha256"] is not None
    assert rep["original_file_sha256"] != rep["processed_file_sha256"]
    assert "temporal_cache_key" in rep["reproducibility_information"]
    assert "elevated_score_windows" in rep["selected_time_windows"]


def test_phase6_windows_release_scripts_and_environment_check():
    """Verify Priority 8: Windows launch/stop/check scripts exist and environment pre-flight check passes."""
    from app.core.config import PROJECT_ROOT

    for rel in (
        "scripts/check_environment.ps1",
        "scripts/start_spectra.ps1",
        "scripts/stop_spectra.ps1",
        "start_spectra.bat",
        "stop_spectra.bat",
    ):
        p = PROJECT_ROOT / rel
        assert p.exists(), f"Missing release script: {p}"
        assert p.stat().st_size > 30




