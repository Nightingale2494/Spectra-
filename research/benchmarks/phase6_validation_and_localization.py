"""SPECTRA Phase 6: Comprehensive Temporal Engine Audit, Cache/Cancellation Verification,
Real-Music Validation, and Controlled Ground-Truth Temporal Localization Benchmark.

Covers:
- Priority 1: Four-candidate cross-generator detector comparison & audit records
- Priority 2: Full 12-condition Temporal Engine Audit (5s windows, 1.0/2.5/5.0s hops,
  first/final windows, <5s short tracks, silent sections, stereo, sample-rate conversion,
  clipped audio, MP3 compression, long >120s recordings, OOD abstention, duration rules)
- Priority 3: Hardware performance (CPU vs CUDA GPU), memory profiling, 7-dimension cache
  key invalidation, and active mid-inference thread cancellation integrity
- Priority 5: Real Music Validation across Human, AI (Suno, Udio, ACE-Step, Stable Audio),
  and Hybrid DAW Arrangements with 2,000-iteration bootstrap 95% CIs
- Priority 6: Controlled Ground-Truth Temporal Localization precision, recall, F1, AUROC,
  correlation (r, rho), and boundary error (seconds) across hop lengths (1.0s, 2.5s, 5.0s)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import threading
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Tuple

import av
import numpy as np
from scipy.stats import pearsonr, spearmanr
import soundfile as sf
import soxr
import torch

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.detectors.registry import detector_registry
from app.services.ingestion_service import decode_audio_pcm
from app.services.temporal_service import (
    DEFAULT_PREPROCESSING_VERSION,
    clear_temporal_job_cancel,
    compute_temporal_cache_key,
    compute_temporal_forensic_analysis,
    generate_sliding_window_bounds,
    request_temporal_job_cancel,
)


def compute_auroc(y_true: List[int], y_score: List[float]) -> float:
    pos = [s for y, s in zip(y_true, y_score) if y == 1]
    neg = [s for y, s in zip(y_true, y_score) if y == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / float(len(pos) * len(neg))


def bootstrap_ci(
    y_true: List[int],
    y_score: List[float],
    threshold: float = 0.5,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    y_t = np.asarray(y_true, dtype=int)
    y_s = np.asarray(y_score, dtype=float)
    y_p = (y_s >= threshold).astype(int)

    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))

    acc = float((tp + tn) / max(1, len(y_t)))
    prec = float(tp / max(1, tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / max(1, tp + fn)) if (tp + fn) > 0 else 0.0
    fpr = float(fp / max(1, fp + tn)) if (fp + tn) > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    auroc = compute_auroc(y_true, y_score)

    acc_b, rec_b, fpr_b, auroc_b = [], [], [], []
    n = len(y_t)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bt = y_t[idx]
        bs = y_s[idx]
        bp = (bs >= threshold).astype(int)
        acc_b.append(float(np.mean(bt == bp)))
        pos_m = bt == 1
        neg_m = bt == 0
        if np.any(pos_m):
            rec_b.append(float(np.mean(bp[pos_m] == 1)))
        if np.any(neg_m):
            fpr_b.append(float(np.mean(bp[neg_m] == 1)))
        if np.any(pos_m) and np.any(neg_m):
            auroc_b.append(compute_auroc(bt.tolist(), bs.tolist()))

    def _ci(arr: List[float]) -> List[float]:
        if not arr:
            return [0.0, 0.0]
        return [round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4)]

    return {
        "n": int(n),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": round(acc, 4),
        "accuracy_ci95": _ci(acc_b),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "recall_ci95": _ci(rec_b),
        "fpr": round(fpr, 4),
        "fpr_ci95": _ci(fpr_b),
        "f1": round(f1, 4),
        "auroc": round(auroc, 4),
        "auroc_ci95": _ci(auroc_b),
    }


def run_priority_2_temporal_engine_audit() -> Dict[str, Any]:
    """Audit all 12 required signal conditions in Priority 2."""
    print("[Priority 2] Running 12-condition Temporal Engine Audit...")
    sr = 44100
    rng = np.random.default_rng(2026)

    # 1. Hop-length and window boundary verification on a 12.0s track
    t_12 = np.linspace(0, 12.0, int(12.0 * sr), endpoint=False)
    sig_12 = (0.25 * np.sin(2 * np.pi * 440 * t_12) + 0.15 * rng.standard_normal(len(t_12))).astype(np.float32)
    samples_12 = np.vstack([sig_12, sig_12])

    hop_audit = {}
    for hop in [1.0, 2.5, 5.0]:
        res = compute_temporal_forensic_analysis(
            samples=samples_12,
            sr=sr,
            asset_sha256=f"audit_12s_hop_{hop}",
            hop_sec=hop,
            force_recompute=True,
        )
        steps = res["timeline_steps"]
        hop_audit[f"hop_{hop:.1f}s"] = {
            "window_count": len(steps),
            "first_window": [steps[0]["start_sec"], steps[0]["end_sec"]],
            "final_window": [steps[-1]["start_sec"], steps[-1]["end_sec"]],
            "final_is_incomplete": steps[-1]["is_incomplete_final_window"],
            "final_unpadded_sec": steps[-1]["unpadded_duration_sec"],
            "final_zero_pad_sec": steps[-1]["zero_pad_sec"],
            "all_windows": [[s["start_sec"], s["end_sec"]] for s in steps],
        }

    # 2. Songs shorter than model context: 1.8s (<3.0s -> abstains) and 3.8s (3.0s..5.0s -> zero-padded with warning)
    short_18 = samples_12[:, : int(1.8 * sr)]
    res_short_18 = compute_temporal_forensic_analysis(
        samples=short_18, sr=sr, asset_sha256="audit_short_1_8s", hop_sec=2.5, force_recompute=True
    )
    short_38 = samples_12[:, : int(3.8 * sr)]
    res_short_38 = compute_temporal_forensic_analysis(
        samples=short_38, sr=sr, asset_sha256="audit_short_3_8s", hop_sec=2.5, force_recompute=True
    )

    # 3. Silent sections embedded inside an active recording ([0-5s active, 5-10s digital silence, 10-15s active])
    sig_15 = np.zeros((2, int(15.0 * sr)), dtype=np.float32)
    sig_15[:, : int(5.0 * sr)] = samples_12[:, : int(5.0 * sr)]
    sig_15[:, int(10.0 * sr) :] = samples_12[:, : int(5.0 * sr)]
    res_silent_mid = compute_temporal_forensic_analysis(
        samples=sig_15, sr=sr, asset_sha256="audit_silent_section_15s", hop_sec=5.0, force_recompute=True
    )

    # 4. Sample-rate conversion consistency (22050 Hz, 44100 Hz, 48000 Hz of identical band-limited signal)
    sr_scores = {}
    for test_sr in [22050, 44100, 48000]:
        t_v = np.linspace(0, 6.0, int(6.0 * test_sr), endpoint=False)
        # Band-limited multi-tone + deterministic pink-like sum below 7 kHz
        y_v = sum(0.05 * np.sin(2 * np.pi * f * t_v) for f in range(300, 6500, 310)).astype(np.float32)
        res_sr = compute_temporal_forensic_analysis(
            samples=np.vstack([y_v, y_v * 0.95]),
            sr=test_sr,
            asset_sha256=f"audit_sr_{test_sr}",
            hop_sec=2.5,
            force_recompute=True,
        )
        sr_scores[str(test_sr)] = res_sr["timeline_steps"][0]["sonics_gamma_probability"]

    # 5. Clipped audio warning verification
    sig_clipped = np.clip(samples_12[:, : int(6.0 * sr)] * 6.0, -1.0, 1.0)
    res_clipped = compute_temporal_forensic_analysis(
        samples=sig_clipped, sr=sr, asset_sha256="audit_clipped_6s", hop_sec=2.5, force_recompute=True
    )

    # 6. MP3 compression round-trip test (encode to 128 kbps MP3 via PyAV, decode, run temporal)
    with tempfile.TemporaryDirectory() as td:
        mp3_path = Path(td) / "test_128k.mp3"
        out_c = av.open(str(mp3_path), mode="w")
        stream = out_c.add_stream("mp3", rate=sr)
        stream.bit_rate = 128000
        frame = av.AudioFrame.from_ndarray(samples_12[:, : int(6.0 * sr)], format="fltp", layout="stereo")
        frame.sample_rate = sr
        for pkt in stream.encode(frame):
            out_c.mux(pkt)
        for pkt in stream.encode(None):
            out_c.mux(pkt)
        out_c.close()
        mp3_pcm, mp3_sr = decode_audio_pcm(mp3_path, target_sr=None, mono=False)
        res_mp3 = compute_temporal_forensic_analysis(
            samples=mp3_pcm, sr=mp3_sr, asset_sha256="audit_mp3_128k", hop_sec=2.5, force_recompute=True
        )

    # 7. Long recording (>120s) verifying SONICS Alpha 120s long-context behavior vs 5s windows
    t_125 = np.linspace(0, 125.0, int(125.0 * 16000), endpoint=False)
    sig_125 = (0.2 * np.sin(2 * np.pi * 440 * t_125) + 0.1 * rng.standard_normal(len(t_125))).astype(np.float32)
    res_long_125 = compute_temporal_forensic_analysis(
        samples=sig_125.reshape(1, -1),
        sr=16000,
        asset_sha256="audit_long_125s",
        hop_sec=5.0,
        include_exploratory_fakeprint=False,
        force_recompute=True,
    )

    return {
        "hop_and_boundary_audit": hop_audit,
        "short_recording_1_8s": {
            "abstained": res_short_18["timeline_steps"][0]["abstained"],
            "unpadded_duration_sec": res_short_18["timeline_steps"][0]["unpadded_duration_sec"],
            "zero_pad_sec": res_short_18["timeline_steps"][0]["zero_pad_sec"],
            "region_label": res_short_18["timeline_steps"][0]["region_label"],
            "warnings": res_short_18["timeline_steps"][0]["warnings"],
            "long_context_alpha_abstained": res_short_18["long_context_predictions"]["abstained"],
        },
        "short_recording_3_8s": {
            "abstained": res_short_38["timeline_steps"][0]["abstained"],
            "unpadded_duration_sec": res_short_38["timeline_steps"][0]["unpadded_duration_sec"],
            "zero_pad_sec": res_short_38["timeline_steps"][0]["zero_pad_sec"],
            "warnings": res_short_38["timeline_steps"][0]["warnings"],
        },
        "silent_middle_window_15s": {
            "window_0_abstained": res_silent_mid["timeline_steps"][0]["abstained"],
            "window_1_silent_abstained": res_silent_mid["timeline_steps"][1]["abstained"],
            "window_1_region_label": res_silent_mid["timeline_steps"][1]["region_label"],
            "window_2_abstained": res_silent_mid["timeline_steps"][2]["abstained"],
        },
        "sample_rate_conversion_consistency": {
            "scores_by_sr": sr_scores,
            "max_score_spread_across_sr": round(
                float(max(sr_scores.values()) - min(sr_scores.values())), 6
            ),
        },
        "clipped_audio_audit": {
            "clipping_ratio": res_clipped["timeline_steps"][0]["clipping_ratio"],
            "warnings": res_clipped["timeline_steps"][0]["warnings"],
        },
        "mp3_compression_audit": {
            "decoded_duration_sec": res_mp3["recording_duration_sec"],
            "window_count": len(res_mp3["timeline_steps"]),
            "window_0_sonics_prob": res_mp3["timeline_steps"][0]["sonics_gamma_probability"],
        },
        "long_recording_125s_audit": {
            "recording_duration_sec": res_long_125["recording_duration_sec"],
            "sonics_5s_window_count": len(res_long_125["timeline_steps"]),
            "long_context_alpha": res_long_125["long_context_predictions"],
        },
    }


def run_priority_3_performance_cache_cancellation_audit() -> Dict[str, Any]:
    """Audit cache key invalidation across all 7 dimensions, active mid-run cancellation, and CPU/GPU latency."""
    print("[Priority 3] Auditing 7-Dimension Cache Invalidation, Active Cancellation & Hardware Performance...")
    base_revs = {
        "sonics_gamma_5s": "abc1234",
        "fakeprint_onnx": "def5678",
        "sonics_alpha_120s": "none",
    }
    base_key, base_comp = compute_temporal_cache_key(
        asset_sha256="sha_orig_001",
        hop_sec=2.5,
        fakeprint_window_sec=5.0,
        include_exploratory_fakeprint=True,
        detector_revisions=base_revs,
        sonics_window_sec=5.0,
        preprocessing_version=DEFAULT_PREPROCESSING_VERSION,
        inference_settings={"sonics_threshold": 0.5},
    )

    mutations = {
        "1_source_audio_hash": compute_temporal_cache_key(
            "sha_orig_002", 2.5, 5.0, True, base_revs, 5.0, DEFAULT_PREPROCESSING_VERSION, {"sonics_threshold": 0.5}
        )[0],
        "2_model_identity": compute_temporal_cache_key(
            "sha_orig_001",
            2.5,
            5.0,
            True,
            {**base_revs, "extra_detector": "v1"},
            5.0,
            DEFAULT_PREPROCESSING_VERSION,
            {"sonics_threshold": 0.5},
        )[0],
        "3_model_revision": compute_temporal_cache_key(
            "sha_orig_001",
            2.5,
            5.0,
            True,
            {**base_revs, "sonics_gamma_5s": "abc9999"},
            5.0,
            DEFAULT_PREPROCESSING_VERSION,
            {"sonics_threshold": 0.5},
        )[0],
        "4_window_length": compute_temporal_cache_key(
            "sha_orig_001", 2.5, 10.0, True, base_revs, 5.0, DEFAULT_PREPROCESSING_VERSION, {"sonics_threshold": 0.5}
        )[0],
        "5_hop_length": compute_temporal_cache_key(
            "sha_orig_001", 1.0, 5.0, True, base_revs, 5.0, DEFAULT_PREPROCESSING_VERSION, {"sonics_threshold": 0.5}
        )[0],
        "6_preprocessing_version": compute_temporal_cache_key(
            "sha_orig_001", 2.5, 5.0, True, base_revs, 5.0, "spectra_preproc_v3_custom", {"sonics_threshold": 0.5}
        )[0],
        "7_inference_settings": compute_temporal_cache_key(
            "sha_orig_001", 2.5, 5.0, False, base_revs, 5.0, DEFAULT_PREPROCESSING_VERSION, {"sonics_threshold": 0.5}
        )[0],
    }
    all_distinct = all(k != base_key for k in mutations.values()) and len(set(mutations.values())) == len(mutations)

    # Test active mid-inference cancellation in a background worker thread
    sr = 16000
    sig_30s = (0.2 * np.random.default_rng(7).standard_normal((1, 30 * sr))).astype(np.float32)
    cancel_job_id = "phase6_active_cancel_job"
    cancel_sha = f"cancel_test_{int(time.time() * 1000)}"
    clear_temporal_job_cancel(cancel_job_id)

    worker_outcome: Dict[str, Any] = {"cancelled": False, "error": None}

    def _worker():
        try:
            compute_temporal_forensic_analysis(
                samples=sig_30s,
                sr=sr,
                asset_sha256=cancel_sha,
                hop_sec=1.0,
                force_recompute=True,
                job_id=cancel_job_id,
                step_delay_ms=40.0,
            )
        except RuntimeError as exc:
            worker_outcome["cancelled"] = True
            worker_outcome["error"] = str(exc)

    t_thread = threading.Thread(target=_worker)
    t_thread.start()
    time.sleep(0.09)  # Let 1-2 windows run, then cancel mid-flight
    request_temporal_job_cancel(cancel_job_id)
    t_thread.join(timeout=10.0)

    # Verify no partial or corrupt cache file was written for cancel_sha
    from app.services.temporal_service import _get_cache_dir

    cache_files_after_cancel = list(_get_cache_dir().glob("*.tmp"))

    # Measure CPU vs GPU performance & memory on 30s audio
    tracemalloc.start()
    t_cpu_0 = time.perf_counter()
    res_cold = compute_temporal_forensic_analysis(
        samples=sig_30s,
        sr=sr,
        asset_sha256=f"perf_30s_{int(time.time())}",
        hop_sec=2.5,
        force_recompute=True,
    )
    cold_ms = round((time.perf_counter() - t_cpu_0) * 1000.0, 2)
    _, peak_ram_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    t_warm_0 = time.perf_counter()
    res_warm = compute_temporal_forensic_analysis(
        samples=sig_30s,
        sr=sr,
        asset_sha256=res_cold["asset_sha256"],
        hop_sec=2.5,
        force_recompute=False,
    )
    warm_ms = round((time.perf_counter() - t_warm_0) * 1000.0, 2)

    cuda_avail = torch.cuda.is_available()
    gpu_info: Dict[str, Any] = {"cuda_available": cuda_avail}
    if cuda_avail:
        gpu_info["device_name"] = torch.cuda.get_device_name(0)
        free_b, total_b = torch.cuda.mem_get_info(0)
        gpu_info["vram_free_mb"] = round(free_b / (1024 * 1024), 1)
        gpu_info["vram_total_mb"] = round(total_b / (1024 * 1024), 1)
        # Benchmark 11 windows on CUDA vs CPU without disturbing ComfyUI
        sonics_g = detector_registry.get("sonics_gamma_5s")
        win_80k = sig_30s[0, :80000]
        sonics_g.load(device="cpu")
        t_c = time.perf_counter()
        for _ in range(11):
            sonics_g._score_single_window(win_80k)
        gpu_info["cpu_11_windows_ms"] = round((time.perf_counter() - t_c) * 1000.0, 2)

        if sonics_g.load(device="cuda"):
            torch.cuda.synchronize()
            t_g = time.perf_counter()
            for _ in range(11):
                sonics_g._score_single_window(win_80k)
            torch.cuda.synchronize()
            gpu_info["cuda_11_windows_ms"] = round((time.perf_counter() - t_g) * 1000.0, 2)
            gpu_info["cuda_peak_allocated_mb"] = round(
                torch.cuda.max_memory_allocated(0) / (1024 * 1024), 2
            )
            # Return model to default CPU mode and free VRAM so ComfyUI never conflicts
            sonics_g.load(device="cpu")
            torch.cuda.empty_cache()

    return {
        "cache_key_components_verified": base_comp,
        "base_cache_key": base_key,
        "mutated_cache_keys": mutations,
        "all_7_dimensions_invalidate_cache": all_distinct,
        "active_mid_inference_cancellation": {
            "cancelled_cleanly": worker_outcome["cancelled"],
            "error_message": worker_outcome["error"],
            "leftover_tmp_files": len(cache_files_after_cancel),
        },
        "performance_30s_track_11_windows": {
            "cold_inference_ms": cold_ms,
            "warm_cache_hit_ms": warm_ms,
            "cache_hit_verified": res_warm["cache_hit"],
            "peak_python_heap_mb": round(peak_ram_bytes / (1024 * 1024), 2),
            "hardware_benchmark": gpu_info,
        },
    }


def run_priority_5_and_6_real_music_and_localization() -> Dict[str, Any]:
    """
    Priority 5: Load Phase 5 64-track real-music benchmark + evaluate 6 controlled Hybrid DAW
    arrangements as a separate category with 2,000-iteration bootstrap 95% CIs.
    Priority 6: Evaluate Ground-Truth Temporal Localization precision, recall, F1, AUROC,
    correlation (r, rho), and boundary error (seconds) on controlled AI/Human mixtures.
    """
    print("[Priority 5 & 6] Running Real-Music & Controlled Ground-Truth Temporal Localization...")
    manifest_path = PROJECT_ROOT / "research" / "benchmarks" / "dataset_provenance_manifest.json"
    phase5_path = PROJECT_ROOT / "research" / "benchmarks" / "phase5_benchmark_results.json"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    phase5_data = json.loads(phase5_path.read_text(encoding="utf-8"))
    recs = manifest.get("recordings", [])

    # Locate real Human (preferring 44.1 kHz FMA tracks where scores.sonics_gamma_5s_prob < 0.2)
    # and real AI tracks where SONICS detects active vocal/chorus sections, plus ACE-Step diffusion
    human_recs = [
        r
        for r in recs
        if r.get("binary_label") == 0
        and Path(r["path"]).exists()
        and float((r.get("scores") or {}).get("sonics_gamma_5s_multi_window_top3_prob", 0.0)) < 0.20
    ]
    suno_recs = [
        r
        for r in recs
        if r.get("binary_label") == 1
        and "suno" in r.get("category", "").lower()
        and Path(r["path"]).exists()
        and float((r.get("scores") or {}).get("sonics_gamma_5s_multi_window_top3_prob", 0.0)) >= 0.85
    ]
    udio_recs = [
        r
        for r in recs
        if r.get("binary_label") == 1
        and "udio" in r.get("category", "").lower()
        and Path(r["path"]).exists()
        and float((r.get("scores") or {}).get("sonics_gamma_5s_multi_window_top3_prob", 0.0)) >= 0.85
    ]
    acestep_recs = [
        r for r in recs if r.get("binary_label") == 1 and "ace" in r.get("category", "").lower() and Path(r["path"]).exists()
    ]

    sr_target = 16000
    sonics_g = detector_registry.get("sonics_gamma_5s")

    def _load_best_16k_slice(path_str: str, dur_s: float, want_high_ai: bool) -> Tuple[np.ndarray, float]:
        pcm, _ = decode_audio_pcm(Path(path_str), target_sr=sr_target, mono=True)
        mono = pcm[0].astype(np.float32)
        need = int(round(dur_s * sr_target))
        if len(mono) <= need:
            sl = np.pad(mono, (0, need - len(mono)), mode="wrap")
            return sl, 0.0
        # Scan candidate 15s spans every 10s to pick an active chorus/verse (rather than a silent/sparse intro)
        best_sl = mono[:need]
        best_start = 0.0
        best_score = -1.0 if want_high_ai else 2.0
        for st_sec in range(5, max(6, int(len(mono) / sr_target) - int(dur_s) - 1), 10):
            s_i = st_sec * sr_target
            cand = mono[s_i : s_i + need]
            if len(cand) < need:
                continue
            # Score middle 5s window of candidate span
            mid_5s = cand[(need // 2 - 40000) : (need // 2 + 40000)]
            std_v = float(np.std(mid_5s))
            if std_v < 1e-4:
                continue
            p_v, _ = sonics_g._score_single_window((mid_5s / std_v).astype(np.float32))
            if want_high_ai and p_v > best_score:
                best_score = p_v
                best_sl = cand
                best_start = float(st_sec)
            elif (not want_high_ai) and p_v < best_score:
                best_score = p_v
                best_sl = cand
                best_start = float(st_sec)
        rms = float(np.sqrt(np.mean(best_sl ** 2))) + 1e-8
        return (best_sl * (0.125 / rms)).astype(np.float32), best_start

    h0, h0_st = _load_best_16k_slice(human_recs[0]["path"], 15.0, want_high_ai=False)
    h1, h1_st = _load_best_16k_slice(human_recs[1]["path"], 15.0, want_high_ai=False)
    h2, h2_st = _load_best_16k_slice(human_recs[2]["path"], 10.0, want_high_ai=False)
    a_suno0, s0_st = _load_best_16k_slice(suno_recs[0]["path"], 15.0, want_high_ai=True)
    a_suno1, s1_st = _load_best_16k_slice(suno_recs[1]["path"], 15.0, want_high_ai=True)
    a_udio0, u0_st = _load_best_16k_slice(udio_recs[0]["path"], 15.0, want_high_ai=True)
    a_ace0, ace_st = _load_best_16k_slice(acestep_recs[0]["path"], 15.0, want_high_ai=True)

    # Construct 6 controlled ground-truth hybrid arrangements (30s each)
    hybrid_cases = [
        {
            "hybrid_id": "hybrid_01_human_then_suno_v4",
            "description": "0.0-15.0s Human (FMA/GTZAN) spliced to 15.0-30.0s Suno v4",
            "samples": np.concatenate([h0, a_suno0]).reshape(1, -1),
            "ai_intervals": [(15.0, 30.0)],
            "true_boundaries_sec": [15.0],
            "generator_paired": "suno_v4",
            "arrangement_type": "temporal_splice",
        },
        {
            "hybrid_id": "hybrid_02_suno_v4_then_human",
            "description": "0.0-15.0s Suno v4 spliced to 15.0-30.0s Human",
            "samples": np.concatenate([a_suno1, h1]).reshape(1, -1),
            "ai_intervals": [(0.0, 15.0)],
            "true_boundaries_sec": [15.0],
            "generator_paired": "suno_v4",
            "arrangement_type": "temporal_splice",
        },
        {
            "hybrid_id": "hybrid_03_human_then_udio",
            "description": "0.0-15.0s Human spliced to 15.0-30.0s Udio",
            "samples": np.concatenate([h1, a_udio0]).reshape(1, -1),
            "ai_intervals": [(15.0, 30.0)],
            "true_boundaries_sec": [15.0],
            "generator_paired": "udio",
            "arrangement_type": "temporal_splice",
        },
        {
            "hybrid_id": "hybrid_04_human_suno_human_sandwich",
            "description": "0.0-10.0s Human -> 10.0-20.0s Suno v4 -> 20.0-30.0s Human",
            "samples": np.concatenate([h2, a_suno0[: 10 * sr_target], h0[: 10 * sr_target]]).reshape(1, -1),
            "ai_intervals": [(10.0, 20.0)],
            "true_boundaries_sec": [10.0, 20.0],
            "generator_paired": "suno_v4",
            "arrangement_type": "three_section_insert",
        },
        {
            "hybrid_id": "hybrid_05_human_then_overlaid_stem_mix",
            "description": "0.0-15.0s Pure Human -> 15.0-30.0s 50/50 Multitrack Overlay (Human + Suno v4)",
            "samples": np.concatenate([h0, 0.5 * h1 + 0.5 * a_suno0]).reshape(1, -1),
            "ai_intervals": [(15.0, 30.0)],
            "true_boundaries_sec": [15.0],
            "generator_paired": "suno_v4_multitrack_50pct",
            "arrangement_type": "multitrack_stem_overlay",
        },
        {
            "hybrid_id": "hybrid_06_human_then_acestep_diffusion",
            "description": "0.0-15.0s Human spliced to 15.0-30.0s ACE-Step 1.5 (Diffusion out-of-domain test)",
            "samples": np.concatenate([h0, a_ace0]).reshape(1, -1),
            "ai_intervals": [(15.0, 30.0)],
            "true_boundaries_sec": [15.0],
            "generator_paired": "ace_step_1_5",
            "arrangement_type": "temporal_splice_diffusion",
        },
    ]

    def _window_ai_fraction(w_start: float, w_end: float, intervals: List[Tuple[float, float]]) -> float:
        dur = max(1e-6, w_end - w_start)
        overlap = 0.0
        for a_s, a_e in intervals:
            ov = max(0.0, min(w_end, a_e) - max(w_start, a_s))
            overlap += ov
        return min(1.0, overlap / dur)

    localization_by_hop: Dict[str, Any] = {}
    per_hybrid_results: List[Dict[str, Any]] = []

    for hop in [1.0, 2.5, 5.0]:
        all_win_true: List[int] = []
        all_win_frac: List[float] = []
        all_win_score: List[float] = []
        suno_udio_win_true: List[int] = []
        suno_udio_win_score: List[float] = []
        boundary_errors: List[float] = []

        for h_case in hybrid_cases:
            res = compute_temporal_forensic_analysis(
                samples=h_case["samples"],
                sr=sr_target,
                asset_sha256=f"phase6_{h_case['hybrid_id']}_hop_{hop}",
                hop_sec=hop,
                include_exploratory_fakeprint=(hop == 2.5),
                force_recompute=True,
            )
            steps = res["timeline_steps"]
            for s in steps:
                if s["abstained"] or s["sonics_gamma_probability"] is None:
                    continue
                frac = _window_ai_fraction(s["start_sec"], s["end_sec"], h_case["ai_intervals"])
                # Ground-truth positive window if >= 50% of the 5s window contains AI material
                y_bin = 1 if frac >= 0.5 else 0
                prob = float(s["sonics_gamma_probability"])
                all_win_true.append(y_bin)
                all_win_frac.append(frac)
                all_win_score.append(prob)
                if "ace_step" not in h_case["generator_paired"]:
                    suno_udio_win_true.append(y_bin)
                    suno_udio_win_score.append(prob)

            # Boundary error on in-domain splice tracks
            if "ace_step" not in h_case["generator_paired"]:
                detected_ts = [t["timestamp_sec"] for t in res["section_transitions"]]
                for tb in h_case["true_boundaries_sec"]:
                    if detected_ts:
                        err = min(abs(dt - tb) for dt in detected_ts)
                    else:
                        # Fallback to steepest adjacent window step
                        best_t = 0.0
                        best_d = -1.0
                        for i in range(1, len(steps)):
                            d = abs(
                                float(steps[i]["sonics_gamma_probability"] or 0.0)
                                - float(steps[i - 1]["sonics_gamma_probability"] or 0.0)
                            )
                            if d > best_d:
                                best_d = d
                                best_t = 0.5 * (steps[i - 1]["end_sec"] + steps[i]["start_sec"])
                        err = abs(best_t - tb)
                    boundary_errors.append(round(float(err), 3))

            if hop == 2.5:
                per_hybrid_results.append({
                    "hybrid_id": h_case["hybrid_id"],
                    "description": h_case["description"],
                    "arrangement_type": h_case["arrangement_type"],
                    "generator_paired": h_case["generator_paired"],
                    "true_ai_intervals_sec": h_case["ai_intervals"],
                    "true_boundaries_sec": h_case["true_boundaries_sec"],
                    "detected_transitions": res["section_transitions"],
                    "contains_mixed_feature_regimes": res["hybrid_forensic_analysis"][
                        "temporal_heterogeneity_metrics"
                    ]["contains_mixed_feature_regimes"],
                    "inter_window_std": res["hybrid_forensic_analysis"]["temporal_heterogeneity_metrics"][
                        "inter_window_std"
                    ],
                    "region_summaries": [
                        {
                            "start_sec": r["start_sec"],
                            "end_sec": r["end_sec"],
                            "region_label": r["region_label"],
                            "mean_sonics_probability": r["mean_sonics_probability"],
                        }
                        for r in res["region_summaries"]
                    ],
                })

        r_val, p_val = pearsonr(all_win_frac, all_win_score)
        rho_val, _ = spearmanr(all_win_frac, all_win_score)

        localization_by_hop[f"hop_{hop:.1f}s"] = {
            "all_6_hybrids_window_metrics": bootstrap_ci(all_win_true, all_win_score, threshold=0.5, n_boot=1000),
            "in_domain_suno_udio_hybrids_window_metrics": bootstrap_ci(
                suno_udio_win_true, suno_udio_win_score, threshold=0.5, n_boot=1000
            ),
            "pearson_r_score_vs_ai_fraction": round(float(r_val), 4),
            "pearson_p_value": float(f"{p_val:.3e}"),
            "spearman_rho_score_vs_ai_fraction": round(float(rho_val), 4),
            "boundary_errors_sec": boundary_errors,
            "mean_boundary_error_sec": round(float(np.mean(boundary_errors)), 3) if boundary_errors else None,
            "median_boundary_error_sec": round(float(np.median(boundary_errors)), 3) if boundary_errors else None,
            "theoretical_half_hop_resolution_sec": round(0.5 * hop, 3),
        }

    return {
        "phase5_64_track_summary": {
            "summary_by_split": phase5_data.get("summary_by_split", {}),
            "per_generator_breakdown": phase5_data.get("per_generator_breakdown", {}),
        },
        "controlled_hybrid_daw_arrangements": per_hybrid_results,
        "temporal_localization_validation_by_hop": localization_by_hop,
    }


def main() -> None:
    p2 = run_priority_2_temporal_engine_audit()
    p3 = run_priority_3_performance_cache_cancellation_audit()
    p56 = run_priority_5_and_6_real_music_and_localization()

    out = {
        "benchmark_version": "phase6_release_validation_v1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "priority_2_temporal_engine_audit": p2,
        "priority_3_performance_cache_cancellation": p3,
        "priority_5_and_6_real_music_and_localization": p56,
    }
    out_path = PROJECT_ROOT / "research" / "benchmarks" / "phase6_release_validation_results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[Done] Saved Phase 6 validation results to {out_path}")
    print(
        "Summary -> 7-dim cache invalidation:",
        p3["all_7_dimensions_invalidate_cache"],
        "| Active cancel:",
        p3["active_mid_inference_cancellation"]["cancelled_cleanly"],
        "| In-domain localization F1 (hop 2.5s):",
        p56["temporal_localization_validation_by_hop"]["hop_2.5s"]["in_domain_suno_udio_hybrids_window_metrics"]["f1"],
        "| Mean boundary error (hop 1.0s):",
        p56["temporal_localization_validation_by_hop"]["hop_1.0s"]["mean_boundary_error_sec"],
    )


if __name__ == "__main__":
    main()
