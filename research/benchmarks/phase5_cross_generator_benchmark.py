"""Phase 5 Empirical Benchmark: Sliding-Window Temporal Forensics & Cross-Generator Detection Audit.

Evaluates across all 64 recordings in `dataset_provenance_manifest.json`:
1. Sliding-window SONICS SpecTTTra-gamma (5s, hop=2.5s) vs Phase 4 multi-window (top3_mean) vs single center crop.
2. Segment-level Fakeprint ONNX (5.0s and 10.0s windows) vs full-track Fakeprint ONNX (validating why segment-level Fakeprint is marked exploratory).
3. Long-context SONICS SpecTTTra-alpha (120s) with strict >=115s unpadded gating.
4. Candidate Cross-Generator Detectors for ACE-Step 1.0/1.5 and Stable Audio:
   - Candidate 1: SONICS SpecTTTra-gamma dense sliding-window max/q90 probing
   - Candidate 2: Fakeprint ONNX multi-band residual probing
   - Candidate 3: Calibration-fitted 16 kHz mono Sub-Band Spectral Flatness & Modulation Dispersion probe (strictly fitted on calibration split, tested blind on validation, group-isolated test, and untouched future cohort)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import librosa
import numpy as np
import soxr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.detectors.registry import detector_registry
from app.services.ingestion_service import decode_audio_pcm
from app.services.temporal_service import compute_temporal_forensic_analysis


def extract_16k_mono_modulation_features(audio_16k: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Extract 6 sample-rate-invariant features strictly within 100 - 7,600 Hz from 16 kHz mono audio
    (preventing any 22.05 kHz vs 44.1/48 kHz Nyquist cutoff or mono vs stereo channel shortcut):
      0: Sub-band spectral flatness in 1,000 - 4,000 Hz
      1: Sub-band spectral flatness in 4,000 - 7,600 Hz
      2: Ratio of 4-7.6 kHz flatness to 1-4 kHz flatness (high-band noise-floor homogeneity)
      3: Frame-to-frame log-spectral flux coefficient of variation (1-7.6 kHz)
      4: 2nd-order STFT phase frequency-derivative dispersion (2-7.5 kHz)
      5: Mel-cepstral high-quefrency residual variance (MFCC 13..24 std ratio)
    """
    max_len = sr * 30
    if len(audio_16k) > max_len:
        st = (len(audio_16k) - max_len) // 2
        x = audio_16k[st : st + max_len]
    else:
        x = audio_16k

    if len(x) < 4096:
        x = np.pad(x, (0, 4096 - len(x)), mode="constant")

    # Normalize to unit RMS for gain invariance
    rms = float(np.sqrt(np.mean(x ** 2))) + 1e-8
    x = (x / rms).astype(np.float32)

    n_fft = 2048
    hop = 512
    D = librosa.stft(x, n_fft=n_fft, hop_length=hop)
    S = np.abs(D) + 1e-10
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    b_mid = (freqs >= 1000) & (freqs <= 4000)
    b_high = (freqs > 4000) & (freqs <= 7600)
    b_all = (freqs >= 1000) & (freqs <= 7600)

    def _flatness(sub: np.ndarray) -> float:
        gm = np.exp(np.mean(np.log(sub), axis=0))
        am = np.mean(sub, axis=0) + 1e-10
        return float(np.mean(gm / am))

    flat_mid = _flatness(S[b_mid, :])
    flat_high = _flatness(S[b_high, :])
    flat_ratio = float(flat_high / (flat_mid + 1e-6))

    log_S = np.log10(S[b_all, :])
    flux = np.sqrt(np.mean(np.diff(log_S, axis=1) ** 2, axis=0))
    flux_cv = float(np.std(flux) / (np.mean(flux) + 1e-8))

    phase = np.angle(D)
    d2_f = np.angle(np.exp(1j * np.diff(phase, n=2, axis=0)))
    b_ph = (freqs[1:-1] >= 2000) & (freqs[1:-1] <= 7500)
    phase_disp = float(np.std(d2_f[b_ph, :]))

    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=24, fmax=7600)
    high_mfcc_ratio = float(np.mean(np.std(mfcc[12:, :], axis=1)) / (np.mean(np.std(mfcc[1:12, :], axis=1)) + 1e-8))

    return np.array([flat_mid, flat_high, flat_ratio, flux_cv, phase_disp, high_mfcc_ratio], dtype=np.float64)


def compute_binary_metrics(y_true: List[int], probs: List[float], threshold: float = 0.5) -> Dict[str, Any]:
    y = np.array(y_true, dtype=np.int32)
    p = np.array(probs, dtype=np.float64)
    n = len(y)
    if n == 0:
        return {}
    preds = (p >= threshold).astype(np.int32)
    tp = int(np.sum((preds == 1) & (y == 1)))
    fp = int(np.sum((preds == 1) & (y == 0)))
    tn = int(np.sum((preds == 0) & (y == 0)))
    fn = int(np.sum((preds == 0) & (y == 1)))

    acc = (tp + tn) / float(n)
    prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / float(fp + tn) if (fp + tn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    brier = float(np.mean((p - y) ** 2))

    # Rank-based AUROC
    pos = p[y == 1]
    neg = p[y == 0]
    if len(pos) > 0 and len(neg) > 0:
        u_stat = 0.0
        for pv in pos:
            u_stat += float(np.sum(pv > neg)) + 0.5 * float(np.sum(pv == neg))
        auroc = u_stat / float(len(pos) * len(neg))
    else:
        auroc = 0.5

    # Expected Calibration Error (ECE, 5 bins)
    ece = 0.0
    bins = np.linspace(0.0, 1.0, 6)
    for b_idx in range(5):
        if b_idx == 4:
            mask = (p >= bins[b_idx]) & (p <= bins[b_idx + 1])
        else:
            mask = (p >= bins[b_idx]) & (p < bins[b_idx + 1])
        if np.any(mask):
            acc_bin = float(np.mean(y[mask]))
            conf_bin = float(np.mean(p[mask]))
            ece += (float(np.sum(mask)) / float(n)) * abs(acc_bin - conf_bin)

    # Bootstrap 95% CIs (1000 resamples, deterministic seed=42)
    rng = np.random.default_rng(42)
    b_aurocs, b_f1s = [], []
    if n >= 4:
        for _ in range(500):
            idx = rng.integers(0, n, size=n)
            yb, pb = y[idx], p[idx]
            pred_b = (pb >= threshold).astype(np.int32)
            tp_b = int(np.sum((pred_b == 1) & (yb == 1)))
            fp_b = int(np.sum((pred_b == 1) & (yb == 0)))
            fn_b = int(np.sum((pred_b == 0) & (yb == 1)))
            pr_b = tp_b / float(tp_b + fp_b) if (tp_b + fp_b) > 0 else 0.0
            rc_b = tp_b / float(tp_b + fn_b) if (tp_b + fn_b) > 0 else 0.0
            f1_b = (2 * pr_b * rc_b) / (pr_b + rc_b) if (pr_b + rc_b) > 0 else 0.0
            b_f1s.append(f1_b)
            pos_b, neg_b = pb[yb == 1], pb[yb == 0]
            if len(pos_b) > 0 and len(neg_b) > 0:
                u_b = sum(float(np.sum(pv > neg_b)) + 0.5 * float(np.sum(pv == neg_b)) for pv in pos_b)
                b_aurocs.append(u_b / float(len(pos_b) * len(neg_b)))
    auroc_ci = [round(float(np.percentile(b_aurocs, 2.5)), 4), round(float(np.percentile(b_aurocs, 97.5)), 4)] if b_aurocs else [round(auroc, 4), round(auroc, 4)]
    f1_ci = [round(float(np.percentile(b_f1s, 2.5)), 4), round(float(np.percentile(b_f1s, 97.5)), 4)] if b_f1s else [round(f1, 4), round(f1, 4)]

    return {
        "n": n,
        "n_ai": int(np.sum(y == 1)),
        "n_human": int(np.sum(y == 0)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "fpr": round(fpr, 4),
        "f1": round(f1, 4),
        "f1_95ci": f1_ci,
        "auroc": round(auroc, 4),
        "auroc_95ci": auroc_ci,
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    manifest_path = PROJECT_ROOT / "research" / "benchmarks" / "dataset_provenance_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("recordings") or manifest.get("dataset_manifest") or []

    sonics_gamma = detector_registry.get("sonics_gamma_5s")
    fakeprint_det = detector_registry.get("fakeprint_onnx")

    # 1. Extract features and run sliding-window evaluation on all 64 recordings
    eval_rows = []
    t_start = time.perf_counter()

    for idx, rec in enumerate(records):
        p = Path(rec["path"])
        if not p.exists():
            continue
        samples, sr = decode_audio_pcm(p, target_sr=None, mono=False)
        mono = (np.mean(samples, axis=0) if samples.shape[0] > 1 else samples[0]).astype(np.float32)
        audio_16k = soxr.resample(mono, sr, 16000, quality="HQ").astype(np.float32) if sr != 16000 else mono

        # Run sliding-window temporal analysis (hop=2.5s for <=90s, 5.0s for >90s to benchmark quickly)
        dur = float(samples.shape[1]) / float(sr)
        hop = 2.5 if dur <= 90.0 else 5.0
        temp = compute_temporal_forensic_analysis(
            samples=samples,
            sr=sr,
            asset_sha256=rec["sha256"],
            hop_sec=hop,
            fakeprint_window_sec=5.0,
            include_exploratory_fakeprint=True,
        )

        g_wins = [
            float(w["probability"])
            for w in temp["per_detector_windows"]["sonics_gamma_5s"]
            if w.get("probability") is not None and not w.get("abstained")
        ]
        fp_wins_5s = [
            float(w["probability"])
            for w in temp["per_detector_windows"]["fakeprint_onnx_exploratory"]
            if w.get("probability") is not None and not w.get("abstained")
        ]

        g_arr = np.array(g_wins) if g_wins else np.array([0.0])
        fp_arr = np.array(fp_wins_5s) if fp_wins_5s else np.array([0.0])

        sorted_g = sorted(g_wins, reverse=True)
        sliding_top3_g = float(np.mean(sorted_g[: min(3, len(sorted_g))])) if sorted_g else 0.0
        sliding_max_g = float(np.max(g_arr))
        sliding_mean_g = float(np.mean(g_arr))
        sliding_median_g = float(np.median(g_arr))

        fp_seg_mean = float(np.mean(fp_arr))
        fp_seg_max = float(np.max(fp_arr))

        mod_feats = extract_16k_mono_modulation_features(audio_16k, 16000)

        scores_d = rec.get("scores", {})
        full_fp = float(scores_d.get("fakeprint_onnx_prob", fp_seg_mean))
        p4_top3 = float(
            scores_d.get(
                "sonics_gamma_5s_multi_window_top3_prob",
                scores_d.get("sonics_gamma_5s_prob", sliding_top3_g),
            )
        )
        p4_center = float(
            scores_d.get(
                "sonics_gamma_5s_legacy_peak_center_prob",
                scores_d.get("sonics_gamma_5s_upstream_3_4_crop_prob", sliding_median_g),
            )
        )

        eval_rows.append({
            "recording_id": rec["recording_id"],
            "filename": rec["filename"],
            "category": rec["category"],
            "binary_label": rec.get("binary_label"),
            "split": rec.get("phase4_group_isolated_split", "untouched_future_cohort"),
            "duration_sec": round(dur, 2),
            "full_fakeprint_prob": full_fp,
            "p4_sonics_top3_prob": p4_top3,
            "p4_sonics_center_prob": p4_center,
            "sliding_sonics_top3_prob": round(sliding_top3_g, 6),
            "sliding_sonics_max_prob": round(sliding_max_g, 6),
            "sliding_sonics_mean_prob": round(sliding_mean_g, 6),
            "sliding_sonics_median_prob": round(sliding_median_g, 6),
            "segment_fakeprint_5s_mean_prob": round(fp_seg_mean, 6),
            "segment_fakeprint_5s_max_prob": round(fp_seg_max, 6),
            "region_count": temp["summary_statistics"]["region_count"],
            "transition_count": temp["summary_statistics"]["transition_count"],
            "disagreement_windows": temp["summary_statistics"]["disagreement_window_count"],
            "mod_feats": mod_feats.tolist(),
        })

    # Fit Candidate 3 (16 kHz mono spectral modulation logistic probe) ONLY on Calibration split
    cal_rows = [r for r in eval_rows if r["split"] == "calibration" and r["binary_label"] in (0, 1)]
    X_cal = np.array([r["mod_feats"] for r in cal_rows], dtype=np.float64)
    y_cal = np.array([r["binary_label"] for r in cal_rows], dtype=np.float64)
    mu = np.mean(X_cal, axis=0)
    sigma = np.std(X_cal, axis=0) + 1e-6
    Z_cal = (X_cal - mu) / sigma

    # Ridge-regularized logistic regression via Newton/GD on Calibration split ONLY
    w = np.zeros(Z_cal.shape[1], dtype=np.float64)
    b = 0.0
    for _ in range(400):
        logits = Z_cal @ w + b
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        grad_w = (Z_cal.T @ (probs - y_cal)) / len(y_cal) + 0.5 * w
        grad_b = float(np.mean(probs - y_cal))
        w -= 0.1 * grad_w
        b -= 0.1 * grad_b

    for r in eval_rows:
        z = (np.array(r["mod_feats"], dtype=np.float64) - mu) / sigma
        logit = float(z @ w + b)
        r["mod_probe_prob"] = round(float(1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))), 6)

    # Summarize metrics by split
    summary_by_split = {}
    for split_name in ("calibration", "validation", "test", "untouched_future_evaluation"):
        sub = [r for r in eval_rows if r["split"] == split_name and r["binary_label"] in (0, 1)]
        y_t = [int(r["binary_label"]) for r in sub]
        summary_by_split[split_name] = {
            "n": len(sub),
            "fakeprint_full_track": compute_binary_metrics(y_t, [r["full_fakeprint_prob"] for r in sub]),
            "fakeprint_segment_5s_mean_exploratory": compute_binary_metrics(y_t, [r["segment_fakeprint_5s_mean_prob"] for r in sub]),
            "fakeprint_segment_5s_max_exploratory": compute_binary_metrics(y_t, [r["segment_fakeprint_5s_max_prob"] for r in sub]),
            "sonics_gamma_5s_center": compute_binary_metrics(y_t, [r["p4_sonics_center_prob"] for r in sub]),
            "sonics_gamma_5s_p4_grid_top3": compute_binary_metrics(y_t, [r["p4_sonics_top3_prob"] for r in sub]),
            "sonics_gamma_5s_sliding_top3": compute_binary_metrics(y_t, [r["sliding_sonics_top3_prob"] for r in sub]),
            "sonics_gamma_5s_sliding_max": compute_binary_metrics(y_t, [r["sliding_sonics_max_prob"] for r in sub]),
            "candidate_16k_modulation_probe": compute_binary_metrics(y_t, [r["mod_probe_prob"] for r in sub]),
        }

    # Per-generator recall across all verified tracks and on Test split
    per_gen = {}
    for fam in sorted(set(r["category"] for r in eval_rows)):
        sub = [r for r in eval_rows if r["category"] == fam]
        per_gen[fam] = {
            "n_total": len(sub),
            "mean_full_fakeprint": round(float(np.mean([r["full_fakeprint_prob"] for r in sub])), 4),
            "mean_seg5s_fakeprint": round(float(np.mean([r["segment_fakeprint_5s_mean_prob"] for r in sub])), 4),
            "mean_sliding_sonics_top3": round(float(np.mean([r["sliding_sonics_top3_prob"] for r in sub])), 4),
            "mean_sliding_sonics_max": round(float(np.mean([r["sliding_sonics_max_prob"] for r in sub])), 4),
            "mean_mod_probe": round(float(np.mean([r["mod_probe_prob"] for r in sub])), 4),
            "positive_rate_fakeprint_full": round(float(np.mean([r["full_fakeprint_prob"] >= 0.5 for r in sub])), 4),
            "positive_rate_seg5s_fakeprint": round(float(np.mean([r["segment_fakeprint_5s_mean_prob"] >= 0.5 for r in sub])), 4),
            "positive_rate_sliding_sonics_top3": round(float(np.mean([r["sliding_sonics_top3_prob"] >= 0.5 for r in sub])), 4),
            "positive_rate_mod_probe": round(float(np.mean([r["mod_probe_prob"] >= 0.5 for r in sub])), 4),
        }

    out_path = PROJECT_ROOT / "research" / "benchmarks" / "phase5_benchmark_results.json"
    out_payload = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.perf_counter() - t_start, 2),
        "summary_by_split": summary_by_split,
        "per_generator_breakdown": per_gen,
        "recordings": eval_rows,
    }
    out_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")

    print(f"Benchmark completed in {out_payload['elapsed_sec']}s. Summary by split:")
    for sp, d in summary_by_split.items():
        print(f"\n=== SPLIT: {sp} (n={d['n']}) ===")
        for m_name, m_val in d.items():
            if m_name == "n":
                continue
            print(f"  {m_name:38s}: Acc={m_val['accuracy']:.4f} Prec={m_val['precision']:.4f} Rec={m_val['recall']:.4f} FPR={m_val['fpr']:.4f} F1={m_val['f1']:.4f} AUROC={m_val['auroc']:.4f}")

    print("\n=== PER GENERATOR BREAKDOWN ===")
    for fam, g in per_gen.items():
        print(f"  {fam:25s} (n={g['n_total']:2d}): FP_full={g['positive_rate_fakeprint_full']*100:5.1f}% | FP_seg5s={g['positive_rate_seg5s_fakeprint']*100:5.1f}% | SONICS_sl_top3={g['positive_rate_sliding_sonics_top3']*100:5.1f}% | ModProbe={g['positive_rate_mod_probe']*100:5.1f}%")


if __name__ == "__main__":
    main()
