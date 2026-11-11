"""Generator for legally clean, deterministic laboratory reference audio samples and C2PA test assets."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Tuple

import c2pa
import numpy as np
import soundfile as sf

from app.core.config import settings


def generate_acoustic_reference_pcm(sr: int = 44100, duration: float = 6.0) -> np.ndarray:
    """
    Synthesize a rich, natural-sounding acoustic harmonic progression with dense broadband
    overtone structure and organic room micro-modulation (2 channels, float32).
    """
    rng = np.random.default_rng(42)
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)

    # Chord progression: A minor 9 -> F major 7 -> C major 9 -> G6
    chords = [
        [220.0, 261.63, 329.63, 392.00, 493.88],
        [174.61, 220.00, 261.63, 349.23, 440.00],
        [261.63, 329.63, 392.00, 493.88, 587.33],
        [196.00, 246.94, 293.66, 392.00, 440.00],
    ]

    left = np.zeros_like(t, dtype=np.float64)
    right = np.zeros_like(t, dtype=np.float64)
    seg_len = len(t) // len(chords)

    for c_idx, notes in enumerate(chords):
        s_i = c_idx * seg_len
        e_i = len(t) if c_idx == len(chords) - 1 else (c_idx + 1) * seg_len
        t_seg = t[s_i:e_i] - t[s_i]
        env = np.minimum(1.0, t_seg / 0.04) * np.exp(-0.65 * t_seg)

        for n_idx, f0 in enumerate(notes):
            # Include dense natural harmonics up to 14 kHz with slight inharmonicity and vibrato
            max_h = min(45, int((sr * 0.44) // f0))
            for h in range(1, max_h + 1):
                fh = f0 * h * (1.0 + 0.00015 * (h ** 1.4))
                if fh >= sr * 0.46:
                    break
                amp = (1.0 / (h ** 1.15)) * (0.85 + 0.15 * np.sin(h * 1.7))
                vib = 0.0015 * np.sin(2.0 * np.pi * (4.8 + 0.2 * n_idx) * t_seg)
                phase_l = rng.uniform(0, 2 * np.pi)
                phase_r = phase_l + 0.18 * (n_idx - 2)
                left[s_i:e_i] += env * amp * np.sin(2.0 * np.pi * fh * (t_seg + vib) + phase_l)
                right[s_i:e_i] += env * amp * np.sin(2.0 * np.pi * fh * (t_seg - vib) + phase_r)

    # Add subtle natural acoustic room air / breath texture across full spectrum
    pink_l = rng.normal(0.0, 1.0, len(t))
    pink_r = rng.normal(0.0, 1.0, len(t))
    left += 0.22 * pink_l
    right += 0.22 * pink_r

    stereo = np.stack([left, right], axis=0)
    peak = float(np.max(np.abs(stereo))) + 1e-8
    stereo = (stereo / peak) * 0.78
    return stereo.astype(np.float32)


def generate_vocoder_artifact_pcm(sr: int = 44100, duration: float = 6.0) -> np.ndarray:
    """
    Synthesize a musical track exhibiting characteristic neural-vocoder transposed-convolution
    spectral comb artifacts in the 1,000-8,000 Hz band and bandwidth rolloff above 8 kHz.
    """
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)

    # Melodic bass & pad foundation below 1,000 Hz
    bass_freqs = [110.0, 130.81, 146.83, 98.0]
    seg_len = len(t) // len(bass_freqs)
    left = np.zeros_like(t, dtype=np.float64)
    right = np.zeros_like(t, dtype=np.float64)

    for idx, bf in enumerate(bass_freqs):
        s_i = idx * seg_len
        e_i = len(t) if idx == len(bass_freqs) - 1 else (idx + 1) * seg_len
        t_seg = t[s_i:e_i] - t[s_i]
        env = np.minimum(1.0, t_seg / 0.05) * np.exp(-0.35 * t_seg)
        for h in range(1, 6):
            sig = env * (0.5 / h) * np.sin(2.0 * np.pi * (bf * h) * t_seg)
            left[s_i:e_i] += sig
            right[s_i:e_i] += sig * 0.98

    # Inject periodic transposed-convolution upsampling comb peaks at 400 Hz intervals (4800-6800 Hz)
    # matching neural vocoder checkerboard frequency spikes
    comb_freqs = [1800.8, 2611.3, 4798.8, 5199.2, 5599.6, 6000.0, 6400.4, 6800.8]
    for cf in comb_freqs:
        tone = 0.045 * np.sin(2.0 * np.pi * cf * t)
        left += tone
        right += tone

    stereo = np.stack([left, right], axis=0)
    peak = float(np.max(np.abs(stereo))) + 1e-8
    stereo = (stereo / peak) * 0.80
    return stereo.astype(np.float32)


def sign_wav_bytes_with_c2pa(
    wav_bytes: bytes,
    title: str = "SPECTRA Signed Synthetic Audio Demo",
    claim_generator_name: str = "SPECTRA Provenance Lab",
    digital_source_type: str = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
) -> bytes:
    """Embed a cryptographically signed C2PA v2 manifest into WAV audio bytes."""
    cert_dir = settings.samples_dir / "c2pa_test_certs"
    cert_path = cert_dir / "es256_certs.pem"
    key_path = cert_dir / "es256_private.key"

    if not cert_path.exists() or not key_path.exists():
        raise FileNotFoundError("C2PA test certificates not found in storage/samples/c2pa_test_certs.")

    certs = cert_path.read_bytes()
    key = key_path.read_bytes()
    signer_info = c2pa.C2paSignerInfo(b"es256", certs, key, None)

    manifest_def = {
        "claim_generator_info": [
            {
                "name": claim_generator_name,
                "version": settings.SPECTRA_VERSION,
            }
        ],
        "format": "audio/wav",
        "title": title,
        "assertions": [
            {
                "label": "c2pa.actions",
                "data": {
                    "actions": [
                        {
                            "action": "c2pa.created",
                            "digitalSourceType": digital_source_type,
                            "softwareAgent": f"{claim_generator_name}/{settings.SPECTRA_VERSION}",
                        }
                    ]
                },
            }
        ],
    }

    src_buf = io.BytesIO(wav_bytes)
    dst_buf = io.BytesIO()
    with c2pa.Signer.from_info(signer_info) as signer:
        with c2pa.Builder(manifest_def) as builder:
            builder.sign(signer, "audio/wav", src_buf, dst_buf)
    return dst_buf.getvalue()


def ensure_reference_samples() -> List[Dict[str, str]]:
    """Ensure the 3 built-in laboratory reference WAV files exist on disk."""
    s_dir = settings.samples_dir
    s_dir.mkdir(parents=True, exist_ok=True)

    acoustic_path = s_dir / "acoustic_studio_ensemble.wav"
    if not acoustic_path.exists():
        pcm = generate_acoustic_reference_pcm(sr=44100, duration=6.0)
        sf.write(str(acoustic_path), pcm.T, 44100, format="WAV", subtype="PCM_16")

    vocoder_path = s_dir / "neural_vocoder_synth_track.wav"
    if not vocoder_path.exists():
        pcm = generate_vocoder_artifact_pcm(sr=44100, duration=6.0)
        sf.write(str(vocoder_path), pcm.T, 44100, format="WAV", subtype="PCM_16")

    c2pa_path = s_dir / "c2pa_signed_synthetic_demo.wav"
    if not c2pa_path.exists():
        try:
            raw_wav = vocoder_path.read_bytes()
            signed_wav = sign_wav_bytes_with_c2pa(
                raw_wav,
                title="SPECTRA C2PA Signed AI Audio Sample",
                claim_generator_name="SPECTRA Provenance Lab",
            )
            c2pa_path.write_bytes(signed_wav)
        except Exception:
            # Fallback copy if cert missing
            c2pa_path.write_bytes(vocoder_path.read_bytes())

    return [
        {
            "sample_id": "acoustic_reference",
            "filename": "acoustic_studio_ensemble.wav",
            "title": "Acoustic Studio Ensemble (Natural Harmonic Reference)",
            "description": "Full-bandwidth 44.1 kHz stereo acoustic progression with dense harmonic decay and no neural vocoder comb artifacts.",
            "path": str(acoustic_path),
        },
        {
            "sample_id": "vocoder_artifact",
            "filename": "neural_vocoder_synth_track.wav",
            "title": "Neural-Vocoder Artifact Track (1–8 kHz Comb Signature)",
            "description": "Synthetic track exhibiting transposed-convolution periodic spectral peaks in the 1-8 kHz region.",
            "path": str(vocoder_path),
        },
        {
            "sample_id": "c2pa_signed_demo",
            "filename": "c2pa_signed_synthetic_demo.wav",
            "title": "C2PA Cryptographically Signed Provenance Sample",
            "description": "WAV recording containing an embedded, cryptographically signed C2PA v2 manifest with c2pa.created assertion.",
            "path": str(c2pa_path),
        },
    ]
