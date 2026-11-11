"""Secure audio ingestion, validation, metadata extraction, and PCM decoding."""
from __future__ import annotations

import hashlib
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import av
import mutagen
import numpy as np
import soundfile as sf
import soxr
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.models import AudioAsset

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a"}

EXTENSION_MIME_MAP = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
}

WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
}


class AudioIngestionError(Exception):
    """Raised when an uploaded audio file fails security or format validation."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def sanitize_filename(filename: str) -> str:
    """Sanitize user-supplied filename and prevent path traversal or reserved names."""
    if not filename or "\x00" in filename:
        raise AudioIngestionError("Invalid or empty filename.")

    # Reject explicit traversal sequences in raw input
    normalized_raw = filename.replace("\\", "/")
    if "../" in normalized_raw or normalized_raw.startswith("/") or ":" in Path(filename).drive:
        raise AudioIngestionError("Dangerous filename detected (path traversal rejected).")

    base = os.path.basename(normalized_raw).strip()
    if not base or base in {".", ".."}:
        raise AudioIngestionError("Invalid filename.")

    stem, ext = os.path.splitext(base)
    ext_lower = ext.lower()
    if ext_lower not in SUPPORTED_EXTENSIONS:
        raise AudioIngestionError(
            f"Unsupported file extension '{ext_lower}'. Supported formats: WAV, MP3, FLAC, M4A.",
            status_code=415,
        )

    clean_stem = re.sub(r"[^A-Za-z0-9._-]", "_", stem).strip("._-")
    if not clean_stem:
        clean_stem = "audio_sample"
    if clean_stem.upper() in WINDOWS_RESERVED_NAMES:
        clean_stem = f"file_{clean_stem}"

    return f"{clean_stem[:120]}{ext_lower}"


def verify_magic_bytes(header: bytes, expected_ext: str) -> str:
    """Inspect magic bytes to detect actual container format and reject spoofed extensions."""
    if len(header) < 12:
        raise AudioIngestionError("File is too small or truncated to be a valid audio container.")

    detected_format: Optional[str] = None

    if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
        detected_format = ".wav"
    elif header[:4] == b"fLaC":
        detected_format = ".flac"
    elif header[4:8] == b"ftyp":
        detected_format = ".m4a"
    elif header[:3] == b"ID3" or (header[0] == 0xFF and (header[1] & 0xE0) == 0xE0):
        detected_format = ".mp3"
    elif header[:4] == b"OggS":
        raise AudioIngestionError(
            "OGG container detected; only WAV, MP3, FLAC, and M4A are supported.",
            status_code=415,
        )

    if detected_format is None:
        raise AudioIngestionError(
            "Unrecognized or corrupted audio file header (magic byte check failed).",
            status_code=400,
        )

    if detected_format != expected_ext:
        raise AudioIngestionError(
            f"File extension mismatch: filename has '{expected_ext}' but binary header is '{detected_format}'.",
            status_code=400,
        )

    return detected_format[1:].upper()


def inspect_audio_technical_metadata(file_path: Path) -> Dict[str, Any]:
    """Extract technical audio parameters and verify decodability without modifying the file."""
    info: Dict[str, Any] = {
        "codec": "unknown",
        "container_format": file_path.suffix.lstrip(".").lower(),
        "duration_seconds": 0.0,
        "sample_rate": 0,
        "channels": 0,
        "bit_depth": None,
        "bitrate": None,
    }

    # Try PyAV container inspection first (handles WAV, MP3, FLAC, M4A/AAC uniformly)
    try:
        with av.open(str(file_path), mode="r") as container:
            audio_streams = [s for s in container.streams if s.type == "audio"]
            if not audio_streams:
                raise AudioIngestionError("File contains no decodable audio streams.")
            stream = audio_streams[0]
            codec_ctx = stream.codec_context

            info["container_format"] = (container.format.name or info["container_format"]).split(",")[0]
            info["codec"] = codec_ctx.name or "unknown"
            info["sample_rate"] = int(codec_ctx.sample_rate or stream.rate or 0)
            info["channels"] = int(codec_ctx.channels or 0)

            if container.duration is not None and container.duration > 0:
                info["duration_seconds"] = float(container.duration) / float(av.time_base)
            elif stream.duration is not None and stream.time_base is not None:
                info["duration_seconds"] = float(stream.duration * stream.time_base)

            if container.bit_rate:
                info["bitrate"] = int(container.bit_rate)
            elif codec_ctx.bit_rate:
                info["bitrate"] = int(codec_ctx.bit_rate)

            # Decode first few frames and count total samples if duration is uncertain
            decoded_samples = 0
            frames_checked = 0
            for frame in container.decode(stream):
                arr = frame.to_ndarray()
                if arr.size == 0:
                    continue
                frames_checked += 1
                decoded_samples += frame.samples
                if info["duration_seconds"] > 0 and frames_checked >= 5:
                    break

            if frames_checked == 0:
                raise AudioIngestionError("Audio stream could not be decoded (0 valid frames).")

            if info["duration_seconds"] <= 0 and info["sample_rate"] > 0:
                info["duration_seconds"] = float(decoded_samples) / float(info["sample_rate"])
    except AudioIngestionError:
        raise
    except Exception as exc:
        raise AudioIngestionError(f"Failed to decode audio container: {exc}") from exc

    # Enrich bit depth and exact sample frames via SoundFile when supported (WAV/FLAC/MP3)
    try:
        sf_info = sf.info(str(file_path))
        if sf_info.samplerate > 0:
            info["sample_rate"] = int(sf_info.samplerate)
        if sf_info.channels > 0:
            info["channels"] = int(sf_info.channels)
        if sf_info.duration > 0:
            info["duration_seconds"] = float(sf_info.duration)
        subtype = (sf_info.subtype or "").upper()
        if "PCM_16" in subtype:
            info["bit_depth"] = 16
        elif "PCM_24" in subtype:
            info["bit_depth"] = 24
        elif "PCM_32" in subtype or "FLOAT" in subtype:
            info["bit_depth"] = 32
        elif "PCM_U8" in subtype or "PCM_S8" in subtype:
            info["bit_depth"] = 8
    except Exception:
        pass

    # Enrich bitrate/bit_depth via Mutagen if still missing
    try:
        m_file = mutagen.File(str(file_path))
        if m_file is not None and hasattr(m_file, "info"):
            m_info = m_file.info
            if getattr(m_info, "bitrate", None) and not info["bitrate"]:
                info["bitrate"] = int(m_info.bitrate)
            if getattr(m_info, "bits_per_sample", None) and not info["bit_depth"]:
                info["bit_depth"] = int(m_info.bits_per_sample)
            if getattr(m_info, "length", None) and info["duration_seconds"] <= 0:
                info["duration_seconds"] = float(m_info.length)
    except Exception:
        pass

    if info["sample_rate"] <= 0 or info["channels"] <= 0 or info["duration_seconds"] <= 0:
        raise AudioIngestionError("Audio file has invalid technical parameters (zero duration, sample rate, or channels).")

    if info["duration_seconds"] > settings.SPECTRA_MAX_DURATION_SEC:
        raise AudioIngestionError(
            f"Audio duration ({info['duration_seconds']:.1f}s) exceeds maximum limit ({settings.SPECTRA_MAX_DURATION_SEC:.0f}s).",
            status_code=413,
        )

    return info


def ingest_uploaded_bytes(
    db: Session,
    raw_bytes: bytes,
    original_filename: str,
    is_processed_variant: bool = False,
    parent_asset_id: Optional[str] = None,
    variant_description: Optional[str] = None,
) -> AudioAsset:
    """Validate, hash, archive, and register an uploaded or generated audio file."""
    max_bytes = settings.SPECTRA_MAX_UPLOAD_MB * 1024 * 1024
    if len(raw_bytes) == 0:
        raise AudioIngestionError("Uploaded file is empty (0 bytes).")
    if len(raw_bytes) > max_bytes:
        raise AudioIngestionError(
            f"File size ({len(raw_bytes) / (1024 * 1024):.2f} MB) exceeds limit of {settings.SPECTRA_MAX_UPLOAD_MB} MB.",
            status_code=413,
        )

    sanitized_name = sanitize_filename(original_filename)
    ext = Path(sanitized_name).suffix.lower()
    fmt_upper = verify_magic_bytes(raw_bytes[:64], ext)

    sha256_hex = hashlib.sha256(raw_bytes).hexdigest()
    asset_id = str(uuid.uuid4())

    target_subdir = "processed" if is_processed_variant else "originals"
    stored_filename = f"{asset_id}{ext}"
    rel_path = f"{target_subdir}/{stored_filename}"
    abs_path = settings.storage_path / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)

    # Write exact bytes untouched
    abs_path.write_bytes(raw_bytes)

    try:
        tech = inspect_audio_technical_metadata(abs_path)
    except Exception:
        if abs_path.exists():
            try:
                abs_path.unlink()
            except OSError:
                pass
        raise

    expires_at = datetime.now(timezone.utc) + timedelta(hours=settings.SPECTRA_RETENTION_HOURS)

    asset = AudioAsset(
        id=asset_id,
        original_filename=os.path.basename(original_filename.replace("\\", "/"))[:255],
        sanitized_filename=sanitized_name,
        storage_rel_path=rel_path,
        original_format=fmt_upper,
        mime_type=EXTENSION_MIME_MAP[ext],
        codec=tech["codec"],
        container_format=tech["container_format"],
        duration_seconds=round(float(tech["duration_seconds"]), 4),
        sample_rate=int(tech["sample_rate"]),
        channels=int(tech["channels"]),
        bit_depth=tech["bit_depth"],
        bitrate=tech["bitrate"],
        file_size_bytes=len(raw_bytes),
        sha256_checksum=sha256_hex,
        is_processed_variant=is_processed_variant,
        parent_asset_id=parent_asset_id,
        variant_description=variant_description,
        expires_at=expires_at,
    )
    db.add(asset)
    db.commit()
    db.refresh(asset)
    return asset


def resolve_asset_path(asset: AudioAsset) -> Path:
    """Safely resolve the internal storage path for an AudioAsset, preventing traversal."""
    root = settings.storage_path.resolve()
    candidate = (root / asset.storage_rel_path).resolve()
    if not str(candidate).startswith(str(root)):
        raise AudioIngestionError("Storage path traversal violation detected.", status_code=403)
    if not candidate.exists():
        raise AudioIngestionError("Archived audio file not found on disk.", status_code=404)
    return candidate


def decode_audio_pcm(
    file_path: Path,
    target_sr: Optional[int] = None,
    mono: bool = False,
    max_duration: Optional[float] = None,
    start_sec: float = 0.0,
) -> Tuple[np.ndarray, int]:
    """
    Decode an audio file into float32 PCM ndarray of shape (channels, samples)
    without modifying the archived file.
    """
    samples: Optional[np.ndarray] = None
    sr: int = 0

    # Try soundfile first for WAV/FLAC/MP3
    try:
        data, sr = sf.read(str(file_path), dtype="float32", always_2d=True)
        # soundfile returns (samples, channels) -> transpose to (channels, samples)
        samples = data.T
    except Exception:
        # Fallback to PyAV for M4A/AAC or complex containers
        with av.open(str(file_path), mode="r") as container:
            stream = next(s for s in container.streams if s.type == "audio")
            sr = int(stream.codec_context.sample_rate or stream.rate)
            resampler = av.AudioResampler(format="fltp", rate=sr)
            chunks = []
            for frame in container.decode(stream):
                for out_frame in resampler.resample(frame):
                    chunks.append(out_frame.to_ndarray())
            for out_frame in resampler.resample(None):
                chunks.append(out_frame.to_ndarray())
            if not chunks:
                raise AudioIngestionError("No audio samples decoded from stream.")
            samples = np.concatenate(chunks, axis=1).astype(np.float32)

    if samples is None or samples.size == 0 or sr <= 0:
        raise AudioIngestionError("Decoded audio buffer is empty.")

    if start_sec > 0:
        start_idx = min(int(start_sec * sr), samples.shape[1])
        samples = samples[:, start_idx:]

    if max_duration is not None and max_duration > 0:
        max_idx = int(max_duration * sr)
        if samples.shape[1] > max_idx:
            samples = samples[:, :max_idx]

    if mono and samples.shape[0] > 1:
        samples = np.mean(samples, axis=0, keepdims=True).astype(np.float32)

    if target_sr is not None and target_sr != sr:
        # Resample each channel using high-quality libsoxr
        resampled_channels = [
            soxr.resample(samples[ch], sr, target_sr, quality="HQ")
            for ch in range(samples.shape[0])
        ]
        samples = np.stack(resampled_channels, axis=0).astype(np.float32)
        sr = target_sr

    return samples, sr
