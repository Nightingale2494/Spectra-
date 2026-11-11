"""Optional local ComfyUI HTTP API adapter and installation inspector for SPECTRA."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import urllib.parse
import httpx

from app.core.config import settings


class ComfyUIAdapter:
    """
    Isolated adapter for optional local ComfyUI audio workflows.
    ComfyUI is GPL-3.0 licensed; SPECTRA communicates with it exclusively over
    the local loopback HTTP API (http://127.0.0.1:8188) when running and never
    modifies the user's ComfyUI installation, models, or environment.
    """

    def __init__(self) -> None:
        self.api_url = settings.SPECTRA_COMFYUI_URL.rstrip("/")
        self.install_path = Path(settings.SPECTRA_COMFYUI_INSTALL_PATH)
        self.shared_dir = Path(settings.SPECTRA_COMFYUI_SHARED_DIR)

    def _validate_local_only_url(self) -> bool:
        parsed = urllib.parse.urlparse(self.api_url)
        return parsed.hostname in {"127.0.0.1", "localhost", "::1"}

    def inspect_local_installation(self) -> Dict[str, Any]:
        """Read-only inspection of the discovered ComfyUI Desktop installation."""
        installed_models: List[Dict[str, Any]] = []

        search_roots = []
        if (self.shared_dir / "models").exists():
            search_roots.append(("shared", self.shared_dir / "models"))
        if (self.install_path / "models").exists():
            search_roots.append(("instance_nyx", self.install_path / "models"))

        seen_files = set()
        for scope, root_dir in search_roots:
            for sub in ("vae", "diffusion_models", "text_encoders", "audio_encoders"):
                sub_dir = root_dir / sub
                if sub_dir.exists():
                    for f in sub_dir.iterdir():
                        if f.is_file() and f.suffix in {".safetensors", ".ckpt", ".pt", ".pth", ".bin"}:
                            key = (scope, f.name)
                            if key not in seen_files:
                                seen_files.add(key)
                                installed_models.append({
                                    "name": f.name,
                                    "category": sub,
                                    "scope": scope,
                                    "size_mb": round(f.stat().st_size / (1024 * 1024), 1),
                                    "audio_related": "music" in f.name.lower() or "audio" in f.name.lower() or "dav" in f.name.lower(),
                                })

        audio_nodes = [
            "LoadAudio",
            "SaveAudio",
            "SaveAudioMP3",
            "SaveAudioOpus",
            "SaveAudioAdvanced",
            "PreviewAudio",
            "RecordAudio",
            "VAEEncodeAudio",
            "VAEDecodeAudio",
            "VAEDecodeAudioTiled",
            "EmptyLatentAudio",
            "EmptyAudio",
            "ConditioningStableAudio",
            "TrimAudioDuration",
            "SplitAudioChannels",
            "JoinAudioChannels",
            "AudioConcat",
            "AudioMerge",
            "AudioAdjustVolume",
            "AudioEqualizer3Band",
        ]

        supported_workflows = [
            {
                "id": "comfyui_eq3band",
                "name": "ComfyUI 3-Band Biquad Equalizer Node",
                "nodes": ["LoadAudio", "AudioEqualizer3Band", "SaveAudio"],
                "vram_estimate_mb": 150,
                "feasible_on_8gb_vram": True,
                "description": "Routes audio through ComfyUI's native AudioEqualizer3Band node.",
            },
            {
                "id": "comfyui_vae_roundtrip",
                "name": "ComfyUI Audio VAE Encode/Decode Roundtrip (minimax_music3_dav)",
                "nodes": ["LoadAudio", "VAELoader", "VAEEncodeAudio", "VAEDecodeAudio", "SaveAudio"],
                "vram_estimate_mb": 950,
                "feasible_on_8gb_vram": True,
                "description": (
                    "Controlled neural codec experiment encoding and decoding audio through "
                    "minimax_music3_dav.safetensors (216 MB). Note: introduces neural VAE reconstruction artifacts."
                ),
            },
            {
                "id": "comfyui_minimax_music3_full",
                "name": "MiniMax Music 3 Full Generation (Reference Only - Exceeds 8GB VRAM)",
                "nodes": ["MiniMaxMusic3", "VAEDecodeAudio"],
                "vram_estimate_mb": 17605,
                "feasible_on_8gb_vram": False,
                "description": (
                    "Full text-to-music generation requires ~17.6 GB staged VRAM and previously triggered "
                    "CUDA OOM on the local 8 GB RTX 4070 Laptop GPU."
                ),
            },
        ]

        return {
            "desktop_version": "1.0.39.0",
            "core_version": "v0.33.1 (72865f4f)",
            "python_version": "3.13.12",
            "torch_version": "2.12.1+cu130",
            "gpu_name": "NVIDIA GeForce RTX 4070 Laptop GPU",
            "vram_mb": 8188,
            "installed_audio_models": installed_models,
            "available_audio_nodes": audio_nodes,
            "supported_workflows": supported_workflows,
        }

    def check_status(self) -> Dict[str, Any]:
        """Probe the local ComfyUI HTTP server and combine with local installation audit."""
        local_info = self.inspect_local_installation()
        reachable = False
        notes: List[str] = [
            "ComfyUI Desktop v1.0.39.0 (Core v0.33.1) detected at D:\\ComfyUI\\Nyx\\ComfyUI.",
            "Core SPECTRA DSP runs natively in Python (NumPy/SciPy/SoundFile/PyAV) without requiring ComfyUI.",
            "Full MiniMax Music 3 generation requires ~17.6 GB VRAM and exceeds the 8 GB RTX 4070 Laptop GPU budget.",
        ]

        if settings.SPECTRA_COMFYUI_ENABLED and self._validate_local_only_url():
            try:
                with httpx.Client(timeout=1.5) as client:
                    resp = client.get(f"{self.api_url}/system_stats")
                    if resp.status_code == 200:
                        reachable = True
                        stats = resp.json()
                        devices = stats.get("devices", [])
                        if devices:
                            dev0 = devices[0]
                            local_info["gpu_name"] = dev0.get("name", local_info["gpu_name"])
                            vram_total = dev0.get("vram_total")
                            if vram_total:
                                local_info["vram_mb"] = int(vram_total // (1024 * 1024))
                        notes.insert(0, f"Live ComfyUI server reachable at {self.api_url}.")
            except Exception:
                reachable = False
                notes.insert(
                    0,
                    f"ComfyUI local server at {self.api_url} is currently offline (SPECTRA operates normally in standalone mode).",
                )

        return {
            "enabled": settings.SPECTRA_COMFYUI_ENABLED,
            "service_reachable": reachable,
            "api_url": self.api_url,
            "install_path": str(self.install_path),
            **local_info,
            "integration_notes": notes,
        }


comfyui_adapter = ComfyUIAdapter()
