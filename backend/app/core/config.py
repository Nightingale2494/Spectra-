"""Environment configuration and path management for SPECTRA."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    SPECTRA_ENV: str = "development"
    SPECTRA_VERSION: str = "1.0.0"
    SPECTRA_HOST: str = "127.0.0.1"
    SPECTRA_PORT: int = 8000
    SPECTRA_FRONTEND_URL: str = "http://localhost:3000"
    SPECTRA_CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"

    SPECTRA_STORAGE_DIR: str = str(PROJECT_ROOT / "storage")
    SPECTRA_MODELS_DIR: str = str(PROJECT_ROOT / "models")
    SPECTRA_DB_URL: str = f"sqlite:///{(PROJECT_ROOT / 'storage' / 'spectra.db').as_posix()}"

    SPECTRA_MAX_UPLOAD_MB: int = 100
    SPECTRA_MAX_DURATION_SEC: float = 600.0
    SPECTRA_RETENTION_HOURS: int = 168
    SPECTRA_MAX_WORKERS: int = 2
    SPECTRA_DEFAULT_DEVICE: str = "cpu"
    SPECTRA_ENABLE_SONICS_ALPHA_120S: bool = True

    SPECTRA_FAKEPRINT_MODEL_PATH: str = str(
        PROJECT_ROOT / "models" / "fakeprint" / "ai_music_detector.onnx"
    )
    SPECTRA_FAKEPRINT_CONFIG_PATH: str = str(
        PROJECT_ROOT / "models" / "fakeprint" / "config.yaml"
    )
    SPECTRA_FAKEPRINT_REVISION: str = "6ba389e94a179ac90f3eb134b741ef37baa30434"

    SPECTRA_ENABLE_SONICS: bool = True
    SPECTRA_SONICS_DEFAULT_REPO: str = "awsaf49/sonics-spectttra-gamma-5s"
    SPECTRA_SONICS_REVISION: str = "9156ffad151f797c71556923c4a02fa01fa8fc91"
    SPECTRA_SONICS_AUTO_DOWNLOAD: bool = False

    SPECTRA_COMFYUI_ENABLED: bool = True
    SPECTRA_COMFYUI_URL: str = "http://127.0.0.1:8188"
    SPECTRA_COMFYUI_INSTALL_PATH: str = r"D:\ComfyUI\Nyx\ComfyUI"
    SPECTRA_COMFYUI_SHARED_DIR: str = r"C:\Users\user\AppData\Local\Comfy-Desktop\ComfyUI-Shared"

    @property
    def storage_path(self) -> Path:
        p = Path(self.SPECTRA_STORAGE_DIR).resolve()
        for sub in ("originals", "processed", "reports", "temp", "samples"):
            (p / sub).mkdir(parents=True, exist_ok=True)
        return p

    @property
    def originals_dir(self) -> Path:
        return self.storage_path / "originals"

    @property
    def processed_dir(self) -> Path:
        return self.storage_path / "processed"

    @property
    def reports_dir(self) -> Path:
        return self.storage_path / "reports"

    @property
    def temp_dir(self) -> Path:
        return self.storage_path / "temp"

    @property
    def samples_dir(self) -> Path:
        return self.storage_path / "samples"

    @property
    def models_path(self) -> Path:
        p = Path(self.SPECTRA_MODELS_DIR).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.SPECTRA_CORS_ORIGINS.split(",") if o.strip()]

    def resolve_ffmpeg_exe(self) -> Optional[str]:
        sys_ffmpeg = shutil.which("ffmpeg")
        if sys_ffmpeg:
            return sys_ffmpeg
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None

    def resolve_ffprobe_exe(self) -> Optional[str]:
        return shutil.which("ffprobe")


settings = Settings()
