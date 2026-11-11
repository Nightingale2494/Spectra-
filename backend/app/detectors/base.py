"""Base detector interface and common typed result for SPECTRA AI-Detection System."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class DetectorPrediction:
    detector_id: str
    detector_name: str
    detector_version: str
    model_source: str
    model_revision: str
    probability: Optional[float]
    raw_score: Optional[float]
    threshold: float
    predicted_label: str
    confidence: Optional[float]
    input_sample_rate: int
    input_duration_sec: float
    processing_time_ms: float
    device_used: str
    feature_dim: Optional[int] = None
    status: str = "success"  # success | unavailable | error
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class BaseAudioDetector(ABC):
    detector_id: str
    name: str
    version: str
    model_source: str
    model_revision: str
    license: str
    input_sample_rate: int
    input_duration_requirements: str
    supported_devices: List[str]
    description: str

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if model weights and runtime dependencies are present."""
        ...

    @abstractmethod
    def load(self, device: Optional[str] = None) -> bool:
        """Load the model into memory. Raise or return False on failure."""
        ...

    @abstractmethod
    def predict(
        self,
        samples: np.ndarray,
        sr: int,
        precomputed_features: Optional[np.ndarray] = None,
    ) -> DetectorPrediction:
        """Run inference on decoded PCM audio (channels, samples) and return typed prediction."""
        ...

    @abstractmethod
    def info(self) -> Dict[str, Any]:
        """Return structured metadata and health status for this detector."""
        ...
