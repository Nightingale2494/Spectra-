"""AI-Detection package for SPECTRA."""
from .base import BaseAudioDetector, DetectorPrediction
from .fakeprint_detector import FakeprintDetector
from .sonics_detector import SonicsDetector
from .registry import DetectorRegistry, detector_registry

__all__ = [
    "BaseAudioDetector",
    "DetectorPrediction",
    "FakeprintDetector",
    "SonicsDetector",
    "DetectorRegistry",
    "detector_registry",
]
