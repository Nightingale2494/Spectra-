"""Central Detector Registry for SPECTRA."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from app.detectors.base import BaseAudioDetector, DetectorPrediction
from app.detectors.fakeprint_detector import FakeprintDetector
from app.detectors.sonics_detector import SonicsDetector


class DetectorRegistry:
    """Manages registered AI-music detectors without conflating or averaging unrelated models."""

    def __init__(self) -> None:
        self._detectors: Dict[str, BaseAudioDetector] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        fp = FakeprintDetector()
        if fp.is_available():
            fp.load()
        self.register(fp)

        sonics_5s = SonicsDetector(
            detector_id="sonics_gamma_5s",
            name="SONICS SpecTTTra-γ (5s Spectro-Temporal Transformer)",
            hf_repo="awsaf49/sonics-spectttra-gamma-5s",
            hf_revision="a349d427b681fa15e3d0e22d1708108322f967b1",
            local_subdir="sonics-spectttra-gamma-5s",
            window_seconds=5.0,
        )
        if sonics_5s.is_available():
            sonics_5s.load()
        self.register(sonics_5s)

        sonics_120s = SonicsDetector(
            detector_id="sonics_alpha_120s",
            name="SONICS SpecTTTra-α (120s Long-Context Classifier)",
            hf_repo="awsaf49/sonics-spectttra-alpha-120s",
            hf_revision="094b32a5545098a71c113f6ae9d5c55310564268",
            local_subdir="sonics-spectttra-alpha-120s",
            window_seconds=120.0,
        )
        if sonics_120s.is_available():
            sonics_120s.load()
        self.register(sonics_120s)

    def register(self, detector: BaseAudioDetector) -> None:
        self._detectors[detector.detector_id] = detector

    def get(self, detector_id: str) -> Optional[BaseAudioDetector]:
        return self._detectors.get(detector_id)

    def list_all(self) -> List[Dict[str, Any]]:
        return [det.info() for det in self._detectors.values()]

    def run_all(
        self,
        samples: np.ndarray,
        sr: int,
        precomputed_fakeprint: Optional[np.ndarray] = None,
    ) -> List[DetectorPrediction]:
        results: List[DetectorPrediction] = []
        for det_id, det in self._detectors.items():
            if det_id == "fakeprint_onnx":
                pred = det.predict(samples, sr, precomputed_features=precomputed_fakeprint)
            else:
                pred = det.predict(samples, sr)
            results.append(pred)
        return results


detector_registry = DetectorRegistry()
