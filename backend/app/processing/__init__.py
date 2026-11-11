"""DSP Processing Laboratory and Before/After Comparison Engine."""
from .dsp_engine import (
    DSP_PRESETS,
    apply_dsp_operation,
    encode_audio_bytes,
    run_dsp_experiment,
)
from .comparison_engine import compute_before_after_comparison

__all__ = [
    "DSP_PRESETS",
    "apply_dsp_operation",
    "encode_audio_bytes",
    "run_dsp_experiment",
    "compute_before_after_comparison",
]
