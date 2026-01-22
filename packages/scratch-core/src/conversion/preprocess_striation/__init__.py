"""
Preprocessing pipeline for striated tool and bullet marks.

This package provides functions for preprocessing striation marks:
- Form and noise removal via band-pass filtering
- Fine alignment to make striations horizontal
- Profile extraction
"""

from conversion.preprocess_striation.parameters import PreprocessingStriationParams
from conversion.preprocess_striation.pipeline import (
    preprocess_data,
    apply_shape_noise_removal,
)
from conversion.preprocess_striation.alignment import fine_align_bullet_marks

__all__ = [
    "PreprocessingStriationParams",
    "preprocess_data",
    "apply_shape_noise_removal",
    "fine_align_bullet_marks",
]
