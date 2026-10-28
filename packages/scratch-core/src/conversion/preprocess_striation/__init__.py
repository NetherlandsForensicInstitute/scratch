"""
Preprocessing pipeline for striated tool and bullet marks.

This package provides functions for preprocessing striation marks:
- Form and noise removal via band-pass filtering
- Fine alignment to make striations horizontal
- Profile extraction
"""

from .alignment import fine_align_bullet_marks
from .parameters import PreprocessingStriationParams
from .pipeline import (
    apply_shape_noise_removal,
    preprocess_striation_mark,
)

__all__ = [
    "PreprocessingStriationParams",
    "apply_shape_noise_removal",
    "fine_align_bullet_marks",
    "preprocess_striation_mark",
]
