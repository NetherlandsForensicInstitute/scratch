"""
Preprocessing pipeline for striated tool and bullet marks.

This package provides functions for preprocessing striation marks:
- Form and noise removal via band-pass filtering
- Fine alignment to make striations horizontal
- Profile extraction
"""

from .parameters import PreprocessingStriationParams
from .pipeline import (
    preprocess_striation_mark,
    apply_shape_noise_removal,
)
from .alignment import fine_align_bullet_marks

__all__ = [
    "PreprocessingStriationParams",
    "preprocess_striation_mark",
    "apply_shape_noise_removal",
    "fine_align_bullet_marks",
]
