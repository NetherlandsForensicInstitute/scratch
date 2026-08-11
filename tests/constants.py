from typing import Final

import numpy as np

DEFAULT_RESAMPLING_FACTOR: Final[int] = 4
DEFAULT_STEP_SIZE: Final[int] = 1
CUTOFF_LENGTH: Final[float] = 250e-6
MASK = np.array([[True, False, True], [False, True, False]], dtype=np.bool_)
MASK_BYTES = MASK.tobytes(order="C")
MASK_SHAPE = MASK.shape
