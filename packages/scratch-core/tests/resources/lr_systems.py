import numpy as np
from lir.data.models import FeatureData, LLRData
from lir.lrsystems.lrsystems import LRSystem


class RandomLRSystem(LRSystem):
    """LRSystem that returns seeded random LLR values, for use in tests."""

    def __init__(self) -> None:
        super().__init__(name="random")

    def apply(self, instances: FeatureData) -> LLRData:
        """Return seeded random LLR values, one per input instance."""
        n = len(instances.features)
        rng = np.random.default_rng(seed=42)
        return LLRData(features=rng.random(n))
