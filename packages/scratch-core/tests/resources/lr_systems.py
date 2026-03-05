import numpy as np
from lir.data.models import FeatureData, LLRData, InstanceData
from lir.lrsystems.lrsystems import LRSystem


class RandomLRSystem(LRSystem):
    """LRSystem that returns seeded random LLR values, for use in tests."""

    def __init__(self) -> None:
        pass

    def apply(self, instances: InstanceData) -> LLRData:
        """Return seeded random LLR values, one per input instance."""
        assert isinstance(instances, FeatureData)
        n = len(instances.features)
        rng = np.random.default_rng(seed=42)
        return LLRData(features=rng.random(n))
