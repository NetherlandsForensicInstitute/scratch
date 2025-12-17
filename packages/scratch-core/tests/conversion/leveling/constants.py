from conversion.leveling import SurfaceTerms
from pathlib import Path


SINGLE_TERMS = list(SurfaceTerms)
COMBINED_TERMS = [SurfaceTerms.PLANE, SurfaceTerms.SPHERE]
SINGLE_AND_COMBINED_TERMS = SINGLE_TERMS + COMBINED_TERMS


TEST_ROOT = Path(__file__).parent
RESOURCES_DIR = TEST_ROOT / "resources"
