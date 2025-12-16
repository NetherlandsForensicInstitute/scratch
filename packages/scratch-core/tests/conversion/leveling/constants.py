from conversion.leveling import SurfaceTerms
from pathlib import Path


SURFACE_TERMS = list(SurfaceTerms)
ALL_TERMS = SURFACE_TERMS + [SurfaceTerms.PLANE, SurfaceTerms.SPHERE]


TEST_ROOT = Path(__file__).parent
RESOURCES_DIR = TEST_ROOT / "resources"
