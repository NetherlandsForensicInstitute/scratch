from conversion.container_models.base import FloatArray1D
from conversion.leveling import SurfaceTerms
from conversion.leveling.solver import build_design_matrix
import numpy as np
import pytest

from ..constants import SINGLE_AND_COMBINED_TERMS, RESOURCES_DIR


@pytest.mark.parametrize("terms", SINGLE_AND_COMBINED_TERMS)
def test_design_matrix_shape_matches_number_of_terms(
    xs: FloatArray1D, ys: FloatArray1D, terms: SurfaceTerms
):
    design_matrix = build_design_matrix(xs=xs, ys=ys, terms=terms)
    assert design_matrix.shape == (len(xs), len(terms))


def test_design_matrix_matches_baseline_output(xs: FloatArray1D, ys: FloatArray1D):
    design_matrix = build_design_matrix(xs=xs, ys=ys, terms=SurfaceTerms.SPHERE)
    verified = np.load(RESOURCES_DIR / "baseline_design_matrix.npy")
    assert np.allclose(design_matrix, verified)
