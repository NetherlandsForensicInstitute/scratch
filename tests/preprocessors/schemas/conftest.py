import pytest
from scipy.constants import micro

from preprocessors.schemas import EditImageParameters


@pytest.fixture(scope="module")
def edit_image_parameters() -> EditImageParameters:
    return EditImageParameters(mask=((0, 1, 0), (1, 0, 1)), cutoff_length=250 * micro)  # type: ignore
