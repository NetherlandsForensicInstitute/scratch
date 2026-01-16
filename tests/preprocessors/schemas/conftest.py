import pytest

from preprocessors.schemas import EditImageParameters, Mask


@pytest.fixture(scope="module")
def mask() -> Mask:
    return Mask(data=b"\x00\x01\x00\x01\x00\x01", shape=(2, 3))


@pytest.fixture(scope="module")
def edit_image_parameters(mask: Mask) -> EditImageParameters:
    return EditImageParameters(mask=mask, cutoff_length=250)  # type: ignore
