from http import HTTPStatus

import numpy as np
import pytest
from container_models.base import Pair
from container_models.image import ImageContainer, MaskImage, MetaData
from fastapi import HTTPException
from mutations.base import ImageMutation
from returns.pipeline import pipe

from pipelines import run_pipeline


@pytest.fixture
def flat_scale() -> MetaData:
    return MetaData(scale=Pair(x=1.0, y=1.0))


def test_pipeline_mutates_image(flat_scale: MetaData) -> None:
    # Arrange
    shape = 2, 2

    class FakeMuation(ImageMutation):
        def apply_on_image(self, image: ImageContainer) -> ImageContainer:
            image.data = np.ones(image.data.shape)
            return image

    piplines = pipe(FakeMuation())
    mask_image = MaskImage(data=np.zeros(shape), metadata=flat_scale)
    # Act
    value = run_pipeline(piplines, mask_image, error_message="Failed pipeline")
    # Assert
    assert value == MaskImage(data=np.ones(shape), metadata=flat_scale)


def test_pipeline_failure_raises_http_exception(flat_scale: MetaData) -> None:
    """Test that pipeline failures raise HTTPException with status 500."""

    class FakeMuation(ImageMutation):
        def apply_on_image(self, _):  # type: ignore
            raise ValueError("Kaboem, that is broken")

    piplines = FakeMuation()
    mask_image = MaskImage(data=np.zeros((2, 2)), metadata=flat_scale)
    # Act & Assert
    with pytest.raises(HTTPException, match="Failed pipeline") as exc_info:
        run_pipeline(piplines, mask_image, error_message="Failed pipeline")

    assert exc_info.value.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
