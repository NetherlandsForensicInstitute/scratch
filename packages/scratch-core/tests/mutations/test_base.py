from typing import override
from returns.pipeline import is_successful
from container_models import ImageContainer
import pytest
from container_models.base import Pair
from container_models.image import MetaData
from mutations.base import ImageMutation
import numpy as np


class TestBaseMutations:
    @pytest.fixture
    def image_container(self, flat_scale: MetaData) -> ImageContainer:
        return ImageContainer(data=np.zeros((2, 2)), metadata=flat_scale)

    class FakeMutation(ImageMutation):
        @property
        def skip_predicate(self) -> bool:
            return self.var == 3

        def __init__(self, var: int) -> None:
            self.var = var

        @override
        def apply_on_image(self, image: ImageContainer) -> ImageContainer:
            """Small edit to do a 'mutation'"""
            image.metadata.scale = Pair(self.var, self.var)
            return image

    @pytest.mark.parametrize(
        "get_result",
        [
            pytest.param(
                lambda mutation, image_container: mutation.apply_on_image(
                    image_container
                ),
                id="apply_on_image",
            ),
            pytest.param(
                lambda mutation, image_container: mutation(image_container).unwrap(),
                id="call_interface",
            ),
        ],
    )
    def test_returns_edited_image(self, image_container: ImageContainer, get_result):
        # Arrange
        updated_variable = 2
        mutation = self.FakeMutation(var=updated_variable)
        # Act
        result = get_result(mutation, image_container)
        # Assert
        assert result.metadata.scale == (updated_variable, updated_variable)

    def test_call_returns_success(self, image_container: ImageContainer):
        # Arrange
        mutation = self.FakeMutation(var=2)
        # Act
        result = mutation(image_container)
        # Assert
        assert is_successful(result)

    def test_call_wraps_exception_in_failure(
        self, image_container: ImageContainer, monkeypatch: pytest.MonkeyPatch
    ):
        # Arrange
        def raise_error(_):
            raise RuntimeError("boom")

        monkeypatch.setattr(self.FakeMutation, "apply_on_image", raise_error)
        mutation = self.FakeMutation(var=2)

        # Act
        result = mutation(image_container)
        # Assert
        assert not is_successful(result)

    def test_interface_skips_edit_image_with_predicate(
        self, image_container: ImageContainer
    ):
        mutation = self.FakeMutation(var=3)
        # Act
        resulting_image_container = mutation(image_container).unwrap()
        # Assert
        assert resulting_image_container == image_container, (
            "Mutation should be skipped."
        )
