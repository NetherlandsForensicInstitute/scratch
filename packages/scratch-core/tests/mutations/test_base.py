from returns.pipeline import is_successful
from container_models.scan_image import ScanImage
import pytest
from mutations.base import ImageMutation
import numpy as np


class TestBaseMutations:
    @pytest.fixture
    def scan_image(
        self,
    ) -> ScanImage:
        return ScanImage(data=np.zeros((2, 2)), scale_x=1, scale_y=1)

    class FakeMutation(ImageMutation):
        @property
        def skip_predicate(self) -> bool:
            return self.var == 3

        def __init__(self, var: int) -> None:
            self.var = var

        def apply_on_image(self, scan_image: ScanImage) -> ScanImage:
            """Small edit to do a 'mutation'"""
            scan_image.scale_x = self.var
            scan_image.scale_y = self.var
            return scan_image

    @pytest.mark.parametrize(
        "get_result",
        [
            pytest.param(
                lambda mutation, scan_image: mutation.apply_on_image(scan_image),
                id="apply_on_image",
            ),
            pytest.param(
                lambda mutation, scan_image: mutation(scan_image).unwrap(),
                id="call_interface",
            ),
        ],
    )
    def test_returns_edited_image(self, scan_image: ScanImage, get_result):
        # Arrange
        updated_variable = 2
        mutation = self.FakeMutation(var=updated_variable)
        # Act
        result = get_result(mutation, scan_image)
        # Assert
        assert result.scale_x == updated_variable
        assert result.scale_y == updated_variable

    def test_call_returns_success(self, scan_image: ScanImage):
        # Arrange
        mutation = self.FakeMutation(var=2)
        # Act
        result = mutation(scan_image)
        # Assert
        assert is_successful(result)

    def test_call_wraps_exception_in_failure(
        self, scan_image: ScanImage, monkeypatch: pytest.MonkeyPatch
    ):
        # Arrange
        def raise_error(_):
            raise RuntimeError("boom")

        monkeypatch.setattr(self.FakeMutation, "apply_on_image", raise_error)
        mutation = self.FakeMutation(var=2)

        # Act
        result = mutation(scan_image)
        # Assert
        assert not is_successful(result)

    def test_interface_skips_edit_image_with_predicate(self, scan_image: ScanImage):
        mutation = self.FakeMutation(var=3)
        # Act
        resulting_scan_image = mutation(scan_image=scan_image).unwrap()
        # Assert
        assert resulting_scan_image == scan_image, "Mutation should be skipped."
