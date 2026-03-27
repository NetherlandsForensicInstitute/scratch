import numpy as np
import pytest

from container_models.scan_image import ScanImage
from mutations.base import ImageMutation


class TestBaseMutations:
    @pytest.fixture
    def scan_image(
        self,
    ) -> ScanImage:
        return ScanImage(data=np.zeros((2, 2)), scale_x=1, scale_y=1)

    class FakeMutation(ImageMutation):
        def skip_predicate(self, scan_image: ScanImage) -> bool:
            return self.var == 3

        def __init__(self, var: int) -> None:
            self.var = var

        def apply_on_image(self, scan_image: ScanImage) -> ScanImage:
            """Small edit to do a 'mutation'"""
            scan_image.scale_x = self.var
            scan_image.scale_y = self.var
            return ScanImage(
                data=scan_image.data,
                scale_x=scan_image.scale_x,
                scale_y=scan_image.scale_y,
            )

    @pytest.mark.parametrize(
        "get_result",
        [
            pytest.param(
                lambda mutation, scan_image: mutation.apply_on_image(scan_image),
                id="apply_on_image",
            ),
            pytest.param(
                lambda mutation, scan_image: mutation(scan_image),
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
        assert id(result) != id(scan_image)
        assert result.scale_y == updated_variable

    def test_interface_skips_edit_image_with_predicate(self, scan_image: ScanImage):
        # Arrange
        mutation = self.FakeMutation(var=3)
        # Act
        resulting_scan_image = mutation(scan_image=scan_image)
        # Assert
        assert id(resulting_scan_image) == id(scan_image), "Mutation should be skipped."
