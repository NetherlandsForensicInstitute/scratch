from container_models.scan_image import ScanImage
import pytest
from renders.mutations.base import ImageMutation
import numpy as np
from returns.result import Success, Failure


class TestBaseMutations:
    @pytest.fixture
    def scan_image(
        self,
    ) -> ScanImage:
        return ScanImage(data=np.zeros((2, 2)), scale_x=1, scale_y=1)

    class FakeMutation(ImageMutation):
        def __init__(self, var: int) -> None:
            self.var = var

        def apply_on_image(self, scan_image: ScanImage) -> ScanImage:
            """Small edit to do a 'mutation'"""
            scan_image.scale_x = self.var
            scan_image.scale_y = self.var
            return scan_image

    def test_apply_image_returns_editted_image(self, scan_image: ScanImage):
        # Arrange
        updated_variable = 2
        mutation = self.FakeMutation(var=updated_variable)
        # Act
        result = mutation.apply_on_image(scan_image)
        # Assert
        assert result.scale_x == updated_variable, (
            f"scale shoud be update to {updated_variable}."
        )
        assert result.scale_y == updated_variable, (
            f"scale shoud be update to {updated_variable}."
        )

    def test__call_interface_is_returning_editted_image(self, scan_image: ScanImage):
        # Arrange
        updated_variable = 3
        mutation = self.FakeMutation(var=updated_variable)
        # Act
        result = mutation(scan_image).unwrap()
        # Assert
        assert result.scale_x == updated_variable, (
            f"scale shoud be update to {updated_variable}."
        )
        assert result.scale_y == updated_variable, (
            f"scale shoud be update to {updated_variable}."
        )

    def test_call_returns_success(self, scan_image: ScanImage):
        # Arrange
        mutation = self.FakeMutation(var=2)
        # Act
        result = mutation(scan_image)
        # Assert
        assert isinstance(result, Success)

    def test_call_wraps_exception_in_failure(self, scan_image):
        # Arrange
        class ExplodingMutation(ImageMutation):
            def apply_on_image(self, scan_image: ScanImage) -> ScanImage:
                raise RuntimeError("boom")

        mutation = ExplodingMutation()
        # Act
        result = mutation(scan_image)
        # Assert
        assert isinstance(result, Failure)

    def test_interface_skips_edit_image_with_predicate(self, scan_image: ScanImage):
        # Arrange
        class FakePredicateSkip(self.FakeMutation):
            @property
            def skip_predicate(self) -> bool:
                if self.var == 2:
                    return True
                return False

        mutation = FakePredicateSkip(var=2)
        # Act
        resulting_scan_image = mutation(scan_image=scan_image).unwrap()
        # Assert
        assert resulting_scan_image == scan_image, "Mutation should be skipped."
