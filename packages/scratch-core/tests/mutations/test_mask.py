import re
from container_models.scan_image import ScanImage
from exceptions import ImageShapeMismatchError
from mutations.filter import Mask
import numpy as np
import pytest


class TestMask2dArray:
    @pytest.fixture
    def scan_image(
        self,
    ):
        return ScanImage(
            data=np.array([[1, 2], [3, 4]], dtype=float), scale_x=1.0, scale_y=1.0
        )

    def test_mask_sets_background_pixels_to_nan(self, scan_image: ScanImage) -> None:
        # Arrange
        mask = np.array([[1, 0], [0, 1]], dtype=bool)
        masking_mutator = Mask(mask=mask)
        # Act
        result = masking_mutator.apply_on_image(scan_image=scan_image)
        # Assert
        assert np.array_equal(
            result.data, np.array([[1, np.nan], [np.nan, 4]]), equal_nan=True
        )

    def test_raises_on_shape_mismatch(self, scan_image: ScanImage) -> None:
        # Arrange
        mask = np.array([[1, 0, 0], [0, 1, 0]], dtype=bool)
        masking_mutator = Mask(mask=mask)
        # Act / Assert
        with pytest.raises(
            ImageShapeMismatchError,
            match=re.escape(
                f"Mask shape: {mask.shape} does not match image shape: {scan_image.data.shape}"
            ),
        ):
            masking_mutator.apply_on_image(scan_image=scan_image)

    def test_full_mask_preserves_all_values(
        self, scan_image: ScanImage, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange
        mask = np.ones((2, 2), dtype=bool)
        masking_mutator = Mask(mask=mask)
        # Act
        result = masking_mutator.apply_on_image(scan_image=scan_image)
        # Assert
        assert np.array_equal(result.data, scan_image.data, equal_nan=True)

    def test_full_mask_skips_calculation(
        self, scan_image: ScanImage, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange
        mask = np.ones((2, 2), dtype=bool)
        masking_mutator = Mask(mask=mask)
        # Act
        result = masking_mutator(scan_image=scan_image).unwrap()
        # Assert
        assert np.array_equal(result.data, scan_image.data, equal_nan=True)
        assert (
            "skipping masking, Mask area is not containing any masking fields."
            in caplog.messages
        )

    def test_empty_mask_sets_all_to_nan(
        self, scan_image: ScanImage, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange
        mask = np.zeros((2, 2), dtype=bool)
        masking_mutator = Mask(mask=mask)
        result = masking_mutator(scan_image=scan_image).unwrap()

        assert np.all(np.isnan(result.data))
        assert "Applying mask to scan_image" in caplog.messages
