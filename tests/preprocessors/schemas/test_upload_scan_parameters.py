import pytest
from container_models.light_source import LightSource
from pydantic import ValidationError

from preprocessors.schemas import UploadScanParameters


@pytest.fixture(scope="class")
def params() -> UploadScanParameters:
    return UploadScanParameters.model_construct()


class TestUploadScanParameters:
    """Tests for UploadScanParameters model."""

    def test_default_values(self, params: UploadScanParameters) -> None:
        """Test that default parameters are set correctly."""
        # Assert
        assert params.light_sources == (
            LightSource(azimuth=90, elevation=45),
            LightSource(azimuth=180, elevation=45),
        )
        assert params.observer == LightSource(azimuth=90, elevation=45)
        assert params.scale_x == 1.0
        assert params.scale_y == 1.0
        assert params.step_size_x == 1
        assert params.step_size_y == 1

    def test_custom_parameters(self) -> None:
        """Test that custom parameters can be set."""
        # Arrange
        custom_light = LightSource(azimuth=45, elevation=30)
        custom_observer = LightSource(azimuth=0, elevation=90)

        # Act
        params = UploadScanParameters(  # type: ignore
            light_sources=(custom_light,),
            observer=custom_observer,
            scale_x=2.5,
            scale_y=3.0,
            step_size_x=2,
            step_size_y=3,
        )

        # Assert
        assert params.light_sources == (custom_light,)
        assert params.observer == custom_observer
        assert params.scale_x == 2.5  # noqa: PLR2004
        assert params.scale_y == 3.0  # noqa: PLR2004
        assert params.step_size_x == 2  # noqa: PLR2004
        assert params.step_size_y == 3  # noqa: PLR2004

    @pytest.mark.parametrize(
        ("field_name", "invalid_value"),
        [
            ("scale_x", 0.0),
            ("scale_x", -1.0),
            ("scale_y", 0.0),
            ("scale_y", -1.5),
            ("step_size_x", 0),
            ("step_size_x", -1),
            ("step_size_y", 0),
            ("step_size_y", -2),
        ],
    )
    def test_invalid_scale_and_step_values(self, field_name: str, invalid_value: float | int) -> None:
        """Test that scale and step size values must be positive."""
        # Arrange
        valid_params = {
            "scale_x": 1.0,
            "scale_y": 1.0,
            "step_size_x": 1,
            "step_size_y": 1,
        }
        valid_params[field_name] = invalid_value

        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:  # Pydantic raises ValidationError
            UploadScanParameters(**valid_params)  # type: ignore

        # Verify the error is related to the constraint
        assert "greater than" in str(exc_info.value).lower()
