from typing import Any

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


class TestAsDict:
    """Tests for the as_dict() method of UploadScanParameters."""

    def test_as_dict_returns_all_fields_by_default(self, params: UploadScanParameters) -> None:
        """Test that as_dict returns all model fields when no parameters are provided."""
        # Act
        result = params.as_dict()

        # Assert
        assert set(result) == set(params.model_dump())
        assert result["light_sources"] == params.light_sources
        assert result["observer"] == params.observer
        assert result["scale_x"] == params.scale_x
        assert result["scale_y"] == params.scale_y
        assert result["step_size_x"] == params.step_size_x
        assert result["step_size_y"] == params.step_size_y

    def test_as_dict_preserves_nested_objects(self) -> None:
        """Test that nested objects (LightSource) are not serialized."""
        # Arrange
        custom_light = LightSource(azimuth=45, elevation=30)
        params = UploadScanParameters(light_sources=(custom_light,))  # type: ignore
        dump = params.model_dump()

        # Act
        result = params.as_dict()

        # Assert
        assert result["light_sources"] != dump["light_sources"]
        assert isinstance(result["light_sources"], tuple)
        assert result["light_sources"][0] is custom_light
        assert result["observer"] != dump["observer"]
        assert isinstance(result["observer"], LightSource)

    @pytest.mark.parametrize(
        "input",
        [
            {"include": {"scale_x", "scale_y"}},
            {"exclude": {"scale_x", "scale_y"}},
            {"include": {"scale_x", "nonexistent_field"}},
            {"exclude": {"scale_x", "nonexistent_field"}},
            {"include": {"nonexistent_field"}},
            {"exclude": {"nonexistent_field"}},
            {"include": set()},
            {"exclude": set()},
            {"exclude": {"scale_x"}, "include": {"scale_y"}},
        ],
    )
    def test_similar_behavior_to_model_dump(self, input: dict[str, Any], params: UploadScanParameters) -> None:
        """Test that as_dict behaves similarly to model_dump."""
        # Act
        result = params.as_dict(**input)

        # Assert - all fields should be present
        assert set(result) == set(params.model_dump(**input))
