from pathlib import Path

import pytest
from container_models.light_source import LightSource
from container_models.scan_image import ScanImage
from PIL import Image

from preprocessors.pipelines import surface_map_pipeline
from preprocessors.schemas import UploudScanParameters


@pytest.fixture(scope="module")
def light_sources() -> tuple[LightSource, LightSource]:
    """Surface map light sources."""
    return (
        LightSource(azimuth=45, elevation=45),
        LightSource(azimuth=135, elevation=45),
    )


@pytest.fixture(scope="module")
def observer() -> LightSource:
    """Observer position looking straight down from +Z direction."""
    return LightSource(azimuth=0, elevation=90)


@pytest.mark.integration
class TestSurfaceMapPipeline:
    """Integration tests for surface_map_pipeline function."""

    def test_generate_surface_map_success(
        self,
        parsed_al3d_file: ScanImage,
        default_parameters: UploudScanParameters,
        tmp_path: Path,
    ) -> None:
        """Test that a surface map image is successfully generated from scan data."""
        # Arrange
        output_path = tmp_path / "surface_map.png"

        # Act
        result_path = surface_map_pipeline(parsed_al3d_file, output_path, default_parameters)

        # Assert
        assert result_path == output_path
        assert output_path.exists()
        assert output_path.is_file()
        assert output_path.stat().st_size > 0

    def test_output_is_valid_png_image(
        self,
        parsed_al3d_file: ScanImage,
        default_parameters: UploudScanParameters,
        tmp_path: Path,
    ) -> None:
        """Test that the generated file is a valid PNG image that can be opened."""
        # Arrange

        # Act
        surface_map = surface_map_pipeline(parsed_al3d_file, tmp_path / "surfacemap.png", default_parameters)

        # Assert - verify we can open the PNG file
        with Image.open(surface_map) as img:
            assert img.format == "PNG"
            assert img.size == parsed_al3d_file.data.shape

    def test_surface_map_with_multiple_lights(self, parsed_al3d_file: ScanImage, tmp_path: Path) -> None:
        """Test surface map generation with multiple light sources."""
        # Arrange - simulate lighting from 4 cardinal directions
        parameters = UploudScanParameters(  # type: ignore
            light_sources=(
                LightSource(azimuth=0, elevation=45),  # North
                LightSource(azimuth=90, elevation=45),  # East
                LightSource(azimuth=180, elevation=45),  # South
                LightSource(azimuth=270, elevation=45),  # West
            ),
            observer=LightSource(azimuth=0, elevation=90),
        )

        # Act
        surface_map = surface_map_pipeline(parsed_al3d_file, tmp_path / "multi_light_surfacemap.png", parameters)

        # Assert
        with Image.open(surface_map) as img:
            assert img.format == "PNG"
