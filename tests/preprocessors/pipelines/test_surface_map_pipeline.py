from pathlib import Path

import pytest
from container_models.light_source import LightSource
from container_models.scan_image import ScanImage
from PIL import Image

from preprocessors.pipelines import surface_map_pipeline


@pytest.fixture(scope="module")
def light_sources() -> tuple[LightSource, LightSource]:
    """Surface map light sources."""
    return (
        LightSource(azimuth=90, elevation=45),
        LightSource(azimuth=180, elevation=45),
    )


@pytest.fixture(scope="module")
def observer() -> LightSource:
    """Observer position looking straight down from +Z direction."""
    return LightSource(azimuth=90, elevation=45)


@pytest.mark.integration
class TestSurfaceMapPipeline:
    """Integration tests for surface_map_pipeline function."""

    def test_generate_surface_map_success(
        self,
        parsed_al3d_file: ScanImage,
        tmp_path: Path,
        light_sources: tuple[LightSource, LightSource],
        observer: LightSource,
    ) -> None:
        """Test that a surface map image is successfully generated from scan data."""
        # Arrange
        output_path = tmp_path / "surface_map.png"

        # Act
        surface_map = surface_map_pipeline(parsed_al3d_file, output_path, light_sources, observer, 1.0, 1.0)

        # Assert
        assert surface_map == output_path
        with Image.open(surface_map) as img:
            assert img.format == "PNG"
            assert img.size == parsed_al3d_file.data.shape

    def test_surface_map_with_multiple_lights(self, parsed_al3d_file: ScanImage, tmp_path: Path) -> None:
        """Test surface map generation with multiple light sources."""
        # Arrange - simulate lighting from 4 cardinal directions
        light_sources = (
            LightSource(azimuth=0, elevation=45),  # North
            LightSource(azimuth=90, elevation=45),  # East
            LightSource(azimuth=180, elevation=45),  # South
            LightSource(azimuth=270, elevation=45),  # West
        )
        observer = LightSource(azimuth=0, elevation=90)

        # Act
        surface_map = surface_map_pipeline(
            parsed_al3d_file, tmp_path / "multi_light_surface_map.png", light_sources, observer, 1.0, 1.0
        )

        # Assert
        with Image.open(surface_map) as img:
            assert img.format == "PNG"
