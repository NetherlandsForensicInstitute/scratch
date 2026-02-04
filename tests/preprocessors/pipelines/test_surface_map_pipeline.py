from pathlib import Path

import pytest
from container_models import ImageContainer
from PIL import Image

from preprocessors.pipelines import surface_map_pipeline
from preprocessors.schemas import SphericalOrientation


@pytest.fixture(scope="module")
def light_sources() -> tuple[SphericalOrientation, SphericalOrientation]:
    """Surface map light sources."""
    return (
        SphericalOrientation(azimuth=90, elevation=45),
        SphericalOrientation(azimuth=180, elevation=45),
    )


@pytest.fixture(scope="module")
def observer() -> SphericalOrientation:
    """Observer position looking straight down from +Z direction."""
    return SphericalOrientation(azimuth=90, elevation=45)


@pytest.mark.integration
class TestSurfaceMapPipeline:
    """Integration tests for surface_map_pipeline function."""

    def test_generate_surface_map_success(
        self,
        parsed_al3d_file: ImageContainer,
        tmp_path: Path,
        light_sources: tuple[SphericalOrientation, SphericalOrientation],
        observer: SphericalOrientation,
    ) -> None:
        """Test that a surface map image is successfully generated from scan data."""
        # Arrange
        output_path = tmp_path / "surface_map.png"

        # Act
        surface_map = surface_map_pipeline(
            parsed_al3d_file, output_path, (light.unit_vector for light in light_sources), observer.unit_vector
        )

        # Assert
        assert surface_map == output_path
        with Image.open(surface_map) as img:
            assert img.format == "PNG"
            assert img.size == parsed_al3d_file.data.shape

    def test_surface_map_with_multiple_lights(self, parsed_al3d_file: ImageContainer, tmp_path: Path) -> None:
        """Test surface map generation with multiple light sources."""
        # Arrange - simulate lighting from 4 cardinal directions
        light_sources = (
            SphericalOrientation(azimuth=0, elevation=45),  # North
            SphericalOrientation(azimuth=90, elevation=45),  # East
            SphericalOrientation(azimuth=180, elevation=45),  # South
            SphericalOrientation(azimuth=270, elevation=45),  # West
        )
        observer = SphericalOrientation(azimuth=0, elevation=90)

        # Act
        surface_map = surface_map_pipeline(
            parsed_al3d_file,
            tmp_path / "multi_light_surface_map.png",
            (light.unit_vector for light in light_sources),
            observer.unit_vector,
        )

        # Assert
        with Image.open(surface_map) as img:
            assert img.format == "PNG"
