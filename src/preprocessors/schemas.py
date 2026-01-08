from enum import StrEnum, auto

from container_models.light_source import LightSource
from pydantic import Field, FilePath, field_validator

from models import BaseModelConfig, validate_file_extension


class SupportedExtension(StrEnum):
    AL3D = auto()
    X3P = auto()
    SUR = auto()
    PLU = auto()


class UploadScanParameters(BaseModelConfig):
    """Configuration parameters for upload scan's surface rendering process."""

    light_sources: tuple[LightSource, ...] = Field(
        (
            LightSource(azimuth=90, elevation=45),
            LightSource(azimuth=180, elevation=45),
        ),
        description="Light sources for surface illumination rendering.",
    )
    observer: LightSource = Field(
        LightSource(azimuth=90, elevation=45),
        description="Observer viewpoint vector for surface rendering.",
    )
    scale_x: float = Field(1.0, gt=0.0, description="pixel size in meters (m)")
    scale_y: float = Field(1.0, gt=0.0, description="pixel size in meters (m)")
    step_size_x: int = Field(1, gt=0)
    step_size_y: int = Field(1, gt=0)

    def as_dict(self, *, exclude: set[str] | None = None, include: set[str] | None = None) -> dict:
        """
        Get model fields as dict with nested models intact (not serialized).

        :param exclude: Set of field names to exclude
        :param include: Set of field names to include (mutually exclusive with exclude)
        """
        if exclude and include:
            raise ValueError("Cannot specify both 'exclude' and 'include'")

        fields = set(self.__class__.model_fields)

        if include:
            fields = include
        elif exclude:
            fields = fields - exclude

        return {field: getattr(self, field) for field in fields}


class UploadScan(BaseModelConfig):
    scan_file: FilePath = Field(
        ...,
        description=f"Path to the input scan file. Supported formats: {', '.join(SupportedExtension)}",
    )
    parameters: UploadScanParameters = Field(
        default_factory=UploadScanParameters.model_construct,
    )

    @property
    def name(self) -> str:
        return self.scan_file.stem

    @property
    def surface_map_filename(self) -> str:
        return f"{self.name}_surface_map.png"

    @property
    def preview_filename(self) -> str:
        return f"{self.name}_preview.png"

    @property
    def x3p_filename(self) -> str:
        return f"{self.name}.x3p"

    @field_validator("scan_file", mode="after")
    @classmethod
    def validate_file_extension(cls, scan_file: FilePath) -> FilePath:
        """Validate given file is of a supported type and not empty."""
        validate_file_extension(str(scan_file), SupportedExtension)

        if scan_file.stat().st_size == 0:
            raise ValueError(f"file is empty: {scan_file.name}")

        return scan_file
