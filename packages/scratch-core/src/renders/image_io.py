from pathlib import Path

from returns.io import impure_safe
from x3p import X3Pfile

from container_models import ImageContainer
from utils.logger import log_railway_function


@log_railway_function("Failed to save image")
@impure_safe
def save_image(image: ImageContainer, output_path: Path) -> Path:
    image.pil.save(output_path)
    return output_path


@log_railway_function(
    "Failed to write X3P file",
    "Successfully written X3P",
)
@impure_safe
def save_x3p(x3p: X3Pfile, output_path: Path) -> Path:
    """
    Save an X3P file to disk.

    :param x3p: The X3P file to save.
    :param output_path: The path where the file should be written.
    :returns: An ``IOResult[Path, Exception]`` — ``IOSuccess(Path)`` on success,
              or ``IOFailure(Exception)`` if an error occurs.
    """

    x3p.write(str(output_path))
    return output_path
