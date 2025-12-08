from pathlib import Path

from fastapi.exceptions import HTTPException
from image_generation.data_formats import ScanImage
from image_generation.image_generation import ImageGenerator
from loguru import logger


def export_image_pipeline(file_path: Path, image_generator: ImageGenerator, scan_image: ScanImage) -> None:
    """Given an image generator and a scan image, export the image to the specified file path."""
    try:
        generated_image = image_generator(scan_image)
    except ValueError as err:
        logger.error(f"Image generation failed with:{file_path.stem} from error:{str(err)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate {file_path.stem}: {str(err)}")
    generated_image.image().save(file_path)
