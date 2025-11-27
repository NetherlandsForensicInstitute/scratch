from pathlib import Path

from conversion import subsample_data
from conversion.display import get_image_for_display, grayscale_to_rgba
from fastapi import APIRouter, status
from image_generation import generate_3d_image
from image_generation.data_formats import ScanMap2D
from parsers import ScanImage, save_to_x3p
from parsers.exceptions import InvalidTypeError
from PIL import Image
from PIL.Image import Image as PILImage
from surfalize.exceptions import CorruptedFileError

from .exceptions import ScanImageException
from .schemas import ProcessedDataLocation, UploadScan


class ParseError(Exception): ...


preprocessor_route = APIRouter(
    prefix="/preprocessor",
    tags=["preprocessor"],
)


@preprocessor_route.get(
    path="/",
    summary="check status of comparison proces",
    description="""Some description of pre-processors endpoint, you can use basic **markup**""",
)
async def comparison_root() -> dict[str, str]:
    """Fetch a simple message from the REST API.

    Here is some more information about the function some notes what is expected.
    Special remarks what the function is doing.

    return: dict[str,str] but, use as much as possible Pydantic for return types
    """
    return {"message": "Hello from the pre-processors"}


def _get_scan_image(scan_file: Path) -> ScanImage:
    try:
        scan_image = ScanImage.from_file(scan_file)
    except InvalidTypeError as err:
        raise ScanImageException("Unable to parse file: invalid type exception") from err
    except CorruptedFileError:
        raise ScanImageException("Unable to open the file.", status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)
    except FileNotFoundError:
        raise ScanImageException(f"The file {scan_file} was not found.", status_code=status.HTTP_404_NOT_FOUND)
    except PermissionError:
        raise ScanImageException(f"Permission denied to access {scan_file}.", status_code=status.HTTP_403_FORBIDDEN)
    return scan_image


def _process_scan_image(scan_image: ScanImage) -> tuple[ScanImage, PILImage, PILImage]:
    try:
        # Subsample the parsed file
        scan_image = subsample_data(scan_image, step_size=(1, 1))
    except ValueError as e:
        raise ScanImageException(f"Error during data subsampling: {str(e)}")

    try:
        # Get image for display
        preview_image = get_image_for_display(scan_image)
    except TypeError as e:
        raise ScanImageException(f"Error processing preview image - TypeError: {str(e)}")
    except ValueError as e:
        raise ScanImageException(f"Error processing preview image - ValueError: {str(e)}")
    except AttributeError as e:
        raise ScanImageException(f"Error processing preview image - AttributeError: {str(e)}")
    except Exception as e:
        raise ScanImageException(f"Unexpected error occurred while processing preview image: {str(e)}")

    try:
        # Create surface map and save to output dir
        surface_plot = generate_3d_image(
            depth_data=ScanMap2D(scan_image.data),
            x_dimension=scan_image.scale_x,
            y_dimension=scan_image.scale_y,
        )
    except TypeError as e:
        raise ScanImageException(f"Error generating surface map - TypeError: {str(e)}")
    except ValueError as e:
        raise ScanImageException(f"Error generating surface map - ValueError: {str(e)}")
    except AttributeError as e:
        raise ScanImageException(f"Error generating surface map - AttributeError: {str(e)}")
    except Exception as e:
        raise ScanImageException(f"Unexpected error occurred while generating surface map: {str(e)}")

    try:
        rgba_data = grayscale_to_rgba(surface_plot.root)
        surface_plot_rgba = Image.fromarray(rgba_data)
    except TypeError as e:
        raise ScanImageException(f"Error converting grayscale data to RGBA: {str(e)}")
    except ValueError as e:
        raise ScanImageException(f"Error in grayscale to RGBA conversion, invalid value: {str(e)}")
    except Exception as e:
        raise ScanImageException(f"Unexpected error in grayscale to RGBA conversion: {str(e)}")

    return scan_image, preview_image, surface_plot_rgba


def _save_outputs(scan_image: ScanImage, preview_image: PILImage, surface_plot_rgba: PILImage, output_dir: Path):
    save_to_x3p(scan_image, output_dir / "measurement.x3p")
    preview_image.save(output_dir / "preview.png")
    surface_plot_rgba.save(output_dir / "surface_map.png")


@preprocessor_route.post(
    path="/process-scan",
    summary="Create surface_map and preview image from the scan file.",
    description="""
    Processes the scan file from the given filepath and generates several derived outputs, including
    an X3P file, a preview image, and a surface map, these files are saved to the output directory given as parameter.
    The endpoint parses and validates the file before running the processing pipeline.
""",
)
async def process_scan(upload_scan: UploadScan) -> ProcessedDataLocation:
    """
    Process an uploaded scan file and generate derived output files.

    This endpoint parses and validates the incoming scan file, performs the
    necessary processing steps, and produces several outputs such as an X3P
    file, a preview image, and a surface map saved to the output directory.
    """
    scan_image = _get_scan_image(upload_scan.scan_file)
    scan_image, preview_image, surface_plot_rgba = _process_scan_image(scan_image)
    _save_outputs(scan_image, preview_image, surface_plot_rgba, upload_scan.output_dir)

    return ProcessedDataLocation(
        x3p_image=upload_scan.output_dir / "measurement.x3p",
        preview_image=upload_scan.output_dir / "preview.png",
        surfacemap_image=upload_scan.output_dir / "surface_map.png",
    )
