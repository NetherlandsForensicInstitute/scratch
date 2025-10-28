import os
from pathlib import Path

from src.parsers import ImageFileParser, ParsedImage, ScanFileParser


class Image:
    IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tif"}
    SCAN_FORMATS = {".al3d", ".x3p", ".sur", ".lms", ".plu"}
    FILE_FORMATS = IMAGE_FORMATS | SCAN_FORMATS

    @classmethod
    def from_file(cls, filepath: os.PathLike | Path) -> ParsedImage:
        """Parse and validate an image file."""
        if not isinstance(filepath, Path):
            filepath = Path(str(filepath))

        if not filepath.exists():
            raise FileNotFoundError(f"Path not found: {filepath}")
        if filepath.is_dir():
            raise FileNotFoundError(f"Path is a directory: {filepath}")

        if (ext := filepath.suffix.lower()) not in cls.FILE_FORMATS:
            raise RuntimeError(f"File extension must be one of: {', '.join(cls.FILE_FORMATS)}")

        parser = ScanFileParser() if ext in cls.SCAN_FORMATS else ImageFileParser()
        return parser.parse(filepath)
