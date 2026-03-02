import datetime
from pathlib import Path

from loguru import logger

from constants import LogLevel


def setup_logging(level: LogLevel = LogLevel.INFO):
    """
    Configure Loguru logging for the backend.

    Adds a file handler in the project root directory with:
    - Retention of the last 5 log files
    - Rotation at 00:00, new is made file every day.
    - File is saved in the root of the project.

    :param level: Log level to use (e.g. "DEBUG", "INFO", "WARNING"), Defaulting to "INFO".
    :return: None
    """
    project_root = Path(__file__).resolve().parent
    log_file = project_root / f"backend-{datetime.date.today()}.log"
    logger.add(
        log_file,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        rotation="00:00",
        retention=5,
        compression="zip",
        encoding="utf-8",
    )
    logger.debug("Logging initialized")
