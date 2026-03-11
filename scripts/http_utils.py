import logging
import threading
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)
API_SEMAPHORE = threading.Semaphore(5)


def _post_with_retry(  # noqa: PLR0913
    url: str,
    body: dict | None = None,
    timeout: int = 300,
    retries: int = 5,
    data: dict | None = None,
    files: dict | None = None,
) -> dict[str, Any]:
    for attempt in range(retries):
        try:
            with API_SEMAPHORE:
                if files is not None:
                    # Multipart form upload
                    resp = requests.post(url, data=data, files=files, timeout=timeout)
                else:
                    # JSON body
                    resp = requests.post(url, json=body, timeout=timeout)
                if not resp.ok:
                    logger.error("HTTP %d for %s: %s", resp.status_code, url, resp.text[:500])
                resp.raise_for_status()
                return resp.json()
        except (requests.ConnectionError, OSError):
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("Unreachable")


def download_urls(urls: dict | Any, dest: Path) -> None:
    """Download all HTTP URL values from an API response dict.

    :param urls: dict (or nested dict) whose values may be URL strings.
    :param dest: destination directory for downloaded files.
    """
    if not isinstance(urls, dict):
        return
    for key, url in urls.items():
        if not isinstance(url, str) or not url.startswith("http"):
            continue
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            filename = url.rsplit("/", 1)[-1]
            (dest / filename).write_bytes(resp.content)
        except requests.RequestException as exc:
            logger.warning("Failed to download %s: %s", key, exc)
