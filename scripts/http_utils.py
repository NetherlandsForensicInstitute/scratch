import threading
import time
from typing import Any

import requests

API_SEMAPHORE = threading.Semaphore(5)


def _post_with_retry(url: str, body: dict, timeout: int = 300, retries: int = 5) -> dict[str, Any]:
    for attempt in range(retries):
        try:
            with API_SEMAPHORE:
                resp = requests.post(url, json=body, timeout=timeout)
                print(resp.json())
                resp.raise_for_status()
                return resp.json()
        except (requests.ConnectionError, OSError):
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("Unreachable")
