"""Shared utilities for conversion scripts."""

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm


@dataclass
class ConversionConfig:
    """Shared configuration for conversion pipelines."""

    root: Path
    output_dir: Path
    api_url: str
    force: bool = False

    def __post_init__(self) -> None:
        self.api_url = self.api_url.rstrip("/")
        self.root = self.root.resolve()
        self.output_dir = self.output_dir.resolve()


def run_parallel(
    tasks: Iterable[tuple[Any, Callable, tuple]],
    workers: int,
    desc: str,
    unit: str,
) -> dict[Any, Any]:
    """Run tasks in parallel with a progress bar.

    :param tasks: iterable of ``(key, fn, args)`` tuples.
    :param workers: number of parallel workers.
    :param desc: progress bar description.
    :param unit: progress bar unit label.
    :returns: dict of ``{key: result}``.
    """
    task_list = list(tasks)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, *args): key for key, fn, args in task_list}
        results: dict[Any, Any] = {}
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc, unit=unit):
            results[futures[future]] = future.result()
    return results
