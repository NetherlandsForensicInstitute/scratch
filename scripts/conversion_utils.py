import logging
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


@dataclass
class ConversionConfig:
    """Shared configuration for conversion pipelines."""

    root: Path
    output_dir: Path
    api_url: str
    force: bool = False
    skip_plots: bool = False

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


def parse_db_scratch(path: Path) -> dict[str, str]:
    """Parse a Java-properties-style db.scratch file.

    :param path: path to the db.scratch file.
    :returns: dict of key-value pairs (empty if file missing).
    """
    if not path.exists():
        return {}
    props: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()  # noqa: PLW2901
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            props[key.strip()] = value.strip().replace("\\:", ":")
    return props


def save_shape(path: Path, shape: tuple[int, int]) -> None:
    """Persist pixel dimensions as a sidecar .shape file."""
    path.with_suffix(".shape").write_text(f"{shape[0]},{shape[1]}")


def load_shape(path: Path) -> tuple[int, int] | None:
    """Read pixel dimensions from a sidecar .shape file, if it exists."""
    shape_file = path.with_suffix(".shape")
    if shape_file.exists():
        x, y = shape_file.read_text().strip().split(",")
        return int(x), int(y)
    return None


def find_mark_folders(root: Path) -> Iterator[tuple[Path, Path]]:
    """Yield (measurement_folder, mark_folder) pairs found under *root*."""
    for mark_mat in root.rglob("mark.mat"):
        mf = mark_mat.parent
        if (mf.parent / "measurement.x3p").exists():
            yield mf.parent, mf


def copy_db_scratch_files(root: Path, output_dir: Path) -> int:
    """Copy all db.scratch files to *output_dir*, appending a conversion timestamp."""
    db_files = list(root.rglob("db.scratch"))
    for db_file in tqdm(db_files, desc="Copying db.scratch", unit=" files"):
        out = output_dir / db_file.relative_to(root)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(db_file.read_text() + f"\nCONVERTED_DATE={datetime.now().isoformat()}\n")
    return len(db_files)


def flatten_processed_folders(output_dir: Path) -> None:
    """Move db.scratch from processed/ subfolders and remove the subfolder."""
    for processed_dir in output_dir.rglob("processed"):
        if not processed_dir.is_dir():
            continue
        parent = processed_dir.parent
        contents = list(processed_dir.iterdir())
        unexpected = [f.name for f in contents if f.name != "db.scratch"]
        if unexpected:
            raise RuntimeError(f"Unexpected files in {processed_dir}: {unexpected}")
        for f in contents:
            f.rename(parent / "db_processed.scratch")
        processed_dir.rmdir()
