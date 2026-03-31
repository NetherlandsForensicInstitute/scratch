"""Shared utilities for conversion scripts."""

import json
import logging
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from conversion.data_formats import MarkImpressionType, MarkStriationType, MarkType
from conversion.surface_comparison.models import ComparisonParams
from tqdm import tqdm

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


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


_MARK_TYPE_FOLDER_MAP: list[tuple[str, MarkType]] = sorted(
    ((mt.value.replace(" ", "_"), mt) for cls in (MarkImpressionType, MarkStriationType) for mt in cls),
    key=lambda x: -len(x[0]),
)


def infer_mark_type(folder_name: str) -> MarkType | None:
    """Infer a :class:`MarkType` from a folder name.

    Handles suffixed variants (``_1``, ``_2``) and ``comparison_results`` folders.
    """
    lower = folder_name.lower()
    for fragment, mt in _MARK_TYPE_FOLDER_MAP:
        if fragment in lower:
            return mt
    return None


def _parse_db_scratch(path: Path) -> dict[str, str]:
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


_tool_entries_root_cache: dict[Path, Path | None] = {}


def _get_tool_entries_root(output_dir: Path) -> Path | None:
    """Find (and cache) the parent of the ``tool-entries`` folder under *output_dir*."""
    if output_dir not in _tool_entries_root_cache:
        _tool_entries_root_cache[output_dir] = next(
            (c.parent for c in output_dir.rglob("tool-entries") if c.is_dir()), None
        )
    return _tool_entries_root_cache[output_dir]


def _resolve_mark_dir(relative_path: str, output_dir: Path) -> Path:
    """Map a MATLAB-relative mark path to the converted output directory."""
    parts = relative_path.replace("\\", "/").strip("/").split("/")
    try:
        suffix = "/".join(parts[parts.index("tool-entries") :])
    except ValueError:
        suffix = "/".join(parts)
    te_root = _get_tool_entries_root(output_dir)
    return (te_root / suffix) if te_root else (output_dir / suffix)


def _firearm_dir(mark_dir: Path) -> Path:
    """Return the firearm directory (first child of ``tool-entries``)."""
    parts = mark_dir.parts
    try:
        return Path(*parts[: parts.index("tool-entries") + 2])
    except ValueError:
        return mark_dir.parent.parent.parent


def _extract_metadata(mark_dir: Path) -> dict[str, str]:
    """Extract MarkMetadata by walking up from *mark_dir* to ``tool-entries``.

    :param mark_dir: path to the mark directory.
    :returns: dict with case_id, firearm_id, specimen_id, measurement_id, mark_id.
    """
    parts = mark_dir.parts
    try:
        te = parts.index("tool-entries")
    except ValueError:
        return {k: "unknown" for k in ("case_id", "firearm_id", "specimen_id", "measurement_id", "mark_id")}

    def _name(idx: int) -> str:
        p = Path(*parts[: idx + 1]) if idx < len(parts) else mark_dir
        return _parse_db_scratch(p / "db.scratch").get("NAME", p.name)

    return {
        "case_id": _name(te - 1) if te > 0 else "unknown",
        "firearm_id": _name(te + 1),
        "specimen_id": _name(te + 2),
        "measurement_id": _name(te + 3),
        "mark_id": _name(te + 4) if len(parts) > te + 4 else mark_dir.name,
    }


def find_comparison_results(root: Path) -> Iterator[tuple[Path, MarkType]]:
    """Yield ``(results_folder, mark_type)`` for each ``results_table.mat`` found."""
    for mat in root.rglob("mark-comparison-results/*/results_table.mat"):
        mt = infer_mark_type(mat.parent.name)
        if mt is None:
            logger.warning("Cannot infer mark type from %s, skipping", mat.parent.name)
            continue
        yield mat.parent, mt


@dataclass
class ComparisonEntry:
    """A single comparison pair with pre-resolved paths."""

    mark_dir_ref: Path
    mark_dir_comp: Path
    mark_type: MarkType
    comparison_out: Path
    row_index: int


def _build_body(entry: ComparisonEntry) -> dict[str, Any]:
    """Build the API request body for a comparison."""
    processed_ref = str(entry.mark_dir_ref)
    processed_comp = str(entry.mark_dir_comp)

    if isinstance(entry.mark_type, MarkStriationType):
        return {
            "mark_dir_ref": processed_ref,
            "mark_dir_comp": processed_comp,
            "metadata_reference": _extract_metadata(entry.mark_dir_ref),
            "metadata_compared": _extract_metadata(entry.mark_dir_comp),
        }
    return {
        "mark_dir_ref": processed_ref,
        "mark_dir_comp": processed_comp,
        "metadata_reference": _extract_metadata(entry.mark_dir_ref),
        "metadata_compared": _extract_metadata(entry.mark_dir_comp),
        "comparison_params": ComparisonParams.for_mark_type(entry.mark_type).model_dump(),
    }


def _save_result(entry: ComparisonEntry, result: dict[str, Any] | None = None, error: str | None = None) -> None:
    """Write comparison_results.json with full context."""
    entry.comparison_out.mkdir(parents=True, exist_ok=True)
    output = {
        "mark_dir_ref": str(entry.mark_dir_ref),
        "mark_dir_comp": str(entry.mark_dir_comp),
        "mark_type": entry.mark_type.value,
        "metadata": {
            "metadata_reference": _extract_metadata(entry.mark_dir_ref),
            "metadata_compared": _extract_metadata(entry.mark_dir_comp),
        },
        "error": error,
        "comparison_results": result.get("comparison_results") if result else None,
    }
    (entry.comparison_out / "comparison_results.json").write_text(json.dumps(output, indent=2, default=str))
