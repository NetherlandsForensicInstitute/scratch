"""
Extract comparisons from mark-comparison-results and calculate scores via API.

Reads results_table.mat files from the mark-comparison-results folder structure,
maps each comparison pair back to already-converted mark directories, and calls
the appropriate calculate-score endpoint.

Can be used standalone or integrated into convert_matlab_results.py.

Usage:
    python convert_scores.py /path/to/root output/
    python convert_scores.py /path/to/root output/ --api-url http://localhost:8000
"""

import argparse
import json
import logging
import random
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import requests
import scipy.io as sio
from conversion.data_formats import MarkType
from tqdm import tqdm

from scripts.http_utils import _post_with_retry

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# Mapping from MarkType enum values to folder name fragments.
_MARK_TYPE_FOLDER_MAP: dict[str, MarkType] = {mt.value.replace(" ", "_"): mt for mt in MarkType}


@dataclass
class ScoreConversionConfig:
    """Configuration for the score conversion pipeline."""

    root: Path
    output_dir: Path
    api_url: str
    force: bool = False

    def __post_init__(self) -> None:
        self.api_url = self.api_url.rstrip("/")
        self.root = self.root.resolve()
        self.output_dir = self.output_dir.resolve()


@dataclass
class ComparisonEntry:
    """A single comparison pair, either from results_table.mat or generated."""

    mark_dir_ref: Path
    mark_dir_comp: Path
    mark_type: MarkType
    comparison_out: Path
    row_index: int


def _scalar(value: Any) -> Any:
    """Unwrap numpy 0-d or single-element arrays."""
    import numpy as np

    while isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        if value.ndim == 0:
            value = value.item()
        elif value.size == 1:
            value = value.flat[0]
        else:
            break
    return value


def _parse_db_scratch(path: Path) -> dict[str, str]:
    """Parse a db.scratch properties file into a dict.

    :param path: path to the db.scratch file.
    :returns: dict of key-value pairs.
    """
    props: dict[str, str] = {}
    if not path.exists():
        return props
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            # Unescape Java properties-style backslash-colon
            props[key.strip()] = value.strip().replace("\\:", ":")
    return props


def _extract_metadata_from_path(mark_dir: Path) -> dict[str, str]:
    """Walk up the folder hierarchy to extract MarkMetadata from db.scratch files.

    Finds ``tool-entries`` in the path and uses the structure below it::

        tool-entries / <firearm> / <specimen> / <measurement> / <mark>

    The case folder is the parent of ``tool-entries``.

    :param mark_dir: path to the mark directory.
    :returns: dict with case_id, firearm_id, specimen_id, measurement_id, mark_id.
    """
    parts = mark_dir.parts
    try:
        te_idx = parts.index("tool-entries")
    except ValueError:
        logger.warning("No tool-entries in path %s, using folder names as fallback", mark_dir)
        return {
            "case_id": "unknown",
            "firearm_id": "unknown",
            "specimen_id": "unknown",
            "measurement_id": mark_dir.parent.name,
            "mark_id": mark_dir.name,
        }

    # Reconstruct paths from the tool-entries index
    # te_idx+1 = firearm, te_idx+2 = specimen, te_idx+3 = measurement, te_idx+4 = mark
    case_dir = Path(*parts[:te_idx]) if te_idx > 0 else mark_dir.parent
    firearm_dir = Path(*parts[: te_idx + 2]) if len(parts) > te_idx + 1 else mark_dir
    specimen_dir = Path(*parts[: te_idx + 3]) if len(parts) > te_idx + 2 else mark_dir
    measurement_dir = Path(*parts[: te_idx + 4]) if len(parts) > te_idx + 3 else mark_dir

    case_id = _parse_db_scratch(case_dir / "db.scratch").get("NAME", case_dir.name)
    firearm_id = _parse_db_scratch(firearm_dir / "db.scratch").get("NAME", firearm_dir.name)
    specimen_id = _parse_db_scratch(specimen_dir / "db.scratch").get("NAME", specimen_dir.name)
    measurement_id = _parse_db_scratch(measurement_dir / "db.scratch").get("NAME", measurement_dir.name)
    mark_id = _parse_db_scratch(mark_dir / "db.scratch").get("NAME", mark_dir.name)

    return {
        "case_id": case_id,
        "firearm_id": firearm_id,
        "specimen_id": specimen_id,
        "measurement_id": measurement_id,
        "mark_id": mark_id,
    }


def infer_mark_category(folder_name: str) -> MarkType | None:
    """Infer a MarkType from a mark or comparison-results folder name.

    Handles suffixed variants like ``firing_pin_impression_mark_1`` and
    comparison-results folders like ``firing_pin_impression_mark_comparison_results``.

    :param folder_name: e.g. 'firing_pin_impression_mark_comparison_results'
        or 'aperture_shear_striation_mark_2'.
    :returns: matching MarkType, or None if unrecognised.
    """
    lower = folder_name.lower()
    # Try longest match first to avoid partial matches
    for folder_fragment, mark_type in sorted(_MARK_TYPE_FOLDER_MAP.items(), key=lambda x: -len(x[0])):
        if folder_fragment in lower:
            return mark_type
    return None


def _mark_type_base(folder_name: str) -> str:
    """Strip trailing numeric suffixes like ``_1``, ``_2`` from a mark type folder name.

    :param folder_name: e.g. 'firing_pin_impression_mark_2'.
    :returns: base name, e.g. 'firing_pin_impression_mark'.
    """
    import re

    return re.sub(r"_\d+$", "", folder_name)


def verify_mark_category(mark_dir: Path, expected: MarkType) -> bool:
    """Verify the inferred mark type against the mark.json in the converted output.

    :param mark_dir: path to the converted mark directory (should contain mark.json).
    :param expected: the expected MarkType.
    :returns: True if verified or mark.json is missing (graceful fallback).
    """
    mark_json = mark_dir / "mark.json"
    if not mark_json.exists():
        logger.warning("No mark.json at %s, skipping verification", mark_dir)
        return True

    try:
        meta = json.loads(mark_json.read_text())
        raw = meta["mark_type"]
        # mark.json may store the enum name (e.g. "APERTURE_SHEAR_STRIATION")
        # or the value (e.g. "aperture shear striation mark")
        try:
            actual = MarkType(raw)
        except ValueError:
            actual = MarkType[raw]

        # Check that both are the same category (impression vs striation)
        if expected.is_impression():
            return actual.is_impression()
        return actual.is_striation()
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("Could not parse mark.json at %s: %s", mark_dir, exc)

    return True  # graceful fallback


def find_comparison_results(root: Path) -> Iterator[tuple[Path, MarkType]]:
    """Yield (results_folder, mark_type) for each results_table.mat found.

    Searches for any ``mark-comparison-results`` folder anywhere under root.

    :param root: the database root.
    """
    for results_mat in root.rglob("mark-comparison-results/*/results_table.mat"):
        folder = results_mat.parent
        mark_type = infer_mark_category(folder.name)
        if mark_type is None:
            logger.warning("Cannot infer mark type from folder %s, skipping", folder.name)
            continue
        yield folder, mark_type


def extract_comparisons(
    results_folder: Path, mark_type: MarkType, root: Path, output_dir: Path
) -> list[ComparisonEntry]:
    """Extract all comparison pairs from a results_table.mat file.

    Handles two possible layouts from loadmat:
    - ``(1, N)`` struct array: each element is one comparison with scalar fields.
    - ``(1, 1)`` struct: each field is an ``(N, 1)`` array of values.

    :param results_folder: folder containing results_table.mat.
    :param mark_type: the MarkType for these comparisons.
    :param root: the database root.
    :param output_dir: the converted output directory.
    :returns: list of ComparisonEntry objects.
    """
    mat_path = results_folder / "results_table.mat"
    data = sio.loadmat(str(mat_path), squeeze_me=False)
    rt = data["results_table"]

    comparison_out_base = output_dir / results_folder.relative_to(root)

    def _make_entry(ref_str: str, comp_str: str, idx: int) -> ComparisonEntry:
        return ComparisonEntry(
            mark_dir_ref=_resolve_mark_dir(ref_str, output_dir),
            mark_dir_comp=_resolve_mark_dir(comp_str, output_dir),
            mark_type=mark_type,
            comparison_out=comparison_out_base / f"{idx:04d}",
            row_index=idx,
        )

    entries = []

    # Determine layout: check if pathReference in the first element is a
    # single value or an array of all comparisons.
    first = rt[0, 0]
    path_ref_field = first["pathReference"]

    if path_ref_field.ndim >= 1 and path_ref_field.shape[0] > 1:
        # Layout: (1,1) struct with (N,1) arrays per field
        path_comps_field = first["pathCompare"]
        n_rows = path_ref_field.shape[0]

        for i in range(n_rows):
            ref = _unwrap_path(path_ref_field[i, 0] if path_ref_field.ndim > 1 else path_ref_field[i])
            comp = _unwrap_path(path_comps_field[i, 0] if path_comps_field.ndim > 1 else path_comps_field[i])

            if ref is None or comp is None:
                logger.warning("Row %d in %s has missing paths, skipping", i, mat_path)
                continue

            entries.append(_make_entry(ref, comp, i))
    else:
        # Layout: (1,N) struct array, one comparison per element
        n_rows = rt.shape[1] if rt.ndim >= 2 else 1

        for i in range(n_rows):
            row = rt[0, i] if rt.ndim >= 2 else rt[i]
            ref = _unwrap_path(row["pathReference"])
            comp = _unwrap_path(row["pathCompare"])

            if ref is None or comp is None:
                logger.warning("Row %d in %s has missing paths, skipping", i, mat_path)
                continue

            entries.append(_make_entry(ref, comp, i))

    return entries


def _unwrap_path(value: Any) -> str | None:
    """Recursively unwrap a MATLAB path value to a plain string.

    MATLAB paths may arrive as nested arrays, e.g. ``array([array(['tool-entries/...'], dtype='<U64')])``.

    :param value: raw value from loadmat.
    :returns: plain string or None.
    """
    import numpy as np

    while isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.flat[0]
    return str(value) if value is not None else None


def _find_tool_entries_root(output_dir: Path) -> Path | None:
    """Find the parent of the tool-entries folder under output_dir.

    :param output_dir: the converted output directory.
    :returns: the parent directory of tool-entries, or None if not found.
    """
    for candidate in output_dir.rglob("tool-entries"):
        if candidate.is_dir():
            return candidate.parent
    return None


# Module-level cache to avoid repeated rglob calls.
_tool_entries_root_cache: dict[Path, Path | None] = {}


def _get_tool_entries_root(output_dir: Path) -> Path | None:
    """Cached version of _find_tool_entries_root."""
    if output_dir not in _tool_entries_root_cache:
        _tool_entries_root_cache[output_dir] = _find_tool_entries_root(output_dir)
    return _tool_entries_root_cache[output_dir]


def _resolve_mark_dir(relative_path: str, output_dir: Path) -> Path:
    """Resolve a mark path from results_table to the converted output directory.

    Paths in results_table.mat are relative and may start with the database
    folder name or directly with ``tool-entries``. This function finds
    ``tool-entries`` in the relative path and resolves it against the output.

    :param relative_path: path as stored in results_table.mat.
    :param output_dir: base output directory (the converted database root).
    :returns: resolved Path to the mark directory in the output.
    """
    cleaned = relative_path.replace("\\", "/").strip("/")
    parts = cleaned.split("/")

    # Find tool-entries in the relative path and take everything from there
    try:
        te_idx = parts.index("tool-entries")
        from_tool_entries = "/".join(parts[te_idx:])
    except ValueError:
        from_tool_entries = cleaned

    # Resolve against the cached tool-entries root in the output
    te_root = _get_tool_entries_root(output_dir)
    if te_root is not None:
        return te_root / from_tool_entries

    return output_dir / from_tool_entries


def _build_metadata_params(mark_dir_ref: Path, mark_dir_comp: Path) -> dict[str, Any]:
    """Build the MetadataParameters dict from db.scratch files.

    :param mark_dir_ref: path to the reference mark directory in the output.
    :param mark_dir_comp: path to the compared mark directory in the output.
    :returns: dict suitable for the 'param' field of the score request.
    """
    return {
        "metadata_reference": _extract_metadata_from_path(mark_dir_ref),
        "metadata_compared": _extract_metadata_from_path(mark_dir_comp),
    }


# ---------------------------------------------------------------------------
# Pair generation (alternative to results_table.mat)
# ---------------------------------------------------------------------------


def _firearm_dir_for_mark(mark_dir: Path) -> Path:
    """Return the firearm directory for a mark.

    Finds ``tool-entries`` in the path and returns the first directory below it.
    Structure: .../tool-entries/<firearm>/<specimen>/<measurement>/<mark>.

    :param mark_dir: path to the mark directory.
    :returns: the firearm directory.
    """
    parts = mark_dir.parts
    try:
        te_idx = parts.index("tool-entries")
        return Path(*parts[: te_idx + 2])
    except ValueError:
        # Fallback: assume 3 levels up
        return mark_dir.parent.parent.parent


def find_all_mark_types(output_dir: Path) -> list[MarkType]:
    """Find all distinct MarkTypes that have processed marks in the output.

    Folders like ``firing_pin_impression_mark`` and ``firing_pin_impression_mark_1``
    map to the same MarkType. Only searches under ``tool-entries`` directories.

    :param output_dir: the converted output directory.
    :returns: sorted list of unique MarkTypes found.
    """
    mark_types: set[MarkType] = set()
    for tool_entries in output_dir.rglob("tool-entries"):
        if not tool_entries.is_dir():
            continue
        for processed_dir in tool_entries.rglob("processed"):
            if processed_dir.is_dir():
                mark_dir = processed_dir.parent
                mt = infer_mark_category(mark_dir.name)
                if mt is not None:
                    mark_types.add(mt)
    return sorted(mark_types, key=lambda mt: mt.value)


def find_marks_by_type(output_dir: Path, mark_type: MarkType) -> list[Path]:
    """Find all processed mark directories matching a MarkType.

    Matches folders whose name infers to the given MarkType (including
    suffixed variants like ``firing_pin_impression_mark_1``).
    Only searches under ``tool-entries`` directories.

    :param output_dir: the converted output directory.
    :param mark_type: the MarkType to search for.
    :returns: list of mark directory paths that have a processed/ subdirectory.
    """
    marks = []
    for tool_entries in output_dir.rglob("tool-entries"):
        if not tool_entries.is_dir():
            continue
        for processed_dir in tool_entries.rglob("processed"):
            if processed_dir.is_dir():
                mark_dir = processed_dir.parent
                if infer_mark_category(mark_dir.name) == mark_type:
                    marks.append(mark_dir)
    return sorted(marks)


def generate_pairs(
    output_dir: Path,
    mark_type: MarkType,
    seed: int | None = None,
) -> list[ComparisonEntry]:
    """Generate same-source and different-source comparison pairs.

    Same-source: all combinations of marks from the same firearm.
    Different-source: random pairs from different firearms, equal in count
    to the number of same-source pairs.

    :param output_dir: the converted output directory.
    :param mark_type: the MarkType to generate pairs for.
    :param seed: random seed for reproducible different-source sampling.
    :returns: list of ComparisonEntry objects.
    """
    marks = find_marks_by_type(output_dir, mark_type)
    if not marks:
        logger.warning("No processed marks found for type '%s'", mark_type.value)
        return []

    # Group by firearm
    by_firearm: dict[Path, list[Path]] = defaultdict(list)
    for mark in marks:
        by_firearm[_firearm_dir_for_mark(mark)].append(mark)

    logger.info(
        "Found %d marks across %d firearms for '%s'",
        len(marks),
        len(by_firearm),
        mark_type.value,
    )

    comparison_out_base = output_dir / "generated-comparison-results" / mark_type.value.replace(" ", "_")

    entries: list[ComparisonEntry] = []
    idx = 0

    # Same-source: all combinations within each firearm
    same_source_pairs: list[tuple[Path, Path]] = []
    for firearm_marks in by_firearm.values():
        for a, b in combinations(firearm_marks, 2):
            same_source_pairs.append((a, b))

    for a, b in same_source_pairs:
        entries.append(
            ComparisonEntry(
                mark_dir_ref=a,
                mark_dir_comp=b,
                mark_type=mark_type,
                comparison_out=comparison_out_base / f"{idx:04d}",
                row_index=idx,
            )
        )
        idx += 1

    n_same = len(same_source_pairs)
    logger.info("Generated %d same-source pairs", n_same)

    # Different-source: random pairs from different firearms
    firearm_keys = list(by_firearm.keys())
    if len(firearm_keys) < 2:
        logger.warning("Only one firearm found, cannot generate different-source pairs")
        return entries

    # Build pool of all possible different-source pairs
    diff_pool: list[tuple[Path, Path]] = []
    for i, fa in enumerate(firearm_keys):
        for fb in firearm_keys[i + 1 :]:
            for ma in by_firearm[fa]:
                for mb in by_firearm[fb]:
                    diff_pool.append((ma, mb))

    rng = random.Random(seed)
    n_diff = min(n_same, len(diff_pool))
    diff_pairs = rng.sample(diff_pool, n_diff)

    for a, b in diff_pairs:
        entries.append(
            ComparisonEntry(
                mark_dir_ref=a,
                mark_dir_comp=b,
                mark_type=mark_type,
                comparison_out=comparison_out_base / f"{idx:04d}",
                row_index=idx,
            )
        )
        idx += 1

    logger.info("Generated %d different-source pairs", n_diff)

    return entries


def calculate_score(
    entry: ComparisonEntry,
    cfg: ScoreConversionConfig,
) -> dict[str, Any] | None:
    """Call the score calculation endpoint for a single comparison pair.

    :param entry: the comparison to process.
    :param cfg: pipeline configuration.
    :returns: API response dict, or None if skipped.
    """
    processed_ref = entry.mark_dir_ref / "processed"
    processed_comp = entry.mark_dir_comp / "processed"

    if (entry.comparison_out / "comparison_results.json").exists() and not cfg.force:
        logger.debug("Skipping already-processed comparison %s", entry.comparison_out)
        return None

    if not processed_ref.exists() or not processed_comp.exists():
        logger.warning("Processed dir missing for row %d", entry.row_index)
        return None

    # Verify mark type against mark.json
    if not verify_mark_category(entry.mark_dir_ref, entry.mark_type):
        logger.warning(
            "Mark type mismatch for reference %s: expected %s (row %d)",
            entry.mark_dir_ref,
            entry.mark_type.value,
            entry.row_index,
        )
        return None

    category = "impression" if entry.mark_type.is_impression() else "striation"
    endpoint = f"processor/calculate-score-{category}"

    if entry.mark_type.is_striation():
        body: dict[str, Any] = {
            "mark_ref": str(processed_ref),
            "mark_comp": str(processed_comp),
            "param": _build_metadata_params(entry.mark_dir_ref, entry.mark_dir_comp),
        }
    else:
        # TODO: update with actual CalculateScoreImpression fields
        body = {
            "mark_ref": str(processed_ref),
            "mark_comp": str(processed_comp),
        }

    result = _post_with_retry(f"{cfg.api_url}/{endpoint}", body)

    # Save results
    entry.comparison_out.mkdir(parents=True, exist_ok=True)

    if "comparison_results" in result:
        (entry.comparison_out / "comparison_results.json").write_text(
            json.dumps(result["comparison_results"], indent=2, default=str)
        )

    # Download all URL-based result files (plots, aligned marks, etc.)
    urls = result.get("urls", result)
    if isinstance(urls, dict):
        for key, url in urls.items():
            if not isinstance(url, str) or not url.startswith("http"):
                continue
            filename = url.rsplit("/", 1)[-1]
            try:
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                (entry.comparison_out / filename).write_bytes(resp.content)
            except requests.RequestException as exc:
                logger.warning("Failed to download %s for %s: %s", key, entry.comparison_out, exc)

    return result


def _run_parallel_scores(
    entries: list[ComparisonEntry],
    cfg: ScoreConversionConfig,
    workers: int,
) -> dict[int, Any]:
    """Run score calculations in parallel with a progress bar."""
    tasks = [(e.row_index, calculate_score, (e, cfg)) for e in entries]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, *args): key for key, fn, args in tasks}
        results = {}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating scores", unit=" comparisons"):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception:
                logger.exception("Failed to calculate score for row %d", key)
                results[key] = None

    return results


def run_score_conversion(
    cfg: ScoreConversionConfig,
    workers: int = 1,
    limit: int | None = None,
    generate: bool = False,
    seed: int | None = None,
) -> None:
    """Run the full score conversion pipeline.

    :param cfg: pipeline configuration.
    :param workers: number of parallel workers.
    :param limit: max comparisons to process (for debugging).
    :param generate: if True, generate pairs from all mark types found in the
        output directory instead of reading from results_table.mat.
    :param seed: random seed for different-source pair sampling.
    """
    if generate:
        all_entries: list[ComparisonEntry] = []
        mark_types = find_all_mark_types(cfg.output_dir)
        logger.info("Found mark types: %s", mark_types)
        for mt in mark_types:
            entries = generate_pairs(cfg.output_dir, mt, seed=seed)
            logger.info("Generated %d pairs for '%s'", len(entries), mt.value)
            all_entries.extend(entries)
    else:
        all_entries = []
        for results_folder, mark_type in tqdm(
            find_comparison_results(cfg.root), desc="Scanning comparison results", unit=" folders"
        ):
            entries = extract_comparisons(results_folder, mark_type, cfg.root, cfg.output_dir)
            logger.info("Found %d comparisons in %s (%s)", len(entries), results_folder.name, mark_type.value)
            all_entries.extend(entries)

    logger.info("Total comparisons: %d", len(all_entries))

    if limit:
        all_entries = all_entries[:limit]

    if not all_entries:
        logger.warning("No comparisons to process")
        return

    results = _run_parallel_scores(all_entries, cfg, workers)

    processed = sum(1 for v in results.values() if v is not None)
    logger.info("Done: %d/%d comparisons processed", processed, len(all_entries))


def main() -> None:
    """Entry point: parse args and run the score conversion pipeline."""
    parser = argparse.ArgumentParser(description="Calculate scores for MATLAB comparison results via Python API")
    parser.add_argument("root", type=Path, help="Root folder containing case folders")
    parser.add_argument("output", type=Path, help="Output folder (same as used for mark conversion)")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--force", action="store_true", help="Recalculate even if output exists")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N comparisons (for debugging)")
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate same-source and different-source pairs from all mark types "
        "found in the output directory, instead of using results_table.mat.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for different-source pair sampling")
    args = parser.parse_args()

    cfg = ScoreConversionConfig(root=args.root, output_dir=args.output, api_url=args.api_url, force=args.force)
    run_score_conversion(
        cfg,
        workers=args.workers,
        limit=args.limit,
        generate=args.generate,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
