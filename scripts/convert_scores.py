"""
Calculate scores for mark comparisons via the Python API.

Two modes of operation:

1. **From results_table.mat** (default): reads comparison pairs from the
   mark-comparison-results folder structure in an existing MATLAB database.

2. **Generated pairs** (``--generate``): discovers all processed marks in
   the output directory, groups them by firearm, and generates all
   same-source pairs plus an equal number of random different-source pairs.

Usage::

    python convert_scores.py /path/to/root output/
    python convert_scores.py /path/to/root output/ --generate --same-source-only
"""

import argparse
import json
import logging
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import requests
import scipy.io as sio
from conversion.data_formats import MarkType
from tqdm import tqdm

from scripts.conversion_utils import ConversionConfig, run_parallel
from scripts.http_utils import _post_with_retry, download_urls
from scripts.matlab_utils import unwrap_path

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_MARK_TYPE_FOLDER_MAP: list[tuple[str, MarkType]] = sorted(
    ((mt.value.replace(" ", "_"), mt) for mt in MarkType),
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


@dataclass
class ComparisonEntry:
    """A single comparison pair with pre-resolved paths."""

    mark_dir_ref: Path
    mark_dir_comp: Path
    mark_type: MarkType
    comparison_out: Path
    row_index: int


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


def find_comparison_results(root: Path) -> Iterator[tuple[Path, MarkType]]:
    """Yield ``(results_folder, mark_type)`` for each ``results_table.mat`` found."""
    for mat in root.rglob("mark-comparison-results/*/results_table.mat"):
        mt = infer_mark_type(mat.parent.name)
        if mt is None:
            logger.warning("Cannot infer mark type from %s, skipping", mat.parent.name)
            continue
        yield mat.parent, mt


def extract_comparisons(
    results_folder: Path, mark_type: MarkType, root: Path, output_dir: Path
) -> list[ComparisonEntry]:
    """Load comparison pairs from a ``results_table.mat``."""
    mat_path = results_folder / "results_table.mat"
    rt = sio.loadmat(str(mat_path), squeeze_me=False)["results_table"][0, 0]
    out_base = output_dir / results_folder.relative_to(root)

    refs = rt["pathReference"]
    comps = rt["pathCompare"]

    if refs.shape[0] <= 1:
        raise ValueError(f"Unexpected results_table layout in {mat_path}: refs shape {refs.shape}")

    return [
        ComparisonEntry(
            _resolve_mark_dir(ref, output_dir),
            _resolve_mark_dir(comp, output_dir),
            mark_type,
            out_base / f"{i:04d}",
            i,
        )
        for i in range(refs.shape[0])
        if (ref := unwrap_path(refs[i, 0] if refs.ndim > 1 else refs[i]))
        and (comp := unwrap_path(comps[i, 0] if comps.ndim > 1 else comps[i]))
    ]


def _find_marks(output_dir: Path, mark_type: MarkType | None = None) -> list[Path]:
    """Find processed mark directories under ``tool-entries`` in *output_dir*."""
    marks = []
    for te in output_dir.rglob("tool-entries"):
        if not te.is_dir():
            continue
        for proc in te.rglob("processed"):
            if not proc.is_dir():
                continue
            md = proc.parent
            mt = infer_mark_type(md.name)
            if mt is not None and (mark_type is None or mt == mark_type):
                marks.append(md)
    return sorted(marks)


def find_all_mark_types(output_dir: Path) -> list[MarkType]:
    """Discover all distinct :class:`MarkType` values present in the output."""
    # _find_marks already filters to known mark types
    types = {infer_mark_type(m.name) for m in _find_marks(output_dir)}
    types.discard(None)
    return sorted(types, key=lambda mt: mt.value)  # type: ignore[union-attr]


def generate_pairs(
    output_dir: Path,
    mark_type: MarkType,
    seed: int | None = None,
) -> list[ComparisonEntry]:
    """Generate same-source (and optionally different-source) comparison pairs."""
    marks = _find_marks(output_dir, mark_type)
    if not marks:
        logger.warning("No processed marks for '%s'", mark_type.value)
        return []

    by_firearm: dict[Path, list[Path]] = defaultdict(list)
    for m in marks:
        by_firearm[_firearm_dir(m)].append(m)

    logger.info("%d marks across %d firearms for '%s'", len(marks), len(by_firearm), mark_type.value)

    out_base = output_dir / "generated-comparison-results" / mark_type.value.replace(" ", "_")
    entries: list[ComparisonEntry] = []
    idx = 0

    for firearm_marks in by_firearm.values():
        for a, b in combinations(firearm_marks, 2):
            entries.append(ComparisonEntry(a, b, mark_type, out_base / f"{idx:04d}", idx))
            idx += 1

    n_same = len(entries)
    logger.info("Generated %d same-source pairs", n_same)

    if len(by_firearm) < 2:  # noqa: PLR2004
        return entries

    keys = list(by_firearm.keys())
    pool = [
        (ma, mb) for i, fa in enumerate(keys) for fb in keys[i + 1 :] for ma in by_firearm[fa] for mb in by_firearm[fb]
    ]
    rng = np.random.default_rng(seed)
    diff_indices = rng.choice(len(pool), size=min(n_same, len(pool)), replace=False)
    for di in diff_indices:
        a, b = pool[di]
        entries.append(ComparisonEntry(a, b, mark_type, out_base / f"{idx:04d}", idx))
        idx += 1

    logger.info("Generated %d different-source pairs", idx - n_same)
    return entries


def _build_body(entry: ComparisonEntry) -> dict[str, Any]:
    """Build the API request body for a comparison."""
    processed_ref = str(entry.mark_dir_ref / "processed")
    processed_comp = str(entry.mark_dir_comp / "processed")

    if entry.mark_type.is_striation():
        return {
            "mark_dir_ref": processed_ref,
            "mark_dir_comp": processed_comp,
            "param": {
                "metadata_reference": _extract_metadata(entry.mark_dir_ref),
                "metadata_compared": _extract_metadata(entry.mark_dir_comp),
            },
        }
    # TODO: update with actual CalculateScoreImpression fields
    return {"mark_dir_ref": processed_ref, "mark_dir_comp": processed_comp}


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


def calculate_score(entry: ComparisonEntry, cfg: ConversionConfig) -> dict[str, Any] | None:
    """Call the score endpoint for a single comparison pair."""
    if (entry.comparison_out / "comparison_results.json").exists() and not cfg.force:
        return None

    processed_ref = entry.mark_dir_ref / "processed"
    processed_comp = entry.mark_dir_comp / "processed"
    if not processed_ref.exists() or not processed_comp.exists():
        logger.warning("Processed dir missing for row %d", entry.row_index)
        return None

    category = "impression" if entry.mark_type.is_impression() else "striation"
    endpoint = f"processor/calculate-score-{category}"

    try:
        result = _post_with_retry(f"{cfg.api_url}/{endpoint}", _build_body(entry))
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 422:  # noqa: PLR2004
            try:
                detail = exc.response.json().get("detail", "unknown")
            except requests.JSONDecodeError:
                detail = exc.response.text[:200]
            logger.info("Row %d failed: %s", entry.row_index, detail)
            _save_result(entry, error=detail)
            return {"error": detail}
        raise

    _save_result(entry, result=result)
    download_urls(result.get("urls", result), entry.comparison_out)
    return result


def run_score_conversion(
    cfg: ConversionConfig,
    workers: int = 1,
    limit: int | None = None,
    use_pairs_from_file: bool = False,
    seed: int | None = None,
) -> None:
    """Run the full score conversion pipeline."""
    all_entries = get_pairs(cfg, use_pairs_from_file, limit, seed)
    if not all_entries:
        logger.warning("No comparisons to process")
        return

    results = run_parallel(
        ((e.row_index, calculate_score, (e, cfg)) for e in all_entries),
        workers,
        "Calculating scores",
        " comparisons",
    )
    processed = sum(1 for v in results.values() if v is not None)
    logger.info("Done: %d/%d comparisons processed", processed, len(all_entries))


def get_pairs(
    cfg: ConversionConfig, use_pairs_from_file: bool, limit: int | None, seed: int | None
) -> list[ComparisonEntry]:
    """
    Collect comparison pairs either from results_table.mat files or by generating them.

    :param cfg: pipeline configuration.
    :param use_pairs_from_file: if False, generate same- and different-source pairs from
        all mark types found in the output directory. Otherwise, read pairs
        from ``results_table.mat`` files under the root.
    :param limit: if set, return at most this many pairs.
    :param seed: random seed for different-source pair sampling (only used
        when *generate* is True).
    :returns: list of comparison entries ready for score calculation.
    """
    all_entries = []
    if not use_pairs_from_file:
        for mt in find_all_mark_types(cfg.output_dir):
            entries = generate_pairs(cfg.output_dir, mt, seed=seed)
            logger.info("Generated %d pairs for '%s'", len(entries), mt.value)
            all_entries.extend(entries)
    else:
        for folder, mt in tqdm(find_comparison_results(cfg.root), desc="Scanning", unit=" folders"):
            entries = extract_comparisons(folder, mt, cfg.root, cfg.output_dir)
            logger.info("Found %d comparisons in %s (%s)", len(entries), folder.name, mt.value)
            all_entries.extend(entries)

    logger.info("Total comparisons: %d", len(all_entries))
    if limit:
        all_entries = all_entries[:limit]

    # Only doing striation for now:
    all_entries = [a for a in all_entries if a.mark_type.is_striation()]
    return all_entries


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Calculate scores for mark comparisons via Python API")
    parser.add_argument("root", type=Path, help="Root database folder")
    parser.add_argument("output", type=Path, help="Output folder (same as used for mark conversion)")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers")
    parser.add_argument("--force", action="store_true", help="Recalculate existing results")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N comparisons")
    parser.add_argument(
        "--use_pairs_from_file", action="store_true", help="Read pairs from results_table.mat instead of generating"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for different-source sampling")
    args = parser.parse_args()

    cfg = ConversionConfig(root=args.root, output_dir=args.output, api_url=args.api_url, force=args.force)
    run_score_conversion(
        cfg,
        workers=args.workers,
        limit=args.limit,
        use_pairs_from_file=args.use_pairs_from_file,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
