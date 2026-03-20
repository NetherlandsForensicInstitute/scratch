"""
Calculate scores for mark comparisons via the Python API.

Two modes of operation:

1. **From results_table.mat** (default): reads comparison pairs from the
   mark-comparison-results folder structure in an existing MATLAB database.

2. **Generated pairs** (``--generate``): discovers all processed marks in
   the output directory, groups them by firearm, and generates all
   same-source pairs plus an equal number of random different-source pairs.
"""

import argparse
import logging
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import requests
import scipy.io as sio
from conversion.data_formats import MarkType
from tqdm import tqdm

from scripts.conversion_utils import (
    ComparisonEntry,
    ConversionConfig,
    _build_body,
    _firearm_dir,
    _resolve_mark_dir,
    _save_result,
    find_comparison_results,
    infer_mark_type,
    run_parallel,
)
from scripts.http_utils import _post_with_retry, download_urls
from scripts.matlab_utils import unwrap_path

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def extract_comparisons(
    results_folder: Path, mark_type: MarkType, root: Path, output_dir: Path
) -> list[ComparisonEntry]:
    """Load comparison pairs from a ``results_table.mat``."""
    mat_path = results_folder / "results_table.mat"
    rt = sio.loadmat(str(mat_path), squeeze_me=False)["results_table"][0, 0]
    out_base = output_dir / results_folder.relative_to(root)

    refs = rt["pathReference"]
    comps = rt["pathCompare"]

    if refs.shape[0] == 1:
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


def find_marks(output_dir: Path, mark_type: MarkType | None = None) -> list[Path]:
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
    types = {infer_mark_type(m.name) for m in find_marks(output_dir)}
    types.discard(None)
    return sorted(types, key=lambda mt: mt.value)  # type: ignore[union-attr]


def _group_by_firearm(marks: list[Path]) -> dict[Path, list[Path]]:
    """Group mark directories by their firearm directory."""
    by_firearm: dict[Path, list[Path]] = defaultdict(list)
    for m in marks:
        by_firearm[_firearm_dir(m)].append(m)
    return by_firearm


def _same_source_pairs(by_firearm: dict[Path, list[Path]]) -> list[tuple[Path, Path]]:
    """All within-firearm combinations."""
    return [(a, b) for marks in by_firearm.values() for a, b in combinations(marks, 2)]


def _different_source_pairs(
    by_firearm: dict[Path, list[Path]], n: int, seed: int | None = None
) -> list[tuple[Path, Path]]:
    """Random cross-firearm pairs."""
    keys = list(by_firearm.keys())
    pool = [
        (ma, mb) for i, fa in enumerate(keys) for fb in keys[i + 1 :] for ma in by_firearm[fa] for mb in by_firearm[fb]
    ]
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return [pool[i] for i in indices]


def generate_pairs(
    output_dir: Path,
    mark_type: MarkType,
    seed: int | None = None,
    same_source_only: bool = False,
) -> list[ComparisonEntry]:
    """Generate comparison pairs for a mark type."""
    marks = find_marks(output_dir, mark_type)
    if not marks:
        logger.warning("No processed marks for '%s'", mark_type.value)
        return []

    by_firearm = _group_by_firearm(marks)
    logger.info(f"{len(marks)} marks across {len(by_firearm)} firearms for '{mark_type.value}'")

    out_base = output_dir / "generated-comparison-results" / mark_type.value.replace(" ", "_")

    pairs = _same_source_pairs(by_firearm)
    logger.info(f"Generated {len(pairs)} same-source pairs")

    if not same_source_only and len(by_firearm) >= 2:  # noqa: PLR2004
        diff = _different_source_pairs(by_firearm, n=len(pairs), seed=seed)
        logger.info(f"Generated {len(diff)} different-source pairs")
        pairs.extend(diff)

    return [ComparisonEntry(a, b, mark_type, out_base / f"{i:04d}", i) for i, (a, b) in enumerate(pairs)]


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

    return all_entries


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
