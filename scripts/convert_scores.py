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
import enum
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests
from conversion.data_formats import MarkImpressionType

from scripts.comparison_utils import (
    ComparisonEntry,
    _build_body,
    _save_result,
    get_pairs,
)
from scripts.conversion_utils import (
    ConversionConfig,
    _find_existing_results,
    run_parallel,
)
from scripts.http_utils import _cleanup_vault, _post_with_retry, download_urls

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class ScoreStatus(enum.Enum):
    """Outcome of a single score calculation attempt."""

    COMPLETED = "completed"
    SKIPPED_EXISTS = "skipped_exists"
    SKIPPED_MISSING = "skipped_missing"
    FAILED_VALIDATION = "failed_validation"
    FAILED_ERROR = "failed_error"


def calculate_score(
    entry: ComparisonEntry, cfg: ConversionConfig, existing: set[Path]
) -> tuple[ScoreStatus, dict[str, Any] | None]:
    """Call the score endpoint for a single comparison pair.

    :returns: a ``(status, result_dict_or_none)`` tuple.
    """
    if entry.comparison_out in existing and not cfg.force:
        return ScoreStatus.SKIPPED_EXISTS, None

    if not entry.mark_dir_ref.exists() or not entry.mark_dir_comp.exists():
        logger.warning("Processed dir missing for row %d", entry.row_index)
        return ScoreStatus.SKIPPED_MISSING, None

    category = "impression" if isinstance(entry.mark_type, MarkImpressionType) else "striation"
    endpoint = f"processor/calculate-score-{category}"

    try:
        result = _post_with_retry(f"{cfg.api_url}/{endpoint}", _build_body(entry, skip_plots=cfg.skip_plots))
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 422:  # noqa: PLR2004
            try:
                detail = exc.response.json().get("detail", "unknown")
            except requests.JSONDecodeError:
                detail = exc.response.text[:200]
            logger.info("Row %d validation error: %s", entry.row_index, detail)
            _save_result(entry, error=detail)
            return ScoreStatus.FAILED_VALIDATION, {"error": detail}
        raise
    except Exception:
        logger.exception("Row %d failed unexpectedly", entry.row_index)
        _save_result(entry, error="unexpected error")
        return ScoreStatus.FAILED_ERROR, None

    _save_result(entry, result=result)
    download_urls(result.get("urls", result), entry.comparison_out)
    _cleanup_vault(result)
    return ScoreStatus.COMPLETED, result


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

    existing: set[Path] = set()
    if not cfg.force:
        existing = _find_existing_results(cfg.output_dir)
        logger.info("Found %d existing results", len(existing))

    results = run_parallel(
        ((e.row_index, calculate_score, (e, cfg, existing)) for e in all_entries),
        workers,
        "Calculating scores",
        " comparisons",
    )

    counts = defaultdict(int)
    for status, _ in results.values():
        counts[status] += 1

    logger.info(
        "Done: %d completed, %d skipped (exists), %d skipped (missing), %d validation errors, %d unexpected errors",
        counts[ScoreStatus.COMPLETED],
        counts[ScoreStatus.SKIPPED_EXISTS],
        counts[ScoreStatus.SKIPPED_MISSING],
        counts[ScoreStatus.FAILED_VALIDATION],
        counts[ScoreStatus.FAILED_ERROR],
    )


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
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation for faster bulk scoring")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for different-source sampling")
    args = parser.parse_args()

    cfg = ConversionConfig(
        root=args.root, output_dir=args.output, api_url=args.api_url, force=args.force, skip_plots=args.skip_plots
    )
    run_score_conversion(
        cfg,
        workers=args.workers,
        limit=args.limit,
        use_pairs_from_file=args.use_pairs_from_file,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
