"""
Calculate scores for mark comparisons via the Python API.

Three modes of operation:

1. **From results_table.mat** (``--use_pairs_from_file``): reads comparison
   pairs from the mark-comparison-results folder structure in an existing
   MATLAB database.

2. **Generated pairs** (default): discovers all processed marks in the output
   directory, groups them by firearm, and generates all same-source pairs plus
   an equal number of random different-source pairs.

3. **From a CSV** (``--pairs-csv``): reads item pairs from the first two
   columns of a CSV file.  Each item folder may hold several mark types, so a
   row expands into one comparison per shared mark type.  The input CSV is
   copied once per mark type with the scores appended (CCF for striation
   marks, total and matching cell counts for impression marks), updated after
   every comparison.  Re-running the same command picks up where an
   interrupted run stopped; ``--retry-failed`` also re-runs the rows that
   errored.  The full result payloads are still saved to the usual
   ``<root>/database/mark-comparison-results/<mark_type>_comparison_results``
   folders.
"""

import argparse
import enum
import json
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
from scripts.csv_pairs import (
    DONE_STATUSES,
    CsvTask,
    ScoreWriter,
    build_tasks,
    extract_metrics,
    find_result_file,
    read_pairs_csv,
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
        result = _post_with_retry(f"{cfg.api_url}/{endpoint}", _build_body(entry))
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


def _log_counts(counts: dict[ScoreStatus, int]) -> None:
    logger.info(
        "Done: %d completed, %d skipped (exists), %d skipped (missing), %d validation errors, %d unexpected errors",
        counts[ScoreStatus.COMPLETED],
        counts[ScoreStatus.SKIPPED_EXISTS],
        counts[ScoreStatus.SKIPPED_MISSING],
        counts[ScoreStatus.FAILED_VALIDATION],
        counts[ScoreStatus.FAILED_ERROR],
    )


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

    counts: dict[ScoreStatus, int] = defaultdict(int)
    for status, _ in results.values():
        counts[status] += 1
    _log_counts(counts)


# --------------------------------------------------------------------------
# CSV mode
# --------------------------------------------------------------------------
def _load_saved_result(comparison_out: Path) -> dict[str, Any] | None:
    """Read a previously saved result payload from disk."""
    path = find_result_file(comparison_out)
    if path is None:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not read existing result %s", path)
        return None


def _row_values(status: ScoreStatus, result: dict[str, Any] | None, task: CsvTask) -> dict[str, Any]:
    """Turn one comparison outcome into the CSV cells for its row."""
    values: dict[str, Any] = {"status": status.value}
    if isinstance(result, dict) and result.get("error"):
        values["error"] = result["error"]
    else:
        values.update(extract_metrics(result, task.mark_type))
    return values


def score_and_record(
    task: CsvTask, cfg: ConversionConfig, existing: set[Path], writer: ScoreWriter
) -> tuple[ScoreStatus, dict[str, Any] | None]:
    """Score one pair and immediately write its row to the scored CSV."""
    try:
        status, result = calculate_score(task.entry, cfg, existing)
    except Exception:
        # Still leave a trace in the CSV before letting run_parallel handle it.
        writer.record(task.mark_type, task.row.index, _row_values(ScoreStatus.FAILED_ERROR, None, task))
        raise

    if status is ScoreStatus.SKIPPED_EXISTS:
        # Nothing was recalculated, so read back what was saved on an earlier run.
        result = _load_saved_result(task.entry.comparison_out)
        if result is None and writer.has_metrics(task.mark_type, task.row.index):
            # Payload unreadable but the resumed CSV already holds the scores: keep them.
            return status, None

    writer.record(task.mark_type, task.row.index, _row_values(status, result, task))
    return status, result


def _existing_results(tasks: list[CsvTask], writer: ScoreWriter, retry_failed: bool) -> set[Path]:
    """Decide which comparisons can be skipped, from the resumed CSV and from disk."""
    existing: set[Path] = set()
    for task in tasks:
        status = writer.recorded_status(task.mark_type, task.row.index)
        if status in DONE_STATUSES:
            existing.add(task.entry.comparison_out)
        elif status is not None and retry_failed:
            continue  # errored last time and the user asked for a retry
        elif find_result_file(task.entry.comparison_out) is not None:
            existing.add(task.entry.comparison_out)
    return existing


def run_csv_score_conversion(
    cfg: ConversionConfig,
    csv_path: Path,
    workers: int = 1,
    limit: int | None = None,
    base: Path | None = None,
    out_dir: Path | None = None,
    delimiter: str = ",",
    flush_every: int = 1,
    retry_failed: bool = False,
    max_depth: int = 2,
) -> None:
    """Score the pairs listed in ``csv_path``, writing one scored CSV per mark type.

    :param base: folder the CSV items are relative to; defaults to the
        processed-mark output directory.
    :param out_dir: where the scored CSV copies are written; defaults to next
        to the input CSV.
    :param flush_every: rewrite a scored CSV after this many comparisons.
    :param retry_failed: re-run rows that ended in an error last time.
    :param max_depth: how deep below an item folder to look for mark folders.
    """
    base = base or cfg.output_dir
    out_dir = out_dir or csv_path.parent

    header, rows = read_pairs_csv(csv_path, base, delimiter)
    if limit is not None:
        rows = rows[:limit]

    tasks = build_tasks(cfg, rows, base, max_depth)
    if not tasks:
        logger.warning("No comparisons to process")
        return
    logger.info("Expanded %d rows into %d comparisons", len(rows), len(tasks))

    writer = ScoreWriter(
        out_dir=out_dir,
        csv_path=csv_path,
        header=header,
        rows=rows,
        mark_types=list(dict.fromkeys(t.mark_type for t in tasks)),
        delimiter=delimiter,
        flush_every=flush_every,
        resume=not cfg.force,
    )

    existing = set() if cfg.force else _existing_results(tasks, writer, retry_failed)
    logger.info("%d of %d comparisons already done", len(existing), len(tasks))

    try:
        results = run_parallel(
            ((t.task_id, score_and_record, (t, cfg, existing, writer)) for t in tasks),
            workers,
            "Calculating scores",
            " comparisons",
        )
    finally:
        writer.flush()

    counts: dict[ScoreStatus, int] = defaultdict(int)
    for task in tasks:
        status, _ = results.get(task.task_id, (ScoreStatus.FAILED_ERROR, None))
        counts[status] += 1
    _log_counts(counts)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Calculate scores for mark comparisons via Python API")
    parser.add_argument("root", type=Path, help="Root database folder")
    parser.add_argument("output", type=Path, help="Output folder (same as used for mark conversion)")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers")
    parser.add_argument("--force", action="store_true", help="Recalculate existing results")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N comparisons (or N CSV rows)")
    parser.add_argument(
        "--use_pairs_from_file", action="store_true", help="Read pairs from results_table.mat instead of generating"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for different-source sampling")

    csv_group = parser.add_argument_group("CSV pairs")
    csv_group.add_argument("--pairs-csv", type=Path, default=None, help="CSV whose first two columns are item pairs")
    csv_group.add_argument(
        "--csv-base",
        choices=("output", "root"),
        default="output",
        help="Folder the CSV items are relative to (default: the output folder with processed marks)",
    )
    csv_group.add_argument("--csv-out-dir", type=Path, default=None, help="Where to write the scored CSV copies")
    csv_group.add_argument("--csv-delimiter", default=",", help="CSV delimiter")
    csv_group.add_argument(
        "--csv-flush-every", type=int, default=1, help="Rewrite a scored CSV after this many comparisons"
    )
    csv_group.add_argument("--retry-failed", action="store_true", help="Re-run rows that errored on a previous run")
    csv_group.add_argument(
        "--csv-max-depth", type=int, default=2, help="How deep below an item folder to look for mark folders"
    )
    args = parser.parse_args()

    if args.pairs_csv and args.use_pairs_from_file:
        parser.error("--pairs-csv and --use_pairs_from_file are mutually exclusive")

    cfg = ConversionConfig(
        root=args.root,
        output_dir=args.output,
        api_url=args.api_url,
        force=args.force,
    )

    if args.pairs_csv:
        run_csv_score_conversion(
            cfg,
            csv_path=args.pairs_csv,
            workers=args.workers,
            limit=args.limit,
            base=args.root if args.csv_base == "root" else args.output,
            out_dir=args.csv_out_dir,
            delimiter=args.csv_delimiter,
            flush_every=args.csv_flush_every,
            retry_failed=args.retry_failed,
            max_depth=args.csv_max_depth,
        )
        return

    run_score_conversion(
        cfg,
        workers=args.workers,
        limit=args.limit,
        use_pairs_from_file=args.use_pairs_from_file,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()