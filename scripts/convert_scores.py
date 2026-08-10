"""
Calculate scores for mark comparisons via the Python API.

Two modes of operation:

1. **Generated pairs** (default): discovers all processed marks in the output
   directory, groups them by firearm, and generates all same-source pairs plus
   an equal number of random different-source pairs.

2. **From a CSV** (``--pairs-csv``): reads item pairs from the first two
   columns of a CSV file.  Each item folder may hold several mark types, so a
   row expands into one comparison per shared mark type.

Either way, results are written as one scored, resumable CSV per mark type
(CCF for striation marks, total and matching cell counts for impression
marks), rewritten after every completed comparison.  Re-running the same
command picks up where an interrupted run stopped; ``--retry-failed`` also
re-runs the rows that errored.  The full result payloads are still saved to
the usual ``<root>/database/mark-comparison-results/<mark_type>_comparison_results``
folders (CSV mode) or ``<output>/generated-comparison-results/...`` (generated mode).
"""

import argparse
import enum
import json
import logging
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import requests
from conversion.data_formats import MarkImpressionType

from scripts.comparison_utils import (
    ComparisonEntry,
    _build_body,
    _save_result,
    find_all_mark_types,
    generate_pairs,
)
from scripts.conversion_utils import ConversionConfig, run_parallel
from scripts.csv_pairs import (
    DONE_STATUSES,
    CsvTask,
    ScoreWriter,
    build_tasks,
    extract_metrics,
    find_result_file,
    read_pairs_csv,
    tasks_from_entries,
)
from scripts.http_utils import _cleanup_vault, _post_with_retry, download_urls

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

GENERATED_CSV_NAME = Path("generated_pairs.csv")
GENERATED_HEADER = ["item_ref", "item_comp"]


class ScoreStatus(enum.Enum):
    """Outcome of a single score calculation attempt."""

    COMPLETED = "completed"
    SKIPPED_EXISTS = "skipped_exists"
    SKIPPED_MISSING = "skipped_missing"
    FAILED_VALIDATION = "failed_validation"
    FAILED_CONNECTION = "failed_connection"
    FAILED_ERROR = "failed_error"


def calculate_score(  # noqa: PLR0911
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
        status_code = exc.response.status_code if exc.response is not None else 0
        if status_code == 422:  # noqa: PLR2004
            if exc.response is None:
                detail = "unknown"
            else:
                try:
                    detail = exc.response.json().get("detail", "unknown")
                except requests.JSONDecodeError:
                    detail = exc.response.text[:200]
            logger.info("Row %d validation error: %s", entry.row_index, detail)
            _save_result(entry, error=detail)
            return ScoreStatus.FAILED_VALIDATION, {"error": detail}
        if status_code >= 500:  # noqa: PLR2004
            # Server-side failure (OOM, worker restart): retryable, so no payload is written.
            detail = exc.response.text[:200] if exc.response is not None else str(exc)
            logger.warning("Row %d: HTTP %d: %s", entry.row_index, status_code, detail)
            return ScoreStatus.FAILED_CONNECTION, {"error": f"HTTP {status_code}: {detail}"}
        raise
    except requests.RequestException as exc:
        # Transient network/transport failure (timeout, dropped connection, dead worker):
        # deliberately no _save_result, so the pair is retried on the next run.
        logger.warning("Row %d: %s: %s", entry.row_index, type(exc).__name__, exc)
        return ScoreStatus.FAILED_CONNECTION, {"error": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:
        logger.exception("Row %d failed unexpectedly", entry.row_index)
        detail = f"{type(exc).__name__}: {exc}"
        _save_result(entry, error=detail)
        return ScoreStatus.FAILED_ERROR, {"error": detail}

    _save_result(entry, result=result)
    downloaded = download_urls(result.get("urls", result), entry.comparison_out, skip=())

    if not downloaded:
        logger.warning(
            "Row %d: no files downloaded into %s; response keys: %s. Leaving the API vault in place.",
            entry.row_index,
            entry.comparison_out,
            ", ".join(sorted(result)),
        )
    else:
        _cleanup_vault(result)
    return ScoreStatus.COMPLETED, result


def _log_counts(counts: dict[ScoreStatus, int]) -> None:
    summary = ", ".join(f"{counts[status]} {status.value}" for status in ScoreStatus if counts.get(status))
    logger.info("Done: %s", summary or "nothing processed")


def _run_scoring(work: Iterable[tuple[int, Any, tuple]], ids: list[int], workers: int) -> dict[ScoreStatus, int]:
    """Run parallel scoring and tally the outcome counts.

    Missing entries (e.g. a worker crashed before recording anything) count as
    FAILED_ERROR rather than being silently dropped from the tally.
    """
    results = run_parallel(work, workers, "Calculating scores", " comparisons")
    counts: dict[ScoreStatus, int] = defaultdict(int)
    for item_id in ids:
        status, _ = results.get(item_id, (ScoreStatus.FAILED_ERROR, None))
        counts[status] += 1
    return counts


def get_tasks(
    cfg: ConversionConfig,
    limit: int | None = None,
    seed: int | None = None,
    csv_path: Path | None = None,
    base: Path | None = None,
    delimiter: str = ",",
    max_depth: int = 2,
) -> tuple[list[str] | None, list[CsvTask]]:
    """
    Collect comparison tasks, either from a CSV of item pairs or generated from all mark types found in the directory.

    :returns: a ``(header, tasks)`` tuple. ``header`` is the original CSV
        header (or ``None`` if the input had none) when *csv_path* is given,
        otherwise a synthetic two-column header for the generated pairs.
    """
    if csv_path is not None:
        base = base or cfg.output_dir
        header, rows = read_pairs_csv(csv_path, base, delimiter)
        if limit is not None:
            rows = rows[:limit]
        return header, build_tasks(cfg, rows, base, max_depth)

    all_entries = []
    for mt in find_all_mark_types(cfg.output_dir):
        entries = generate_pairs(cfg.output_dir, mt, seed=seed)
        logger.info("Generated %d pairs for '%s'", len(entries), mt.value)
        all_entries.extend(entries)
    if limit:
        all_entries = all_entries[:limit]
    return GENERATED_HEADER, tasks_from_entries(all_entries, cfg.output_dir)


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


def run_score_conversion(
    cfg: ConversionConfig,
    workers: int = 1,
    limit: int | None = None,
    seed: int | None = None,
    csv_path: Path | None = None,
    csv_base: Path | None = None,
    out_dir: Path | None = None,
    delimiter: str = ",",
    flush_every: int = 1,
    retry_failed: bool = False,
    max_depth: int = 2,
) -> None:
    """Score comparison pairs and write one scored, resumable CSV per mark type.

    :param csv_path: if set, read pairs from this CSV instead of generating them.
    :param csv_base: folder the CSV items are relative to (CSV mode only).
    :param out_dir: where the scored CSV copies are written; defaults to next
        to the input CSV (CSV mode) or under the output directory (generated mode).
    :param flush_every: rewrite a scored CSV after this many comparisons.
    :param retry_failed: re-run rows that ended in an error last time (CSV mode only).
    :param max_depth: how deep below an item folder to look for mark folders (CSV mode only).
    """
    header, tasks = get_tasks(
        cfg, limit=limit, seed=seed, csv_path=csv_path, base=csv_base, delimiter=delimiter, max_depth=max_depth
    )
    if not tasks:
        logger.warning("No comparisons to process")
        return
    logger.info("Total comparisons: %d", len(tasks))

    out_dir = out_dir or (csv_path.parent if csv_path is not None else cfg.output_dir / "generated-comparison-results")
    writer_csv_path = csv_path if csv_path is not None else GENERATED_CSV_NAME

    writer = ScoreWriter(
        out_dir=out_dir,
        csv_path=writer_csv_path,
        header=header,
        rows=[task.row for task in tasks],
        mark_types=list(dict.fromkeys(task.mark_type for task in tasks)),
        delimiter=delimiter,
        flush_every=flush_every,
        resume=not cfg.force,
    )

    existing = set() if cfg.force else _existing_results(tasks, writer, retry_failed)
    logger.info("%d of %d comparisons already done", len(existing), len(tasks))

    try:
        counts = _run_scoring(
            ((t.task_id, score_and_record, (t, cfg, existing, writer)) for t in tasks),
            [t.task_id for t in tasks],
            workers,
        )
    finally:
        writer.flush()

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
    parser.add_argument("--seed", type=int, default=42, help="Random seed for different-source sampling")
    parser.add_argument("--pairs-csv", type=Path, default=None, help="CSV whose first two columns are item pairs")
    parser.add_argument(
        "--csv-base",
        choices=("output", "root"),
        default="output",
        help="Folder the CSV items are relative to (default: the output folder with processed marks)",
    )
    parser.add_argument("--csv-out-dir", type=Path, default=None, help="Where to write the scored CSV copies")
    parser.add_argument("--csv-delimiter", default=",", help="CSV delimiter")
    parser.add_argument(
        "--csv-flush-every", type=int, default=1, help="Rewrite a scored CSV after this many comparisons"
    )
    parser.add_argument("--retry-failed", action="store_true", help="Re-run rows that errored on a previous run")
    parser.add_argument(
        "--csv-max-depth", type=int, default=2, help="How deep below an item folder to look for mark folders"
    )
    args = parser.parse_args()

    cfg = ConversionConfig(root=args.root, output_dir=args.output, api_url=args.api_url, force=args.force)

    run_score_conversion(
        cfg,
        workers=args.workers,
        limit=args.limit,
        seed=args.seed,
        csv_path=args.pairs_csv,
        csv_base=(args.root if args.csv_base == "root" else args.output) if args.pairs_csv else None,
        out_dir=args.csv_out_dir,
        delimiter=args.csv_delimiter,
        flush_every=args.csv_flush_every,
        retry_failed=args.retry_failed,
        max_depth=args.csv_max_depth,
    )
