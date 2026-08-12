"""Build comparison entries from a CSV file of item pairs.

The CSV holds (at least) two columns: the reference item and the comparison
item, both as paths relative to the database root.  A single item folder can
contain several mark types (``breech_face_impression_mark``,
``firing_pin_impression_mark``, ...), so every CSV row expands into one
comparison per mark type that *both* items have.

Results are written back as one copy of the original CSV per mark type, with
the relevant score columns appended.  The copies are rewritten after every
completed comparison, and an interrupted run is picked up again by reading
those same copies back in.
"""

from __future__ import annotations

import csv
import logging
import os
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from conversion.data_formats import MarkImpressionType, MarkType

from scripts.comparison_utils import ComparisonEntry, infer_mark_type
from scripts.conversion_utils import ConversionConfig

logger = logging.getLogger(__name__)

#: Folder holding the per-mark-type result folders, inside the database folder.
RESULTS_SUBDIR = Path("mark-comparison-results")

#: Keys in the API response. The metrics sit in a nested block
#: (``ComparisonImpressionMetrics`` / ``StriationComparisonResults``).
RESULTS_KEY = "comparison_results"
MATCHING_CELLS_KEY = "score"
TOTAL_CELLS_KEY = "n_cells"
CCF_KEY = "correlation_coefficient"

#: Filename ``_save_result`` writes inside ``entry.comparison_out``.
RESULT_FILENAMES = ("comparison_results.json",)

#: Statuses that count as "already done" when resuming.
DONE_STATUSES = frozenset({"completed", "skipped_exists"})


def find_mark_dirs(base: Path, item: str, max_depth: int = 2) -> dict[MarkType, Path]:
    """Find the mark-type folders inside a single item folder.

    Mark folders sit directly under the item, or one level deeper.  The search
    is breadth-first and stops descending as soon as a folder name resolves to
    a mark type, so ``max_depth`` counts intermediate layers, not nesting
    inside a mark folder itself.
    """
    item_dir = base / item
    found: dict[MarkType, Path] = {}
    if not item_dir.is_dir():
        logger.warning("Item folder not found: %s", item_dir)
        return found

    queue: deque[tuple[Path, int]] = deque([(item_dir, 0)])
    while queue:
        parent, depth = queue.popleft()
        for sub in sorted(p for p in parent.iterdir() if p.is_dir()):
            mark_type = infer_mark_type(sub.name)
            if mark_type is None:
                if depth + 1 < max_depth:
                    queue.append((sub, depth + 1))
                else:
                    logger.debug("Ignoring folder with unrecognised mark type: %s", sub)
            elif mark_type in found:
                logger.warning(
                    "Duplicate %s under %s: using %s, ignoring %s", mark_type.value, item_dir, found[mark_type], sub
                )
            else:
                found[mark_type] = sub
    return found


def get_score_columns(mark_type: MarkType) -> list[str]:
    """Extra CSV columns for this mark type."""
    if isinstance(mark_type, MarkImpressionType):
        return ["total_cells", "matching_cells", "status", "error"]
    return ["ccf", "status", "error"]


def get_metric_columns(mark_type: MarkType) -> list[str]:
    """Get the score columns proper, without the bookkeeping ones."""
    return [c for c in get_score_columns(mark_type) if c not in ("status", "error")]


def extract_metrics(result: dict[str, Any] | None, mark_type: MarkType) -> dict[str, Any]:
    """Pull the score columns out of an API response."""
    if not isinstance(result, dict):
        return {}

    comparison_results = result.get(RESULTS_KEY)
    if not isinstance(comparison_results, dict):
        logger.warning("Response has no %r block; keys present: %s", RESULTS_KEY, ", ".join(sorted(result)))
        return {}

    if isinstance(mark_type, MarkImpressionType):
        metrics = {
            "total_cells": comparison_results.get(TOTAL_CELLS_KEY),
            "matching_cells": comparison_results.get(MATCHING_CELLS_KEY),
        }
    else:
        metrics = {"ccf": comparison_results.get(CCF_KEY)}

    missing = [name for name, value in metrics.items() if value is None]
    if missing:
        logger.warning(
            "Response has no %s; %s keys: %s",
            ", ".join(missing),
            RESULTS_KEY,
            ", ".join(sorted(comparison_results)),
        )
    return metrics


def normalise_item(value: str) -> str:
    """Clean a CSV item path so it can safely be joined onto a base folder.

    Strips whitespace, converts Windows separators, and removes leading and
    trailing slashes.  The leading slash matters: ``Path("/a/b") / "/c"``
    discards the base and returns ``/c``.
    """
    return value.strip().replace("\\", "/").strip("/")


@dataclass
class PairRow:
    """One row of the input CSV."""

    index: int
    fields: list[str]

    @property
    def ref(self) -> str:
        return normalise_item(self.fields[0])

    @property
    def comp(self) -> str:
        return normalise_item(self.fields[1])


def _read_nonblank_rows(csv_path: Path, delimiter: str, encoding: str = "utf-8") -> list[list[str]]:
    """Read a CSV file, skipping fully blank lines."""
    with csv_path.open(newline="", encoding=encoding) as fh:
        return [row for row in csv.reader(fh, delimiter=delimiter) if any(cell.strip() for cell in row)]


def read_pairs_csv(csv_path: Path, base: Path, delimiter: str = ",") -> tuple[list[str] | None, list[PairRow]]:
    """Read the pair CSV, auto-detecting whether the first line is a header.

    :returns: a ``(header_or_none, rows)`` tuple.  All original columns are
        preserved verbatim so the file can be copied out again.
    """
    raw = _read_nonblank_rows(csv_path, delimiter, encoding="utf-8-sig")
    if not raw:
        raise ValueError(f"{csv_path} contains no usable rows")

    min_cols = 2
    if len(raw[0]) < min_cols:
        raise ValueError(f"{csv_path} needs at least two columns, found {len(raw[0])}")

    # If the first cell of the first line resolves to an existing folder it is data, not a header.
    header = None if (base / normalise_item(raw[0][0])).is_dir() else raw.pop(0)
    rows = [PairRow(index=i, fields=fields) for i, fields in enumerate(raw)]
    logger.info("Read %d pairs from %s (header: %s)", len(rows), csv_path.name, header is not None)
    return header, rows


@dataclass
class CsvTask:
    """A single (row, mark type) comparison to run."""

    task_id: int
    row: PairRow
    mark_type: MarkType
    entry: ComparisonEntry


def get_mark_type_slug(mark_type: MarkType) -> str:
    """Get folder-safe name for a mark type, matching ``_MARK_TYPE_FOLDER_MAP``."""
    return mark_type.value.replace(" ", "_")


def get_database_folder(item: str) -> str:
    """Get the database an item belongs to: its first path component below the root."""
    parts = Path(normalise_item(item)).parts
    return parts[0] if parts else ""


def get_item_slug(item: str) -> str:
    """
    Get the identifier for an item path.

    e.g. "database/tool-entries/item123/rep1" -> "tool-entries__item123__rep1"
    """
    parts = Path(normalise_item(item)).parts[1:]  # drop the leading database name
    if parts and parts[0] == "tool-entries":
        parts = parts[1:]  # drop the known "tool-entries" segment
    return "__".join(parts) if parts else Path(normalise_item(item)).name


def build_comparison_out_dir(cfg: ConversionConfig, mark_type: MarkType, row: PairRow) -> Path:
    """Where the full result payload for this comparison is stored.

    Results are grouped by database pair, i.e.
    ``<root>/database-comparisons/<db_a>__vs__<db_b>/<mark_type>_comparison_results/...``.
    Same-database comparisons simply get a pair folder where both names are
    the same (e.g. ``dbA__vs__dbA``). The pair name is sorted so a row with
    ref/comp swapped still lands in the same folder.
    """
    db_ref = get_database_folder(row.ref)
    db_comp = get_database_folder(row.comp)
    if db_ref != db_comp:
        logger.debug("Row %d compares across databases (%s vs %s)", row.index, db_ref, db_comp)

    pair = "__vs__".join(sorted((db_ref, db_comp)))
    folder = cfg.root / "database-comparisons" / pair / f"{get_mark_type_slug(mark_type)}_comparison_results"
    return folder / f"{get_item_slug(row.ref)}_vs_{get_item_slug(row.comp)}"


def build_tasks(cfg: ConversionConfig, rows: list[PairRow], base: Path, max_depth: int = 2) -> list[CsvTask]:
    """Expand every CSV row into one task per shared mark type."""
    tasks: list[CsvTask] = []
    for row in rows:
        ref_marks = find_mark_dirs(base, row.ref, max_depth)
        comp_marks = find_mark_dirs(base, row.comp, max_depth)
        shared = [mark_type for mark_type in ref_marks if mark_type in comp_marks]
        if not shared:
            logger.warning("Row %d: no mark type shared by %s and %s", row.index, row.ref, row.comp)
            continue
        for mark_type in shared:
            entry = ComparisonEntry(
                row_index=len(tasks),
                mark_type=mark_type,
                mark_dir_ref=ref_marks[mark_type],
                mark_dir_comp=comp_marks[mark_type],
                comparison_out=build_comparison_out_dir(cfg, mark_type, row),
            )
            csv_task = CsvTask(task_id=len(tasks), row=row, mark_type=mark_type, entry=entry)
            tasks.append(csv_task)
    return tasks


def build_tasks_from_entries(entries: list[ComparisonEntry], base: Path) -> list[CsvTask]:
    """
    Wrap already-built comparison entries (e.g. generated pairs) as CSV-style tasks.

    Each entry gets its own synthetic single-row "CSV" of two columns (ref, comp),
    with paths shown relative to *base* for readability.
    """
    tasks = []
    for i, entry in enumerate(entries):
        fields = [
            str(entry.mark_dir_ref.relative_to(base)),
            str(entry.mark_dir_comp.relative_to(base)),
        ]
        row = PairRow(index=i, fields=fields)
        tasks.append(CsvTask(task_id=i, row=row, mark_type=entry.mark_type, entry=entry))
    return tasks


def find_result_file(comparison_out: Path) -> Path | None:
    """Locate an already-saved result payload, if there is one."""
    for name in RESULT_FILENAMES:
        candidate = comparison_out / name
        if candidate.is_file():
            return candidate
    return next(iter(sorted(comparison_out.glob("*.json"))), None) if comparison_out.is_dir() else None


class ScoreWriter:
    """Keeps one scored copy of the input CSV per mark type up to date on disk.

    ``record`` is safe to call from several worker threads; each call updates
    the in-memory table and rewrites the affected file atomically, so the CSV
    on disk is never half-written and never more than ``flush_every``
    comparisons behind.

    When ``resume`` is set, any scored CSV already present is read back first,
    so an interrupted run continues instead of starting over.  A resume file
    is only trusted if its original columns still match the input CSV exactly.
    """

    def __init__(
        self,
        out_dir: Path,
        csv_path: Path,
        header: list[str] | None,
        rows: list[PairRow],
        mark_types: list[MarkType],
        delimiter: str = ",",
        flush_every: int = 1,
        resume: bool = True,
    ) -> None:
        self.header = header
        self.rows = rows
        self.delimiter = delimiter
        self.flush_every = max(1, flush_every)
        self.paths = {mt: out_dir / f"{csv_path.stem}_{get_mark_type_slug(mt)}{csv_path.suffix}" for mt in mark_types}
        self.columns = {mt: get_score_columns(mt) for mt in mark_types}
        self.values: dict[MarkType, dict[int, dict[str, Any]]] = {mt: {} for mt in mark_types}
        self._pending: dict[MarkType, int] = defaultdict(int)
        self._lock = threading.Lock()

        out_dir.mkdir(parents=True, exist_ok=True)
        for mark_type in mark_types:
            if resume:
                self.values[mark_type] = self._read_previous(mark_type)
            self._write(mark_type)
        logger.info("Writing scored copies to %s", ", ".join(p.name for p in self.paths.values()))

    def _read_previous(self, mark_type: MarkType) -> dict[int, dict[str, Any]]:
        """Load the scores from an earlier run of this same CSV."""
        path = self.paths[mark_type]
        if not path.is_file():
            return {}

        raw = _read_nonblank_rows(path, self.delimiter)
        if self.header is not None and raw:
            raw.pop(0)

        if len(raw) != len(self.rows):
            logger.warning("Ignoring %s for resume: %d rows, expected %d", path.name, len(raw), len(self.rows))
            return {}

        columns = self.columns[mark_type]
        recovered: dict[int, dict[str, Any]] = {}
        for row, previous in zip(self.rows, raw):
            width = len(row.fields)
            if previous[:width] != row.fields:
                logger.warning("Ignoring %s for resume: row %d no longer matches the input", path.name, row.index)
                return {}
            scores = {col: cell for col, cell in zip(columns, previous[width:]) if cell != ""}
            if scores.get("status"):
                recovered[row.index] = scores

        if recovered:
            done = sum(1 for s in recovered.values() if s.get("status") in DONE_STATUSES)
            logger.info(
                "Resuming %s: %d rows already scored, %d other outcomes", path.name, done, len(recovered) - done
            )
        return recovered

    def recorded_status(self, mark_type: MarkType, row_index: int) -> str | None:
        """Status stored for this row, from a previous run or this one."""
        return self.values.get(mark_type, {}).get(row_index, {}).get("status")

    def has_metrics(self, mark_type: MarkType, row_index: int) -> bool:
        """Whether this row already has actual score values."""
        scores = self.values.get(mark_type, {}).get(row_index, {})
        return any(scores.get(col) not in (None, "") for col in get_metric_columns(mark_type))

    def record(self, mark_type: MarkType, row_index: int, values: dict[str, Any]) -> None:
        """Store the scores for one comparison and flush if due."""
        with self._lock:
            self.values[mark_type][row_index] = values
            self._pending[mark_type] += 1
            if self._pending[mark_type] >= self.flush_every:
                self._write(mark_type)
                self._pending[mark_type] = 0

    def flush(self) -> None:
        """Write every mark type out, regardless of the flush interval."""
        with self._lock:
            for mark_type in self.paths:
                self._write(mark_type)
                self._pending[mark_type] = 0

    def _write(self, mark_type: MarkType) -> None:
        """Rewrite one file. The caller must hold the lock."""
        path = self.paths[mark_type]
        columns = self.columns[mark_type]
        values = self.values[mark_type]
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, delimiter=self.delimiter)
            if self.header is not None:
                writer.writerow([*self.header, *columns])
            for row in self.rows:
                scores = values.get(row.index, {})
                writer.writerow([*row.fields, *("" if scores.get(c) is None else scores[c] for c in columns)])
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
