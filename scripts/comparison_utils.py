import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from conversion.data_formats import MarkImpressionType, MarkStriationType, MarkType
from conversion.surface_comparison.models import ComparisonParams

from scripts.conversion_utils import parse_db_scratch

logger = logging.getLogger(__name__)
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
        return parse_db_scratch(p / "db.scratch").get("NAME", p.name)

    return {
        "case_id": _name(te - 1) if te > 0 else "unknown",
        "firearm_id": _name(te + 1),
        "specimen_id": _name(te + 2),
        "measurement_id": _name(te + 3),
        "mark_id": _name(te + 4) if len(parts) > te + 4 else mark_dir.name,
    }


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
    body = {
        "mark_dir_ref": str(entry.mark_dir_ref),
        "mark_dir_comp": str(entry.mark_dir_comp),
        "metadata_reference": _extract_metadata(entry.mark_dir_ref),
        "metadata_compared": _extract_metadata(entry.mark_dir_comp),
    }
    if isinstance(entry.mark_type, MarkImpressionType):
        body["comparison_params"] = ComparisonParams.for_mark_type(entry.mark_type).model_dump()
    return body


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
    """Random cross-firearm pairs, sampled without materializing the full cross-product."""
    keys = list(by_firearm.keys())
    if len(keys) < 2:  # noqa: PLR2004
        return []

    rng = np.random.default_rng(seed)

    # Build an index of (firearm_pair_index, mark_a_index, mark_b_index) ranges
    # so we can sample uniformly without materializing every combination.
    firearm_pairs = [(i, j) for i in range(len(keys)) for j in range(i + 1, len(keys))]
    pair_sizes = np.array([len(by_firearm[keys[i]]) * len(by_firearm[keys[j]]) for i, j in firearm_pairs])
    total_pool = int(pair_sizes.sum())

    sample_size = min(n, total_pool)
    flat_indices = rng.choice(total_pool, size=sample_size, replace=False)
    flat_indices.sort()

    cumulative = np.cumsum(pair_sizes)
    bucket_indices = np.searchsorted(cumulative, flat_indices, side="right")

    results = []
    for flat_idx, bucket_idx in zip(flat_indices, bucket_indices):
        fi, fj = firearm_pairs[bucket_idx]
        offset = flat_idx - (cumulative[bucket_idx - 1] if bucket_idx > 0 else 0)
        marks_a = by_firearm[keys[fi]]
        marks_b = by_firearm[keys[fj]]
        ai, bi = divmod(int(offset), len(marks_b))
        results.append((marks_a[ai], marks_b[bi]))
    return results


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

    return [
        ComparisonEntry(a, b, mark_type, out_base / f"{i // 1000:04d}" / f"{i:06d}", i)
        for i, (a, b) in enumerate(pairs)
    ]
