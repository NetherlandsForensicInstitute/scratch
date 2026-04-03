import json
import logging
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from conversion.data_formats import MarkImpressionType, MarkStriationType, MarkType
from conversion.surface_comparison.models import ComparisonParams
from scipy import io as sio
from tqdm import tqdm

from scripts.conversion_utils import ConversionConfig, parse_db_scratch
from scripts.matlab_utils import unwrap_path

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
        return parse_db_scratch(p / "db.scratch").get("NAME", p.name)

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


def _build_body(entry: ComparisonEntry, skip_plots: bool = False) -> dict[str, Any]:
    """Build the API request body for a comparison."""
    processed_ref = str(entry.mark_dir_ref)
    processed_comp = str(entry.mark_dir_comp)

    if isinstance(entry.mark_type, MarkImpressionType):
        return {
            "mark_dir_ref": processed_ref,
            "mark_dir_comp": processed_comp,
            "metadata_reference": _extract_metadata(entry.mark_dir_ref),
            "metadata_compared": _extract_metadata(entry.mark_dir_comp),
            "comparison_params": ComparisonParams.for_mark_type(entry.mark_type).model_dump(),
            "skip_plots": skip_plots,
        }
    return {
        "mark_dir_ref": processed_ref,
        "mark_dir_comp": processed_comp,
        "metadata_reference": _extract_metadata(entry.mark_dir_ref),
        "metadata_compared": _extract_metadata(entry.mark_dir_comp),
        "skip_plots": skip_plots,
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

    entries = []
    for i in range(refs.shape[0]):
        ref = unwrap_path(refs[i, 0] if refs.ndim > 1 else refs[i])
        comp = unwrap_path(comps[i, 0] if comps.ndim > 1 else comps[i])
        if not ref or not comp:
            logger.debug("Skipping row %d: empty path (ref=%r, comp=%r)", i, ref, comp)
            continue
        entries.append(
            ComparisonEntry(
                _resolve_mark_dir(ref, output_dir),
                _resolve_mark_dir(comp, output_dir),
                mark_type,
                out_base / f"{i // 1000:04d}" / f"{i:06d}",
                i,
            )
        )
    return entries


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
