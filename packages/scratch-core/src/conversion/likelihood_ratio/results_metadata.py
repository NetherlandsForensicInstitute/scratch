import datetime
from pathlib import Path

from lir import LLRData

from conversion.data_formats import MarkType
from conversion.likelihood_ratio import ModelSpecs


def _format_lr(llr_data: LLRData) -> str:
    """Format a single log-LR value with optional confidence interval."""
    if len(llr_data.llrs) > 1:
        raise ValueError(f"expected single LR value, got {len(llr_data.llrs)}")

    log_lr = llr_data.llrs[0]

    if llr_data.llr_intervals is not None:
        lower, upper = llr_data.llr_intervals[0, 0], llr_data.llr_intervals[0, 1]
        return f"{log_lr:.2f} ({lower:.2f}, {upper:.2f})"
    return f"{log_lr:.2f}"


def _common_results_metadata(
    reference_data: ModelSpecs,
    llr_data: LLRData,
    date_report: datetime.date,
    user_id: str,
    mark_type: MarkType,
) -> dict[str, str]:
    """Results metadata fields shared across all mark types."""
    return {
        "Date report": date_report.isoformat(),
        "User ID": user_id,
        "Mark type": mark_type.value,
        "LogLR (5%, 95%)": _format_lr(llr_data),
        "# of KM scores": str(len(reference_data.km_scores)),
        "# of KNM scores": str(len(reference_data.knm_scores)),
    }


def build_results_metadata_striation(
    reference_data: ModelSpecs,
    llr_data: LLRData,
    date_report: datetime.date,
    user_id: str,
    mark_type: MarkType,
    score: float,
    score_transform: float,
) -> dict[str, str]:
    return {
        **_common_results_metadata(
            reference_data, llr_data, date_report, user_id, mark_type
        ),
        "Score type": "CCF",
        "Score (transform)": f"{score:.2f} ({score_transform:.2f})",
    }


def build_results_metadata_impression(
    reference_data: ModelSpecs,
    llr_data: LLRData,
    date_report: datetime.date,
    user_id: str,
    mark_type: MarkType,
    score: int,
    n_cells: int,
    lr_system_path: Path,
) -> dict[str, str]:
    return {
        **_common_results_metadata(
            reference_data, llr_data, date_report, user_id, mark_type
        ),
        "LR system path": str(lr_system_path),
        "Score type": "CMC",
        "Score (transform)": f"{score} of {n_cells}",
    }
