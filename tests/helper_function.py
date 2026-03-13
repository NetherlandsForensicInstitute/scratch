from conversion.surface_comparison.models import Cell, CellMetaData


def make_cell(  # noqa: PLR0913
    center_reference: tuple[float, float] = (0.0, 0.0),
    best_score: float = 0.8,
    *,
    is_congruent: bool = False,
    angle_deg: float = 0.0,
    center_comparison: tuple[float, float] | None = None,
    cell_size: tuple[float, float] = (50e-6, 50e-6),
    fill_fraction_reference: float = 0.9,
    is_outlier: bool | None = None,
    residual_angle_deg: float = 0.0,
    position_error: tuple[float, float] = (0.0, 0.0),
) -> Cell:
    """Create a Cell instance for testing."""
    if center_comparison is None:
        center_comparison = center_reference
    if is_outlier is None:
        is_outlier = not is_congruent

    return Cell(
        center_reference=center_reference,
        cell_size=cell_size,
        fill_fraction_reference=fill_fraction_reference,
        best_score=best_score,
        angle_deg=angle_deg,
        center_comparison=center_comparison,
        is_congruent=is_congruent,
        meta_data=CellMetaData(
            is_outlier=is_outlier,
            residual_angle_deg=residual_angle_deg,
            position_error=position_error,
        ),
    )
