import numpy as np
from conversion.surface_comparison.models import ComparisonResult, ComparisonParams


def classify_cmc_cells(result: ComparisonResult, params: ComparisonParams) -> None:
    """
    Identify Congruent Matching Cells (CMCs) using pairwise voting (Tong et al., 2015).

    This implementation performs an exhaustive search where every valid cell acts as a
    consensus hypothesis. The hypothesis that recruits the most matching cells is chosen
    as the global consensus.

    :param result: result object containing the list of cells.
    :param params: algorithm parameters.
    """
    valid_cells = [
        cell
        for cell in result.cells
        if cell.area_cross_correlation_function_score >= params.correlation_threshold
    ]
    if not valid_cells:
        return
    num_valid = len(valid_cells)
    votes = np.zeros(num_valid, dtype=int)

    # Exhaustive pairwise voting
    for i in range(num_valid):
        hypothesis_angle = valid_cells[i].registration_angle
        hypothesis_translation = (
            valid_cells[i].center_comparison - valid_cells[i].center_reference
        )

        for j in range(num_valid):
            angle_deviation = np.degrees(
                np.abs(valid_cells[j].registration_angle - hypothesis_angle)
            )
            current_translation = (
                valid_cells[j].center_comparison - valid_cells[j].center_reference
            )
            distance_deviation = np.linalg.norm(
                current_translation - hypothesis_translation
            )

            if (
                angle_deviation <= params.angle_threshold
                and distance_deviation <= params.position_threshold
            ):
                votes[i] += 1

    # Determine global consensus from the winning hypothesis
    winner_index = np.argmax(votes)
    result.consensus_rotation = valid_cells[winner_index].registration_angle
    result.consensus_translation = (
        valid_cells[winner_index].center_comparison
        - valid_cells[winner_index].center_reference
    )

    # Final label assignment relative to the winning consensus
    for cell in result.cells:
        angle_error = np.degrees(
            np.abs(cell.registration_angle - result.consensus_rotation)
        )
        current_shift = cell.center_comparison - cell.center_reference
        position_error = np.linalg.norm(current_shift - result.consensus_translation)

        cell.is_congruent = bool(
            cell.area_cross_correlation_function_score >= params.correlation_threshold
            and angle_error <= params.angle_threshold
            and position_error <= params.position_threshold
        )

    result.update_summary()
