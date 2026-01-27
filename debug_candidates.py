"""Debug: trace candidate search for both failing test cases."""
import sys
sys.path.insert(0, 'packages/scratch-core/src')

import numpy as np
from conversion.profile_correlator.data_types import Profile, AlignmentParameters
from conversion.profile_correlator.candidate_search import find_match_candidates

pixel_size = 3.5e-6

for case_name in ['edge_over_threshold', 'partial_with_nans']:
    print(f"\n{'='*60}")
    print(f"Case: {case_name}")
    print(f"{'='*60}")

    base = f'packages/scratch-core/tests/resources/profile_correlator/{case_name}'
    ref_data = np.load(f'{base}/input_profile_ref.npy').ravel()
    comp_data = np.load(f'{base}/input_profile_comp.npy').ravel()

    print(f"ref: {len(ref_data)}, comp: {len(comp_data)}")
    print(f"ref NaN count: {np.sum(np.isnan(ref_data))}")
    print(f"comp NaN count: {np.sum(np.isnan(comp_data))}")

    pr = Profile(depth_data=ref_data, pixel_size=pixel_size, cutoff_hi=1e-3, cutoff_lo=5e-6)
    pc = Profile(depth_data=comp_data, pixel_size=pixel_size, cutoff_hi=1e-3, cutoff_lo=5e-6)

    # Try different thresholds
    for threshold in [0.5, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.0]:
        params = AlignmentParameters(
            cutoff_hi=1e-3, cutoff_lo=5e-6,
            inclusion_threshold=threshold,
        )
        pos, shape_scales, comp_scale = find_match_candidates(pr, pc, params)
        if len(pos) > 0:
            print(f"  threshold={threshold:.2f}: candidates={pos}, comp_scale={comp_scale*1e6:.0f} um")
            break
        else:
            print(f"  threshold={threshold:.2f}: NO candidates")

    # Check the alignment flow
    print(f"\n  Now checking what align_partial_profile_multiscale does:")
    from conversion.profile_correlator.alignment import align_partial_profile_multiscale
    params_full = AlignmentParameters(
        cutoff_hi=1e-3, cutoff_lo=5e-6,
        partial_mark_threshold=8.0,
        inclusion_threshold=0.5,
    )

    if len(ref_data) > len(comp_data):
        result, start_pos = align_partial_profile_multiscale(pr, pc, params_full)
    else:
        result, start_pos = align_partial_profile_multiscale(pc, pr, params_full)

    print(f"  Best start position: {start_pos}")
    print(f"  Total translation: {result.total_translation:.4f} samples = {result.total_translation * pixel_size * 1e6:.2f} um")
    print(f"  Total scaling: {result.total_scaling:.6f}")
    print(f"  Final correlation: {result.final_correlation:.6f}")
    print(f"  Aligned lengths: ref={len(result.reference_aligned)}, comp={len(result.compared_aligned)}")
    partial_len = min(len(ref_data), len(comp_data))
    print(f"  pOverlap: {len(result.reference_aligned)/partial_len:.6f}")
