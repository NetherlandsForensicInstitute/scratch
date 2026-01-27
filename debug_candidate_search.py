"""Verify candidate search and comp_scale for failing test cases."""
import sys
sys.path.insert(0, 'packages/scratch-core/src')

import numpy as np
from conversion.profile_correlator.candidate_search import find_match_candidates
from conversion.profile_correlator.data_types import Profile, AlignmentParameters

for test_name in ['edge_over_threshold', 'partial_with_nans']:
    base = f'packages/scratch-core/tests/resources/profile_correlator/{test_name}'
    ref_data = np.load(f'{base}/input_profile_ref.npy').ravel()
    comp_data = np.load(f'{base}/input_profile_comp.npy').ravel()

    import json
    with open(f'{base}/metadata.json') as f:
        meta = json.load(f)

    pixel_size_ref = meta['ref_xdim']
    pixel_size_comp = meta['comp_xdim']
    params_dict = meta['params']

    cutoff_hi = params_dict['cutoff_hi'] * 1e-6  # um -> m
    cutoff_lo = params_dict['cutoff_lo'] * 1e-6

    ref_profile = Profile(depth_data=ref_data, pixel_size=pixel_size_ref,
                          cutoff_hi=cutoff_hi, cutoff_lo=cutoff_lo)
    comp_profile = Profile(depth_data=comp_data, pixel_size=pixel_size_comp,
                           cutoff_hi=cutoff_hi, cutoff_lo=cutoff_lo)

    params = AlignmentParameters(
        scale_passes=tuple(p * 1e-6 for p in params_dict['pass']),
        max_translation=params_dict['max_translation'] * 1e-6,
        max_scaling=params_dict['max_scaling'],
        cutoff_hi=cutoff_hi,
        cutoff_lo=cutoff_lo,
        partial_mark_threshold=params_dict['part_mark_perc'],
        inclusion_threshold=params_dict['inclusion_threshold'],
    )

    positions, shape_scales, comp_scale = find_match_candidates(ref_profile, comp_profile, params)

    adjusted_passes = tuple(s for s in params.scale_passes if s <= comp_scale)

    print(f"\n=== {test_name} ===")
    print(f"  ref_length={len(ref_data)}, comp_length={len(comp_data)}")
    print(f"  pixel_size={pixel_size_ref}")
    print(f"  Candidate positions (Python 0-based): {positions}")
    print(f"  Shape scales (m): {shape_scales}")
    print(f"  comp_scale (m): {comp_scale} = {comp_scale*1e6:.0f} um")
    print(f"  Adjusted passes (um): {[f'{s*1e6:.0f}' for s in adjusted_passes]}")
    print(f"  resolution_limit: {max(cutoff_lo, 2*pixel_size_ref)*1e6:.1f} um")

    # Check which passes would actually be processed
    resolution_limit = max(cutoff_lo, 2*pixel_size_ref)
    _ftol = 1e-9
    active_passes = []
    for cutoff in adjusted_passes:
        if cutoff < resolution_limit * (1 - _ftol):
            continue
        if cutoff > cutoff_hi * (1 + _ftol):
            continue
        if cutoff < cutoff_lo * (1 - _ftol):
            continue
        active_passes.append(cutoff)
    print(f"  Active passes (um): {[f'{s*1e6:.0f}' for s in active_passes]}")
