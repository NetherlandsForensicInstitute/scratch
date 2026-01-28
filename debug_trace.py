"""Trace the full partial alignment flow for failing test cases."""

import sys

sys.path.insert(0, "packages/scratch-core/src")

import numpy as np
from conversion.profile_correlator.alignment import align_profiles_multiscale
from conversion.profile_correlator.candidate_search import find_match_candidates
from conversion.profile_correlator.data_types import AlignmentParameters, Profile
from conversion.profile_correlator.transforms import compute_cumulative_transform

pixel_size = 3.5e-6

for case_name in ["edge_over_threshold", "partial_with_nans"]:
    print(f"\n{'=' * 60}")
    print(f"Case: {case_name}")
    print(f"{'=' * 60}")

    base = f"packages/scratch-core/tests/resources/profile_correlator/{case_name}"
    ref_data = np.load(f"{base}/input_profile_ref.npy").ravel()
    comp_data = np.load(f"{base}/input_profile_comp.npy").ravel()

    print(f"ref: {len(ref_data)}, comp: {len(comp_data)}")
    print(f"ref NaN: {np.sum(np.isnan(ref_data))}, comp NaN: {np.sum(np.isnan(comp_data))}")

    pr = Profile(depth_data=ref_data, pixel_size=pixel_size, cutoff_hi=1e-3, cutoff_lo=5e-6)
    pc = Profile(depth_data=comp_data, pixel_size=pixel_size, cutoff_hi=1e-3, cutoff_lo=5e-6)

    params = AlignmentParameters(cutoff_hi=1e-3, cutoff_lo=5e-6, partial_mark_threshold=8.0)

    # Step 1: Determine which is longer
    size1, size2 = len(ref_data), len(comp_data)
    diff_pct = abs(size1 - size2) / max(size1, size2) * 100
    print(f"Length diff: {diff_pct:.2f}% (threshold: 8.0)")
    is_partial = diff_pct >= 8.0
    print(f"Is partial: {is_partial}")

    if not is_partial:
        print("NOT partial - skipping")
        continue

    if size1 > size2:
        ref_profile, partial_profile = pr, pc
        ref_mean = ref_data
        partial_mean = comp_data
    else:
        ref_profile, partial_profile = pc, pr
        ref_mean = comp_data
        partial_mean = ref_data

    partial_length = len(partial_mean)
    print(f"Reference length: {len(ref_mean)}, Partial length: {partial_length}")

    # Step 2: Candidate search
    cand_pos, shape_scales, comp_scale = find_match_candidates(ref_profile, partial_profile, params)
    print(f"\nCandidate positions: {cand_pos}")
    print(f"Shape scales: {shape_scales * 1e6} μm")
    print(f"Comparison scale: {comp_scale * 1e6:.0f} μm")

    # Adjusted scale passes
    adjusted_passes = tuple(s for s in params.scale_passes if s <= comp_scale)
    print(f"Adjusted scale passes: {[f'{s * 1e6:.0f}' for s in adjusted_passes]} μm")

    # Step 3: Fine alignment at each candidate
    params_adj = AlignmentParameters(
        scale_passes=adjusted_passes if adjusted_passes else params.scale_passes,
        cutoff_hi=params.cutoff_hi,
        cutoff_lo=params.cutoff_lo,
        partial_mark_threshold=params.partial_mark_threshold,
    )

    for pos in cand_pos:
        end_idx = min(pos + partial_length, len(ref_mean))
        ref_segment = ref_mean[pos:end_idx]

        ref_seg_prof = Profile(
            depth_data=ref_segment,
            pixel_size=pixel_size,
            cutoff_hi=ref_profile.cutoff_hi,
            cutoff_lo=ref_profile.cutoff_lo,
        )

        if len(ref_segment) < partial_length:
            part_trim = partial_mean[: len(ref_segment)]
        else:
            part_trim = partial_mean

        part_prof = Profile(
            depth_data=part_trim,
            pixel_size=pixel_size,
            cutoff_hi=partial_profile.cutoff_hi,
            cutoff_lo=partial_profile.cutoff_lo,
        )

        try:
            result = align_profiles_multiscale(ref_seg_prof, part_prof, params_adj)

            total_trans, total_scale = compute_cumulative_transform(result.transforms)

            aligned_length = len(result.reference_aligned)
            min_overlap = partial_length // 2

            print(f"\n  Candidate pos={pos}:")
            print(f"    Segment length: {len(ref_segment)}")
            print(f"    Transforms: {[(f't={t.translation:.4f}, s={t.scaling:.6f}') for t in result.transforms]}")
            print(f"    Total translation: {total_trans:.4f} samples = {total_trans * pixel_size * 1e6:.2f} μm")
            print(f"    Total scaling: {total_scale:.6f}")
            print(f"    Final correlation: {result.final_correlation:.6f}")
            print(f"    Aligned length: {aligned_length} (min_overlap: {min_overlap})")
            print(f"    pOverlap would be: {aligned_length / partial_length:.6f}")

            if aligned_length < min_overlap:
                print("    SKIPPED: aligned_length < min_overlap")
        except ValueError as e:
            print(f"\n  Candidate pos={pos}: FAILED - {e}")
