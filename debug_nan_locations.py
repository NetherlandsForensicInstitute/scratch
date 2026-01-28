"""Check NaN locations and their effect on fine alignment."""

import sys

sys.path.insert(0, "packages/scratch-core/src")

import numpy as np

base = "packages/scratch-core/tests/resources/profile_correlator/partial_with_nans"
ref_data = np.load(f"{base}/input_profile_ref.npy").ravel()
comp_data = np.load(f"{base}/input_profile_comp.npy").ravel()

print(f"ref: len={len(ref_data)}, NaN={np.sum(np.isnan(ref_data))}")
print(f"comp: len={len(comp_data)}, NaN={np.sum(np.isnan(comp_data))}")

ref_nan_idx = np.where(np.isnan(ref_data))[0]
comp_nan_idx = np.where(np.isnan(comp_data))[0]
print(f"\nref NaN positions: {ref_nan_idx}")
print(f"comp NaN positions: {comp_nan_idx}")

# At candidate 0: ref[0:400] and comp[0:400]
ref_seg0 = ref_data[0:400]
n_ref_nan_0 = np.sum(np.isnan(ref_seg0))
n_comp_nan = np.sum(np.isnan(comp_data))
combined_nan_0 = np.sum(np.isnan(ref_seg0) | np.isnan(comp_data))
print("\nCandidate 0: ref[0:400]")
print(f"  ref NaN in segment: {n_ref_nan_0}")
print(f"  comp NaN: {n_comp_nan}")
print(f"  Combined NaN (either): {combined_nan_0}")
print(f"  Valid pairs: {400 - combined_nan_0}")

# At candidate 150: ref[150:550] and comp[0:400]
ref_seg150 = ref_data[150:550]
n_ref_nan_150 = np.sum(np.isnan(ref_seg150))
combined_nan_150 = np.sum(np.isnan(ref_seg150) | np.isnan(comp_data))
print("\nCandidate 150: ref[150:550]")
print(f"  ref NaN in segment: {n_ref_nan_150}")
print(f"  comp NaN: {n_comp_nan}")
print(f"  Combined NaN (either): {combined_nan_150}")
print(f"  Valid pairs: {400 - combined_nan_150}")

# Check if NaN is at beginning or end (affects boundary behavior)
print(f"\nref NaN range: {ref_nan_idx.min()}-{ref_nan_idx.max()}")
print(f"comp NaN range: {comp_nan_idx.min()}-{comp_nan_idx.max()}")
