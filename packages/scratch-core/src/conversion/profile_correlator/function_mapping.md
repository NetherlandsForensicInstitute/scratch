# MATLAB to Python Function Mapping

This document maps all MATLAB functions to their Python equivalents in the ProfileCorrelator translation.

## Core Functions

| MATLAB Function | Python Function | Location | Status |
|----------------|-----------------|----------|--------|
| `ProfileCorrelatorSingle.m` | `profile_correlator_single()` | profile_correlator.py:1440 | ✓ Complete |
| `ProfileCorrelatorResInit.m` | `profile_correlator_res_init()` | profile_correlator.py:35 | ✓ Complete |
| `GetParamValue.m` | `get_param_value()` | profile_correlator.py:18 | ✓ Complete |
| `GetAliconaSampling.m` | `get_alicona_sampling()` | profile_correlator.py:28 | ✓ Complete |

## Data Processing Functions

| MATLAB Function | Python Function | Location | Status |
|----------------|-----------------|----------|--------|
| `EqualizeSamplingDistance.m` | `equalize_sampling_distance()` | profile_correlator.py:319 | ✓ Complete |
| `MakeDatasetLengthEqual.m` | `make_dataset_length_equal()` | profile_correlator.py:410 | ✓ Complete |
| `RemoveBoundaryZeros.m` | `remove_boundary_zeros()` | profile_correlator.py:223 | ✓ Complete |

## Transformation Functions

| MATLAB Function | Python Function | Location | Status |
|----------------|-----------------|----------|--------|
| `TranslateScalePointset.m` | `translate_scale_pointset()` | profile_correlator.py:130 | ✓ Complete |

## Similarity and Scoring

| MATLAB Function | Python Function | Location | Status |
|----------------|-----------------|----------|--------|
| `GetSimilarityScore.m` | `get_similarity_score()` | profile_correlator.py:79 | ✓ Complete |
| `GetStriatedMarkComparisonResults.m` | `get_striated_mark_comparison_results()` | profile_correlator.py:651 | ✓ Complete |

## Optimization Functions

| MATLAB Function | Python Function | Location | Status |
|----------------|-----------------|----------|--------|
| `ErrorfuncForInterProfileAlignment.m` | `errorfunc_for_inter_profile_alignment()` | profile_correlator.py:456 | ✓ Complete |
| `fminsearchbnd.m` | `FminSearchBnd.optimize()` | profile_correlator.py:476 | ✓ Complete |

## Alignment Functions

| MATLAB Function | Python Function | Location | Status |
|----------------|-----------------|----------|--------|
| `AlignInterProfilesMultiScale.m` | `align_inter_profiles_multi_scale()` | profile_correlator.py:708 | ⚠️ Missing filters |
| `AlignInterProfilesPartialMultiScale.m` | `align_inter_profiles_partial_multi_scale()` | profile_correlator.py:1019 | ⚠️ Missing functions |

## Missing Functions (Need Implementation)

These functions are called in the MATLAB code but are NOT yet translated. The current Python code includes placeholders or simplified implementations:

### Required for Full Functionality

| MATLAB Function | Purpose | Called By | Impact if Missing |
|----------------|---------|-----------|-------------------|
| `ApplyLowPassFilter.m` | Low-pass filtering of profiles | align_inter_profiles_multi_scale | Profiles not filtered at each scale, affects alignment quality |
| `RemoveNoiseGaussian.m` | Gaussian noise removal | determine_match_candidates_multi_scale | Candidate detection less accurate |
| `RemoveShapeGaussian.m` | Gaussian shape removal | determine_match_candidates_multi_scale | Candidate detection less accurate |
| `DetermineMatchCandidatesMultiScale.m` | Find match candidates for partial profiles | align_inter_profiles_partial_multi_scale | Partial profile alignment uses brute force instead |

### Notes on Missing Functions

1. **ApplyLowPassFilter**:
   - Used in the multi-scale loop to filter profiles at each wavelength
   - Current Python code uses unfiltered profiles
   - This is the **most critical missing function**
   - Affects alignment accuracy significantly

2. **RemoveNoiseGaussian**:
   - Removes noise at specific wavelengths for shape scales
   - Used to find coarse matching positions
   - Current Python code skips this preprocessing

3. **RemoveShapeGaussian**:
   - Removes shape information, keeps texture
   - Used for fine matching at comparison scale
   - Current Python code skips this preprocessing

4. **DetermineMatchCandidatesMultiScale**:
   - Finds candidate positions for partial profile matching
   - Current Python code uses simple brute-force sliding window
   - Less efficient but functionally similar

## Implementation Details

### Resampling

**MATLAB**: `resample()` function from Signal Processing Toolbox
**Python**: `scipy.signal.resample()`

Both use Fourier-based resampling. Minor numerical differences possible.

### Interpolation

**MATLAB**: `interp1(..., 'linear', 0)` - linear interpolation with 0 for out-of-bounds
**Python**: `scipy.interpolate.interp1d(..., kind='linear', bounds_error=False, fill_value=0)`

### Optimization

**MATLAB**: `fminsearch()` and custom `fminsearchbnd()`
**Python**: `scipy.optimize.minimize()` with Nelder-Mead method

The Python `FminSearchBnd` class implements variable transformation for bounds similar to MATLAB's fminsearchbnd.

### Array Indexing

**MATLAB**: 1-based indexing
**Python**: 0-based indexing

All array indices have been carefully converted. For example:
- MATLAB: `array(1:end)` → Python: `array[0:]`
- MATLAB: `array(start:end)` → Python: `array[start-1:end]`

## Data Structure Conversion

### Profile Structures

**MATLAB** (struct):
```matlab
profile.depth_data = [N x 1] array
profile.xdim = scalar
profile.ydim = scalar
profile.cutoff_hi = scalar (optional)
profile.cutoff_lo = scalar (optional)
profile.LR = scalar (optional)
```

**Python** (dict):
```python
profile = {
    'depth_data': np.array([N,1] or [N,]),
    'xdim': float,
    'ydim': float,
    'cutoff_hi': float (optional),
    'cutoff_lo': float (optional),
    'LR': float (optional)
}
```

### Results Structure

**MATLAB** (struct):
```matlab
results.bProfile = scalar
results.ccf = scalar
results.dPos = scalar
... (29 fields total)
```

**Python** (dict):
```python
results = {
    'bProfile': int/float,
    'ccf': float,
    'dPos': float,
    ... (29 keys total)
}
```

## Testing

### Test Data Format

Test data is stored in numpy .npy format:
- `input_profile_ref.npy`: 1D array of 500 depth measurements
- `input_profile_comp.npy`: 1D array of 500 depth measurements
- `metadata.json`: Parameters and expected MATLAB output

### Running Tests

```bash
python test_profile_correlator.py
```

Expected output:
- Comparison of each results field
- Match/mismatch status
- Numerical differences

### Known Test Limitations

Without the filtering functions, test results will differ from MATLAB in:
- `dPos`, `dScale`: Registration parameters
- `ccf`, `simVal`: Correlation values
- `sa_*`, `sq_*`: Topographic measurements
- `lOverlap`, `pOverlap`: Overlap metrics

## Performance Considerations

### Memory Usage

Python implementation uses similar memory patterns to MATLAB:
- Profile data stored as numpy arrays
- Intermediate results stored in dictionaries
- No significant memory overhead

### Speed

Relative performance compared to MATLAB:
- **Similar**: Core numerical operations (numpy is similar to MATLAB)
- **Faster**: No MATLAB JIT warm-up needed
- **Slower**: Python loops (use vectorization where possible)
- **Unknown**: scipy.optimize vs MATLAB fminsearch

## Future Improvements

1. **Implement Missing Filters**:
   - Translate ApplyLowPassFilter, RemoveNoiseGaussian, RemoveShapeGaussian
   - Or create scipy-based equivalents

2. **Optimize Performance**:
   - Vectorize remaining loops
   - Consider numba JIT compilation for hot paths
   - Profile code to identify bottlenecks

3. **Add Visualization**:
   - Plotting of aligned profiles
   - Correlation plots
   - Multi-scale visualization

4. **Extend Functionality**:
   - Support for 2D images (currently 1D profiles only)
   - Additional similarity metrics
   - Parallel processing for multiple comparisons

## Version History

- **v1.0** (2026-01): Initial Python translation
  - Core functions translated
  - Basic testing implemented
  - Documentation created
  - Missing: filtering functions

## References

Original MATLAB code:
- Author: Martin Baiker-Soerensen
- Institution: NFI (Netherlands Forensic Institute)
- Date: March 2021, updated April 2021
- Version: 1.0
