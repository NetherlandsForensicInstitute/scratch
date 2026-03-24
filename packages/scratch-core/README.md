# scratch-core

Core library for forensic ballistics analysis, developed at the Netherlands Forensic Institute (NFI).
Provides algorithms for comparing toolmarks on bullets (striations) and cartridge cases (impressions).

## Repository context

`scratch-core` is the algorithmic engine of the `scratch` repository. The repository also contains a
FastAPI service (`src/`) that exposes the core functionality as HTTP endpoints:

```
Raw scan file
    │
    ▼
POST /preprocessor/prepare_mark_{striation|impression}
    │   Parse scan, apply user mask/crop, run preprocessing pipeline
    │   → saves preprocessed mark files to a vault
    ▼
POST /processor/calculate_score_{striation|impression}
    │   Load preprocessed marks, run profile correlation or CMC comparison
    │   → returns score, per-cell results, and plot URLs
    ▼
POST /processor/calculate_lr_{striation|impression}
        Load score, apply LR system
        → returns log-LR with confidence interval
```

All algorithmic work (filtering, leveling, profile correlation, CMC) is implemented in `scratch-core`.
The API layer in `src/` handles file I/O, HTTP plumbing, and vault management.

## Package Structure

```
src/
├── container_models/       # Pydantic data models (ScanImage, etc.)
├── conversion/
│   ├── filter/             # Gaussian regression filters (ISO 16610)
│   ├── leveling/           # Polynomial surface fitting and leveling
│   ├── preprocess_striation/   # Striation mark preprocessing pipeline
│   ├── preprocess_impression/  # Impression mark preprocessing pipeline
│   ├── profile_correlator/     # Profile alignment and correlation
│   ├── surface_comparison/     # CMC-based cartridge case comparison
│   ├── plots/              # Visualization utilities
│   └── export/             # Profile export utilities
├── parsers/                # File format parsers (AL3D, X3P)
└── utils/                  # Shared utilities
```

## Data Model

The central container is `ScanImage`: a 2D height map with physical pixel scale (in meters) and a boolean
validity mask for NaN regions. It is wrapped in a `Mark`, which adds `MarkType`, metadata, and a crop
rectangle. All models use Pydantic with `frozen=True`.

Supported mark types (see `MarkType`):

| Category   | Types                                                                                      |
|------------|--------------------------------------------------------------------------------------------|
| Impression | breech face, chamber, ejector, extractor, firing pin                                       |
| Striation  | bullet LEA/GEA, chamber, ejector, ejector port, extractor, firing pin drag, aperture shear |

Input scans are parsed from AL3D or X3P files via `parsers/`.

## Filtering

All filtering is implemented as **Gaussian regression filtering** (ISO 16610-21), implemented in
`conversion/filter/`. This is currently the only filter type in the library.

### ISO 16610 Gaussian filter

The standard Gaussian uses the ISO formula `G(x) = exp(-π(x/(α·λc))²)` where `α = √(ln(2)/π)` gives
50% transmission at the cutoff wavelength `λc`. This differs from the `exp(-x²/(2σ²))` convention used
by scipy; the conversion is `σ = α·λc/√(2π)`. All cutoffs are specified in meters.

### NaN-aware filtering via normalized convolution

The filter handles missing data (NaN pixels from scan gaps or masks) using normalized convolution: NaN
pixels are assigned zero weight, so the kernel only accumulates over valid neighbors. For order-0 this
is implemented as FFT-based separable convolution over both the data and a binary weight map, then
dividing the two results pixelwise.

### Regression orders

`apply_gaussian_regression_filter` supports three regression orders:

| Order | Model fitted locally            | Use case                                    |
|-------|---------------------------------|---------------------------------------------|
| 0     | Weighted mean (Nadaraya-Watson) | Standard Gaussian smoothing                 |
| 1     | Plane (`c₀ + c₁x + c₂y`)       | Smoothing robust to local linear trends     |
| 2     | Quadratic surface (6 terms)     | Smoothing robust to local curvature         |

For order 0, the computation reduces to two separable FFT convolutions (fast). For orders 1 and 2, a
weighted least-squares system is solved per pixel using moment-sums precomputed via convolution (a 2D
Savitzky-Golay approach). Order 2 uses a slightly different `α` (`ALPHA_REGRESSION ≈ 0.731`) to
preserve the filter's frequency properties.

### 1D striation-preserving filter

`apply_striation_preserving_filter_1d` applies the Gaussian filter **only along the y-axis** (across
rows), leaving the x-direction (along striations) untouched. This preserves the striation signal while
removing cross-striation shape and noise. Border rows affected by edge effects (±σ pixels top/bottom)
are optionally cropped.

### High-pass and band-pass

Both 1D and 2D variants support `is_high_pass=True`, which returns `data - smoothed` (i.e. the
residual after subtracting the lowpass component). A band-pass is obtained by chaining two high-pass
filters sequentially: highpass at `λ_high` followed by lowpass at `λ_low`. This is mathematically
equivalent to a Difference of Gaussians (DoG) filter with parameters `t₁ = σ_high` and
`t₂ = √(σ_low² + σ_high²)`.

## Leveling

Leveling removes a fitted polynomial surface from a scan to eliminate large-scale form errors (tilt,
curvature). It is implemented in `conversion/leveling/` and used by the impression preprocessing pipeline.

`level_map(scan_image, terms)` fits the selected surface terms to all valid (non-NaN) pixels using
least squares, subtracts the fit, and returns a `LevelingResult` containing the residual map, the
fitted surface, and the RMS of the residuals.

### Surface terms

Terms are defined as a `Flag` enum and can be combined with `|`:

| Term        | Polynomial component  | Preset  |
|-------------|-----------------------|---------|
| `OFFSET`    | constant `c`          | PLANE   |
| `TILT_X`    | `ax`                  | PLANE   |
| `TILT_Y`    | `by`                  | PLANE   |
| `ASTIG_45`  | `cxy`                 |         |
| `DEFOCUS`   | `x² + y²`             |         |
| `ASTIG_0`   | `x² - y²`             |         |

`SurfaceTerms.PLANE` removes mean + tilt. `SurfaceTerms.SPHERE` removes mean + tilt + full quadratic
(covers defocus and both astigmatism orientations), effectively flattening the curved surface of a
cartridge case.

Coordinates are centered on the mark's physical center (or an explicit reference point) before fitting,
to keep the system numerically well-conditioned.

## Main Pipelines

### Striation marks (bullets)

```
     _____
    /|||||\
    \|||||/
     -----
```

`preprocess_striation_mark` prepares a 2D scan for correlation:

**1. Band-pass filtering (shape and noise removal)**

Two 1D Gaussian filters applied sequentially along the y-axis:
- Highpass at `λ_high` (default 250 μm): removes large-scale form — curvature, tilt, waviness.
- Lowpass at `λ_low` (default 5 μm): removes high-frequency noise.

The result passes only spatial frequencies between `λ_low` and `λ_high`, isolating the striation
signal. The two-filter sequence is equivalent to a DoG band-pass filter.

**2. Fine rotation via gradient analysis (iterative)**

Striations should be horizontal before profile extraction. Small angular deviations are corrected
iteratively:
1. Smooth the current data with a Gaussian kernel (σ ≈ 1.75 × 10⁻⁵ m in pixel units).
2. Compute the 2D image gradient (`∇f = (fx, fy)`).
3. For each pixel with gradient magnitude above 1.5× the median, compute the tilt angle
   `θ = arcsin(fx / |∇f|)`, sign-corrected so that the direction is consistent with `fy`.
4. Take the median of all `|θ| < 10°` as the detected misalignment angle.
5. Apply a **shear transform** (shift each column by `tan(θ) × column_index`) to bring striations
   horizontal, accumulate the total angle, and repeat until `|θ| < 0.1°`.

This uses a shear (not a rotation) to preserve pixel scale and avoid interpolation artefacts across
the full image width.

**3. Profile extraction**

After alignment, the mean (or optionally median) is taken across columns (y-direction) to collapse
the 2D image to a 1D `Profile`. The profile inherits the x pixel size of the aligned image.

The output is a `(Mark, Profile)` pair. The `Mark` contains the aligned 2D data; the `Profile` is
used directly for correlation.

---

### Impression marks (cartridge cases)

```
    .-----.
   /   o   \
   \   o   /
    '-----'
```

`preprocess_impression_mark` prepares a 2D scan for surface comparison. The pipeline has 8 stages:

1. **Crop NaN borders** — remove rows/columns that are entirely invalid, compute the physical center
   of the valid region.
2. **Tilt correction** (optional) — adjust for mechanical tilt of the scan stage by fitting a plane
   to the surface and correcting the pixel spacing accordingly.
3. **Initial leveling** — fit and subtract `SurfaceTerms.SPHERE` (or configured terms) to remove
   large-scale form. The fitted surface is saved to re-add later for the leveled-only output.
4. **Anti-aliasing filter** — if the image will be downsampled by more than 1.5×, apply a 2D
   Gaussian lowpass at the target pixel size to prevent aliasing.
5. **Additional lowpass filter** — if a `lowpass_cutoff` is configured and is finer than the
   anti-aliasing cutoff, apply a separate lowpass filter (using the configured regression order).
6. **Resample** — bilinear resampling to the target pixel size (default: the scan's native resolution,
   optionally coarser for efficiency).
7. **Highpass filter** — 2D Gaussian highpass at `highpass_cutoff` removes low-frequency residuals.
8. **Final leveling** — apply the same polynomial leveling again (using the original mark center as
   reference point) to remove any residual tilt introduced by filtering and resampling.

The function returns two marks: the fully filtered mark (used for CMC comparison) and a leveled-only
mark (the mark after stage 3 + resampling + PLANE leveling, without the highpass filter — used for
visualization and roughness analysis).

---

### Profile correlation (striation score)

```
   ___/^^^\____/^\___
      ___/^^^\____/^\
```

`correlate_profiles` compares two `Profile` objects using a brute-force search over all shifts and
scale factors.

**Algorithm:**

1. **Equalize pixel scales** — if the two profiles have different pixel sizes, the higher-resolution
   one is downsampled to match the coarser one. This normalizes the search to a common sample grid.

2. **Scale factor sweep** — generate `n_scale_steps` (default 7) scale factors linearly spaced in
   `[1 - max_scaling, 1 + max_scaling]` (default ±5%), plus their reciprocals, to make the search
   symmetric with respect to which profile is reference vs. comparison.

3. **Shift sweep** — for each scale factor, resample the comparison profile at that scale, then try
   every integer shift that maintains at least `min_overlap_distance` (default 350 μm) of overlap.
   At each shift/scale combination, compute the Pearson correlation of the overlapping segments.

4. **Select maximum** — the shift and scale with the highest correlation define the best alignment.
   This is a global search: it will find the maximum even for periodic patterns where multiple
   near-equal peaks exist.

**Output metrics** (`StriationComparisonResults`):

| Field                              | Description                                                |
|------------------------------------|------------------------------------------------------------|
| `correlation_coefficient`          | Pearson correlation of the aligned overlap (primary score) |
| `overlap_ratio`                    | Overlap length / length of the shorter profile             |
| `scale_factor`                     | Relative scale difference between the two marks            |
| `sa_ref`, `sa_comp`                | ISO 25178 mean absolute roughness of each overlap region   |
| `sq_ref`, `sq_comp`                | RMS roughness of each overlap region                       |
| `sa_diff`, `sq_diff`               | Roughness of the difference profile (ref − comp)           |
| `ds_normalized_ref/comp/combined`  | Normalized signature differences (0 = identical, 1 = max) |

---

### Surface comparison — CMC (impression score)

```
  +--+--+--+
  |##|##|  |
  +--+--+--+
  |  |##|##|
  +--+--+--+
```

`compare_surfaces` runs the **Congruent Matching Cells** (CMC) pipeline on two preprocessed marks.

**Algorithm:**

1. **Resample** — resample the comparison image to the reference pixel size so both share a common
   coordinate grid.

2. **Generate cell grid** — place a regular grid of square cells (size defined by `ComparisonParams`,
   e.g. 450 × 450 μm for breech face) over the reference image. Cells with fewer than
   `minimum_fill_fraction` (default 35%) valid pixels are discarded.

3. **Coarse registration** — for each reference cell, search for the best-matching patch in the
   comparison image over a rotation sweep (default: −180° to +180° in 5° steps). At each angle:
   - Rotate the comparison image by that angle.
   - Find the translation that maximizes the ACCF (area-corrected cross-correlation function) between
     the reference cell and the rotated comparison.
   - Record the best score, angle, and translation across the full sweep.

4. **Fine registration** — currently a pass-through stub for future sub-pixel refinement.

5. **CMC classification** — determine which cells are "congruent" (i.e., consistently registered):
   - **Consensus angle**: compute a circular median of all cell registration angles. Apply the
     generalized ESD test (Rosner 1983) to reject statistical outliers, then tighten the inlier set
     to cells within 2× the `angle_deviation_threshold`. Recompute the median from inliers.
   - **Consensus translation**: rotate reference cell centers by the consensus angle, then take the
     component-wise median of (comparison center − expected position) over non-outlier cells.
   - **Congruence label**: a cell is congruent if it simultaneously satisfies all four criteria:
     `best_score ≥ correlation_threshold`, not an angle outlier, `|residual_angle| ≤ angle_threshold`,
     and both position error components `≤ position_threshold`.

**Output** (`ComparisonResult`):

| Field                      | Description                                             |
|----------------------------|---------------------------------------------------------|
| `cells`                    | Per-cell results: score, angle, positions, `is_congruent` |
| `cmc_count`                | Number of congruent cells (primary score)               |
| `cmc_fraction`             | `cmc_count / total_cells`                               |
| `cmc_area_fraction`        | Fraction of valid reference surface in congruent cells  |
| `consensus_rotation`       | Estimated global rotation between the two marks (°)     |
| `consensus_translation`    | Estimated global translation between the two marks (m)  |

Use `ComparisonParams.for_mark_type(mark_type)` to get the correct default cell size for the mark type.
Default cell sizes: 450 × 450 μm for breech face, 125 × 125 μm for all other impression types.
