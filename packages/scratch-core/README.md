# scratch-core

Core library for forensic ballistics analysis, developed at the Netherlands Forensic Institute (NFI).
Provides algorithms for comparing toolmarks on bullets (striations) and cartridge cases (impressions).

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

## Main Pipelines

### Striation marks (bullets)

`preprocess_striation_mark` prepares a 2D scan for correlation:
1. Highpass filter removes large-scale shape (curvature, tilt)
2. Lowpass filter removes high-frequency noise
3. Fine rotation aligns striations horizontally via gradient analysis
4. Extracts a 1D profile (column mean)

`correlate_profiles` compares two `Profile` objects via brute-force search over shifts and scale factors,
returning the alignment with maximum cross-correlation. The result (`StriationComparisonResults`) includes:

- `correlation_coefficient` — Pearson correlation of the aligned overlap
- `overlap_ratio` — overlap length relative to the shorter profile
- `scale_factor` — relative scale difference between marks
- `sa_ref/comp`, `sq_ref/comp` — ISO 25178 roughness of each overlap region
- `sa_diff`, `sq_diff` — roughness of the difference profile
- `ds_normalized_ref/comp/combined` — normalized signature differences (0 = identical)

### Impression marks (cartridge cases)

`preprocess_impression_mark` prepares a 2D scan for surface comparison:
1. Crop NaN borders, compute center
2. Optional tilt correction
3. Polynomial leveling (mean / plane / quadratic)
4. Band-pass filtering (antialiasing + lowpass + highpass)
5. Resample to target resolution

`compare_surfaces` runs the CMC (Congruent Matching Cells) pipeline on two preprocessed marks: resample to common
pixel size, place a grid of cells over the reference, coarse-register each cell against the comparison image over an
angle sweep, then classify cells as congruent based on consensus rotation and translation.

The result (`ComparisonResult`) includes:
- `cells` — per-cell registration result (`best_score`, `angle_deg`, `center_comparison`, `is_congruent`)
- `cmc_count` / `cmc_fraction` — number and fraction of congruent cells
- `cmc_area_fraction` — fraction of valid reference surface covered by congruent cells
- `consensus_rotation` / `consensus_translation` — estimated global alignment between marks

Use `ComparisonParams.for_mark_type(mark_type)` to construct parameters with the correct default cell size
for the given mark type.
