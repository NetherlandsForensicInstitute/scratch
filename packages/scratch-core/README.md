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

## Main Pipelines

### Striation marks (bullets)

`preprocess_striation_mark` prepares a 2D scan for correlation:
1. Highpass filter removes large-scale shape (curvature, tilt)
2. Lowpass filter removes high-frequency noise
3. Fine rotation aligns striations horizontally via gradient analysis
4. Extracts a 1D profile (column mean)

`correlate_profiles` compares two profiles via brute-force search over shifts and scale factors, returning the
alignment with maximum cross-correlation along with roughness metrics (Sa, Sq) and overlap ratio.

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
