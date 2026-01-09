"""
Test to verify Python implementation matches MATLAB ground truth.
"""

import numpy as np
from scipy.io import loadmat


def test_form_noise_removal_matches_matlab():
    """
    Test that Python preprocessing exactly matches MATLAB output.

    Requires MATLAB test files:
    - 'matlab_input.mat': depth_data, xdim, mask (optional)
    - 'matlab_output.mat': depth_data_out, mask_out

    Generate with the MATLAB export script below.
    """
    from preprocess_data import apply_form_noise_removal

    # Load MATLAB input
    matlab_input = loadmat("matlab_input.mat")
    depth_data = matlab_input["depth_data"]
    xdim = float(matlab_input["xdim"][0, 0])
    mask = matlab_input.get("mask", None)
    if mask is not None:
        mask = mask.astype(bool)

    # Load MATLAB output (ground truth)
    matlab_output = loadmat("matlab_output.mat")
    matlab_result = matlab_output["depth_data_out"]
    matlab_mask = matlab_output["mask_out"].astype(bool)

    # Parameters from MATLAB
    cutoff_hi = float(matlab_input.get("cutoff_hi", [[2000e-6]])[0, 0])
    cutoff_lo = float(matlab_input.get("cutoff_lo", [[250e-6]])[0, 0])

    # Run Python implementation
    python_result, python_mask = apply_form_noise_removal(
        depth_data=depth_data,
        xdim=xdim,
        cutoff_hi=cutoff_hi,
        cutoff_lo=cutoff_lo,
        mask=mask,
        cut_borders_after_smoothing=True,
    )

    # Compare results - allow small numerical error
    np.testing.assert_allclose(
        python_result,
        matlab_result,
        rtol=1e-5,
        atol=1e-8,
        err_msg="Python output doesn't match MATLAB ground truth",
    )

    np.testing.assert_array_equal(python_mask, matlab_mask, err_msg="Masks don't match")

    print("✅ Python matches MATLAB perfectly!")


def test_synthetic_form_noise_removal():
    """
    Test on synthetic data where we know the ground truth.

    Verifies that:
    - Large-scale form is removed
    - Striations are preserved
    - High-frequency noise is removed
    """
    from preprocess_data import apply_form_noise_removal

    # Create synthetic striated surface
    height, width = 200, 150
    xdim = 1e-6  # 1 µm spacing

    x = np.arange(height) * xdim
    y = np.arange(width) * xdim
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Components
    form = 5e-6 * (X / x.max()) ** 2  # Curvature (large wavelength)
    striations = 0.5e-6 * np.sin(2 * np.pi * X / 500e-6)  # 500 µm striations
    noise = 0.1e-6 * np.random.randn(height, width)  # High-freq noise

    depth_data = form + striations + noise

    # Apply preprocessing
    result, mask = apply_form_noise_removal(
        depth_data=depth_data,
        xdim=xdim,
        cutoff_hi=2000e-6,  # Remove > 2000 µm (form)
        cutoff_lo=250e-6,  # Remove < 250 µm (noise)
    )

    # Verify form removed (mean near zero)
    assert np.abs(np.mean(result)) < 1e-7, "Form not removed"

    # Verify striations preserved
    std_result = np.std(result)
    assert std_result > 0.1e-6, "Striations lost"

    # Verify noise reduced
    std_original = np.std(depth_data)
    assert std_result < std_original * 0.5, "Noise not reduced"

    print("✅ Synthetic test passed!")
    print(f"   Noise reduction: {std_original * 1e6:.3f} → {std_result * 1e6:.3f} µm")


if __name__ == "__main__":
    print("\n=== Synthetic Data Test (no MATLAB needed) ===")
    test_synthetic_form_noise_removal()

    print("\n=== MATLAB Comparison Test ===")
    print("Generate MATLAB test files with this script:\n")
    print("""
% MATLAB: Generate test data for Python validation
data_in = <your_data_structure>;  % Load your real data
param.cutoff_hi = 2000e-6;
param.cutoff_lo = 250e-6;
param.cut_borders_after_smoothing = 1;
mask = ones(size(data_in.depth_data));

% Run Step 2 only (lines 156-192 of PreprocessData.m)
sigma = ChebyCutoffToGaussSigma(param.cutoff_hi, data_in.xdim);
if 2 * sigma > size(data_in.depth_data, 1) * 0.2
    param.cut_borders_after_smoothing = 0;
end

[data_no_shape, ~, mask_shape] = RemoveShapeGaussian(data_in, param, mask);
[data_no_noise, ~, mask_out] = RemoveNoiseGaussian(data_no_shape, param, mask_shape);

% Save for Python
save('matlab_input.mat', 'data_in', 'mask', 'param', '-v7.3');
depth_data = data_in.depth_data;
xdim = data_in.xdim;
cutoff_hi = param.cutoff_hi;
cutoff_lo = param.cutoff_lo;
save('matlab_input.mat', 'depth_data', 'xdim', 'mask', 'cutoff_hi', 'cutoff_lo');

depth_data_out = data_no_noise.depth_data;
save('matlab_output.mat', 'depth_data_out', 'mask_out');
    """)
    print("\nThen run: pytest test_preprocessing.py")
