import numpy as np
from scipy.spatial import KDTree
from scipy.ndimage import binary_fill_holes
from typing import Dict, Optional, Union


def remove_holes_and_stitches(
    data_in: Union[np.ndarray, Dict],
    param: Optional[Dict] = None,
    mask: Optional[np.ndarray] = None,
) -> Union[np.ndarray, Dict]:
    """
    Prepares depth data for further processing by interpolation of invalid
    measurement values and interpolation of artifacts.

    param data_in: Data structure containing depth_data, or just the depth array
    param param: Parameters dictionary with optional keys:
        - recal: bool (default=True)
        - index_subsampling: int (default=5)
        - times_median: float (default=15)
        - median_filt_size: int (default=10)
        - save_intermediate_res: bool (default=False)
        - xdim: float (default=438e-9)
        - ydim: float (default=438e-9)
        - show_info: bool (default=True)
        - invalid_pixel_val: float (default=np.nan)
        - interpolate_data: bool (default=False)
    param mask: Binary matrix specifying foreground (1) and background (0)
    return: Preprocessed data (same type as input)
    """

    # Parse parameters
    if param is None:
        param = {}

    recal = param.get("recal", True)
    index_subsampling = param.get("index_subsampling", 5)
    times_median = param.get("times_median", 15)
    median_filt_size = param.get("median_filt_size", 10)
    _ = param.get("save_intermediate_res", False)
    xdim = param.get("xdim", 438e-9)
    ydim = param.get("ydim", 438e-9)
    show_info = param.get("show_info", True)
    _ = param.get("invalid_pixel_val", np.nan)
    interpolate_data = param.get("interpolate_data", False)

    # Handle optimize_bullet_ROI hack
    if "optimize_bullet_ROI" in param:
        interpolate_data = param["optimize_bullet_ROI"]

    times_median_backup = times_median

    # Extract depth data
    if isinstance(data_in, dict):
        depth_data = data_in["depth_data"].copy()
    else:
        depth_data = data_in.copy()

    # Create mask if not provided
    if mask is None:
        mask = np.ones(depth_data.shape, dtype=bool)
    else:
        mask = mask.astype(bool)

    # Adjust parameters based on data size
    size_mask = np.sum(mask)
    tmp = (size_mask / 25e6) ** 2
    index_subsampling_mod = max(1, int(np.round(index_subsampling * tmp)))

    median_filt_size_mod = median_filt_size / 2
    times_median = times_median * 6

    # Check if small strip
    is_small_strip = depth_data.shape[1] <= 20

    # --- Remove invalid points NOT connected to border ---
    if interpolate_data:
        depth_data_tmp = depth_data.copy()
        depth_data_tmp[~mask] = 0

        indices_invalid = np.where(np.isnan(depth_data_tmp))
        indices_invalid_flat = np.ravel_multi_index(indices_invalid, depth_data.shape)

        if len(indices_invalid_flat) > 0:
            mask_tmp = mask.copy().astype(float)
            mask_tmp[~mask] = np.nan
            indices_valid_flat = np.where(
                ~np.isnan(depth_data.ravel() * mask_tmp.ravel())
            )[0]

            test_image = np.abs(depth_data_tmp)
            test_image[indices_invalid] = 0
            test_image = test_image.astype(bool)

            # Find holes not connected to borders
            temp_5 = ~binary_fill_holes(test_image)
            indices_mask = (~test_image) & (~temp_5)
            indices_mask_flat = np.where(indices_mask.ravel())[0]

            if len(indices_mask_flat) > 0:
                # Create coordinate grid
                X, Y = np.meshgrid(
                    np.arange(depth_data.shape[1]), np.arange(depth_data.shape[0])
                )
                Y = ydim * Y
                X = xdim * X

                vertices = np.stack([X.ravel(), Y.ravel(), depth_data.ravel()], axis=1)

                # Subsample valid points for KD-tree
                tmp_indices = indices_valid_flat[::index_subsampling_mod]
                kd_tree = KDTree(vertices[tmp_indices, :2])

                # Determine number of nearest neighbors
                amt_NN = max(
                    10, int(np.round((len(indices_valid_flat) / 25e6) ** 2 * 50))
                )

                if show_info:
                    print("Filling holes and removing invalid points...")

                # Find nearest neighbors
                distances, idx50 = kd_tree.query(
                    vertices[indices_mask_flat, :2], k=amt_NN
                )

                # Get z-values from nearest neighbors and compute median
                fifty_NNs = vertices[tmp_indices[idx50], 2]
                vertices[indices_mask_flat, 2] = np.median(fifty_NNs, axis=1)

                depth_data = vertices[:, 2].reshape(depth_data.shape)

            mask[indices_invalid] = False
        else:
            if show_info:
                print("No holes and invalid points...")

    # --- Remove needles/stitches ---
    if median_filt_size_mod > 1:
        if not is_small_strip:
            # Determine subsampling factor
            subsamp_factor = max(1, int(np.ceil(70e-6 / median_filt_size_mod / xdim)))

            depth_data_tmp = depth_data * mask
            depth_data_tmp[~mask] = np.nan

            if subsamp_factor > 1:
                depth_data_sub = depth_data_tmp[::subsamp_factor, ::subsamp_factor]
            else:
                depth_data_sub = depth_data_tmp

            # Apply median filter (handles NaN)
            depth_data_sub_filt = median_filter_nan(
                depth_data_sub, int(median_filt_size_mod)
            )

            if subsamp_factor > 1:
                # Upsample filtered data
                from scipy.ndimage import zoom

                depth_data_sub_filt_res = zoom(
                    depth_data_sub_filt, subsamp_factor, order=1, mode="nearest"
                )
            else:
                depth_data_sub_filt_res = depth_data_sub_filt

            residual_image = (
                depth_data_tmp
                - depth_data_sub_filt_res[: depth_data.shape[0], : depth_data.shape[1]]
            )
        else:
            # Small strip - use 1D-like filtering
            median_filt_size_mod = max(2, int(np.round(np.sqrt(median_filt_size_mod))))
            depth_data_sub_filt = median_filter_nan(depth_data, median_filt_size_mod)
            residual_image = depth_data - depth_data_sub_filt

        # Find outliers based on residuals
        threshold = times_median * np.nanmedian(np.abs(residual_image))
        indices_valid_flat = np.where(np.abs(residual_image.ravel()) <= threshold)[0]
        indices_invalid_flat = np.where(np.abs(residual_image.ravel()) > threshold)[0]

        if len(indices_invalid_flat) > 0:
            if interpolate_data:
                # Create coordinate grid
                X, Y = np.meshgrid(
                    np.arange(depth_data.shape[1]), np.arange(depth_data.shape[0])
                )
                Y = ydim * Y
                X = xdim * X

                vertices = np.stack([X.ravel(), Y.ravel(), depth_data.ravel()], axis=1)

                # Subsample valid points
                tmp_indices = indices_valid_flat[::index_subsampling_mod]
                kd_tree = KDTree(vertices[tmp_indices, :2])

                # Determine number of nearest neighbors
                amt_NN = int(np.round((len(indices_valid_flat) / 25e6) ** 2 * 10))
                if is_small_strip:
                    amt_NN = max(3, int(np.round(np.sqrt(amt_NN))))
                else:
                    amt_NN = max(10, amt_NN)

                if show_info:
                    print("Removing stitches...")

                # Find nearest neighbors
                distances, idx10 = kd_tree.query(
                    vertices[indices_invalid_flat, :2], k=amt_NN
                )

                # Interpolate using median of nearest neighbors
                ten_NNs = vertices[tmp_indices[idx10], 2]
                vertices[indices_invalid_flat, 2] = np.median(ten_NNs, axis=1)

                depth_data = vertices[:, 2].reshape(depth_data.shape)
            else:
                depth_data.ravel()[indices_invalid_flat] = np.nan
        else:
            if show_info:
                print("No stitches...")

    # Prepare output
    if isinstance(data_in, dict):
        data_out = data_in.copy()
        data_out["depth_data"] = depth_data
        data_out["texture_data"] = None
        data_out["quality_data"] = None
        data_out["is_interp"] = interpolate_data

        if "data_param" not in data_out:
            data_out["data_param"] = {}

        if interpolate_data or median_filt_size_mod > 1:
            data_out["data_param"]["recal"] = recal
            data_out["data_param"]["index_subsampling"] = index_subsampling
            data_out["data_param"]["times_median"] = times_median_backup
            data_out["data_param"]["median_filt_size"] = median_filt_size

        return data_out
    else:
        return depth_data


def median_filter_nan(image: np.ndarray, filter_size: int) -> np.ndarray:
    """
    Apply median filter that handles NaN values.

    param image: Input image
    param filter_size: Size of the median filter kernel
    return: Filtered image
    """
    if filter_size % 2 == 0:
        filter_size += 1

    half_size = filter_size // 2

    # Pad image
    padded = np.pad(image, half_size, mode="constant", constant_values=np.nan)
    output = np.zeros_like(image)

    # Create all shifted versions
    shifts = []
    for i in range(-half_size, half_size + 1):
        for j in range(-half_size, half_size + 1):
            shifted = np.roll(np.roll(padded, i, axis=0), j, axis=1)
            shifts.append(shifted[half_size:-half_size, half_size:-half_size])

    # Stack and compute nanmedian
    stack = np.stack(shifts, axis=-1)
    output = np.nanmedian(stack, axis=-1)

    return output


# Example usage
if __name__ == "__main__":
    # Create synthetic depth data with holes and outliers
    depth_data = np.random.randn(100, 100) * 0.1 + 10.0

    # Add some NaN holes
    depth_data[20:25, 30:35] = np.nan

    # Add some outliers (needles/stitches)
    depth_data[50, 50] = 20.0
    depth_data[60, 60] = 5.0

    # Process the data
    param = {
        "interpolate_data": True,
        "show_info": True,
        "median_filt_size": 10,
        "times_median": 15,
    }

    result = remove_holes_and_stitches(depth_data, param)

    print(f"Original data shape: {depth_data.shape}")
    print(f"Processed data shape: {result.shape}")
    print(f"NaN count before: {np.sum(np.isnan(depth_data))}")
    print(f"NaN count after: {np.sum(np.isnan(result))}")
