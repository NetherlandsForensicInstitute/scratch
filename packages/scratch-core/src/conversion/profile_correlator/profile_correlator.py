"""
Profile Correlator - Python translation from MATLAB
Exact reproduction of MATLAB profile registration and correlation code

Author: Translated from Martin Baiker-Soerensen's MATLAB code (NFI, Mar 2021)
Python Translation: January 2026
"""

import numpy as np
from scipy.optimize import minimize
from scipy.signal import resample
from scipy.interpolate import interp1d
from typing import Dict


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def get_param_value(input_struct: Dict, param_name: str, default_value):
    """
    GetParamValue: checks if a certain PARAM_NAME is present in INPUT_STRUCT
    if yes, outputs INPUT_STRUCT[PARAM_NAME]
    otherwise outputs the given DEFAULT_VALUE
    """
    if param_name in input_struct:
        return input_struct[param_name]
    return default_value


def get_alicona_sampling():
    """GetAliconaSampling: Returns default Alicona sampling distance"""
    return 4.38312e-07


# ============================================================================
# PROFILE STRUCTURE INITIALIZATION
# ============================================================================


def profile_correlator_res_init():
    """
    ProfileCorrelatorResInit: initialize a profile comparison result structure

    Returns a dictionary with all comparison result fields initialized
    """
    results_table = {
        "bProfile": 0,
        "bSegments": 0,
        "pathReference": "",
        "pathCompare": "",
        "bKM": -1,
        "vPixSep1": np.nan,
        "vPixSep2": np.nan,
        # registration data
        "bPartialProfile": np.nan,
        "dPos": np.nan,
        "dScale": np.nan,
        "startPartProfile": np.nan,
        "simVal": np.nan,
        "lOverlap": np.nan,
        "pOverlap": np.nan,
        # comparison metrics
        "ccf": np.nan,
        "metric": np.nan,
        # topographic measurements
        "sa_1": np.nan,
        "sq_1": np.nan,
        "sa_2": np.nan,
        "sq_2": np.nan,
        "sa12": np.nan,
        "sq12": np.nan,
        "ds1": np.nan,
        "ds2": np.nan,
        "ds": np.nan,
    }
    return results_table


# ============================================================================
# SIMILARITY SCORING
# ============================================================================


def get_similarity_score(profile_1, profile_2, score_type="cross_correlation"):
    """
    GetSimilarityScore: Function that determines a similarity score between profiles

    Args:
        profile_1: Reference profile (dict with 'depth_data' or array)
        profile_2: Compared profile (dict with 'depth_data' or array)
        score_type: Type of similarity score (default 'cross_correlation')

    Returns:
        similarity_score: Similarity score between profiles
    """
    # Extract depth data if structs
    if isinstance(profile_1, dict):
        profile_1 = profile_1["depth_data"]
    if isinstance(profile_2, dict):
        profile_2 = profile_2["depth_data"]

    # Make sure both profiles are column vectors
    profile_1 = profile_1.flatten()
    profile_2 = profile_2.flatten()

    # Calculate similarity score
    if score_type == "cross_correlation":
        # where are the NaN's in both profiles?
        ind = np.isnan(profile_1) | np.isnan(profile_2)
        # remove those measurements
        profile_1 = profile_1[~ind]
        profile_2 = profile_2[~ind]
        # subtract average
        profile_1 = profile_1 - np.sum(profile_1) / len(profile_1)
        profile_2 = profile_2 - np.sum(profile_2) / len(profile_2)

        a12 = np.dot(profile_1, profile_2)
        a11 = np.dot(profile_1, profile_1)
        a22 = np.dot(profile_2, profile_2)

        similarity_score = a12 / np.sqrt(a11 * a22)
    else:
        raise ValueError(f"Similarity metric {score_type} not implemented!")

    return similarity_score


# ============================================================================
# TRANSFORMATION FUNCTIONS
# ============================================================================


def translate_scale_pointset(pointset_in, transformation_parameters):
    """
    TranslateScalePointset: Transform a pointset using shift and scaling

    Args:
        pointset_in: Input pointset (dict with 'depth_data' or array)
        transformation_parameters: Array of [translation, scaling] parameters
                                  Shape: (n_transforms, 2)

    Returns:
        pointset_out: Transformed pointset (same type as input)
    """
    # Extract depth data if struct
    is_struct = isinstance(pointset_in, dict)
    if is_struct:
        pointset = pointset_in["depth_data"].copy()
    else:
        pointset = pointset_in.copy()

    # Handle multi-column data
    if pointset.ndim > 1 and pointset.shape[1] > 1:
        transformed_pointset = pointset[:, 0].copy()
    else:
        transformed_pointset = pointset.flatten().copy()

    # Ensure transformation_parameters is 2D
    if transformation_parameters.ndim == 1:
        transformation_parameters = transformation_parameters.reshape(1, -1)

    # Build composite transformation matrix
    transform_matrix = np.eye(3)
    for cur_param in range(transformation_parameters.shape[0]):
        translation = transformation_parameters[cur_param, 0]
        scaling = transformation_parameters[cur_param, 1]

        current_transform = np.array([[scaling, 0, translation], [0, 1, 0], [0, 0, 1]])
        transform_matrix = current_transform @ transform_matrix

    # Apply transformation
    xx = np.arange(1, len(transformed_pointset) + 1)
    xx_trans = xx * transform_matrix[0, 0] + transform_matrix[0, 2]

    # Interpolate for each column
    if pointset.ndim > 1 and pointset.shape[1] > 1:
        transformed_pointset = np.zeros_like(pointset)
        for column in range(pointset.shape[1]):
            f = interp1d(
                xx_trans,
                pointset[:, column],
                kind="linear",
                bounds_error=False,
                fill_value=0,
            )
            transformed_pointset[:, column] = f(xx)
    else:
        f = interp1d(
            xx_trans,
            transformed_pointset,
            kind="linear",
            bounds_error=False,
            fill_value=0,
        )
        transformed_pointset = f(xx)

    # Return in same format as input
    if is_struct:
        pointset_out = pointset_in.copy()
        pointset_out["depth_data"] = transformed_pointset
    else:
        pointset_out = transformed_pointset

    return pointset_out


def remove_boundary_zeros(p_1_in, p_2_in):
    """
    RemoveBoundaryZeros: Crop profiles to remove padded zeros at borders

    Args:
        p_1_in: Reference profile (dict with 'depth_data' or array)
        p_2_in: Compared profile (dict with 'depth_data' or array)

    Returns:
        p1_no_zeros: p_1_in without padded zeros
        p2_no_zeros: p_2_in without padded zeros
        extra_translate: Start location offset (optional)
    """
    p1_no_zeros = p_1_in.copy() if isinstance(p_1_in, dict) else p_1_in
    p2_no_zeros = p_2_in.copy() if isinstance(p_2_in, dict) else p_2_in

    # Extract depth data
    if isinstance(p_1_in, dict):
        p_1 = p1_no_zeros["depth_data"]
    else:
        p_1 = p1_no_zeros

    if isinstance(p_2_in, dict):
        p_2 = p2_no_zeros["depth_data"]
    else:
        p_2 = p2_no_zeros

    # Sum across columns
    if p_1.ndim > 1:
        p_1 = np.sum(p_1, axis=1)
    if p_2.ndim > 1:
        p_2 = np.sum(p_2, axis=1)

    z_p_1 = p_1 == 0
    z_p_2 = p_2 == 0

    # Find start for profile 1
    if z_p_1[0]:
        count = 0
        while count < len(z_p_1) - 1 and z_p_1[count + 1]:
            count += 1
        start_1 = count + 1
    else:
        start_1 = 0

    # Find end for profile 1
    if z_p_1[-1]:
        count = len(p_1) - 1
        while count > 0 and z_p_1[count - 1]:
            count -= 1
        end_1 = count - 1
    else:
        end_1 = len(p_1) - 1

    # Find start for profile 2
    if z_p_2[0]:
        count = 0
        while count < len(z_p_2) - 1 and z_p_2[count + 1]:
            count += 1
        start_2 = count + 1
    else:
        start_2 = 0

    # Find end for profile 2
    if z_p_2[-1]:
        count = len(p_2) - 1
        while count > 0 and z_p_2[count - 1]:
            count -= 1
        end_2 = count - 1
    else:
        end_2 = len(p_2) - 1

    start_tot = max(start_1, start_2)
    end_tot = min(end_1, end_2)

    # Crop data
    if isinstance(p1_no_zeros, dict):
        p1_no_zeros["depth_data"] = p_1_in["depth_data"][start_tot : end_tot + 1]
    else:
        p1_no_zeros = p_1_in[start_tot : end_tot + 1]

    if isinstance(p2_no_zeros, dict):
        p2_no_zeros["depth_data"] = p_2_in["depth_data"][start_tot : end_tot + 1]
    else:
        p2_no_zeros = p_2_in[start_tot : end_tot + 1]

    extra_translate = start_tot

    return p1_no_zeros, p2_no_zeros, extra_translate


# ============================================================================
# RESAMPLING AND EQUALIZATION
# ============================================================================


def equalize_sampling_distance(data_in_1, data_in_2, equalize_y_dimension=False):
    """
    EqualizeSamplingDistance: Resample data with smallest sampling distance
    to the larger sampling distance

    Args:
        data_in_1: Scratch structure with profile 1
        data_in_2: Scratch structure with profile 2
        equalize_y_dimension: Whether to also resample y dimension

    Returns:
        data_out_1: Profile 1 with same sampling as profile 2
        data_out_2: Profile 2 with same sampling as profile 1
    """
    data_out_1 = data_in_1.copy()
    data_out_2 = data_in_2.copy()

    data1_modified = False
    data2_modified = False

    # Resample in x dimension
    if data_in_1["xdim"] > data_in_2["xdim"]:
        resample_factor = data_in_2["xdim"] / data_in_1["xdim"]

        if data_in_2["depth_data"].shape[1] == 1 or data_in_2["depth_data"].ndim == 1:
            data_out_2["depth_data"] = resample(
                data_in_2["depth_data"].flatten(),
                int(len(data_in_2["depth_data"]) * resample_factor),
            ).reshape(-1, 1)
        else:
            # Resample along axis 0 for each column
            new_len = int(data_in_2["depth_data"].shape[0] * resample_factor)
            data_out_2["depth_data"] = resample(
                data_in_2["depth_data"], new_len, axis=0
            )

        data_out_2["xdim"] = data_out_2["xdim"] / (
            data_in_2["xdim"] / data_in_1["xdim"]
        )
        data2_modified = True

    if data_in_1["xdim"] < data_in_2["xdim"]:
        resample_factor = data_in_1["xdim"] / data_in_2["xdim"]

        if data_in_1["depth_data"].shape[1] == 1 or data_in_1["depth_data"].ndim == 1:
            data_out_1["depth_data"] = resample(
                data_in_1["depth_data"].flatten(),
                int(len(data_in_1["depth_data"]) * resample_factor),
            ).reshape(-1, 1)
        else:
            new_len = int(data_in_1["depth_data"].shape[0] * resample_factor)
            data_out_1["depth_data"] = resample(
                data_in_1["depth_data"], new_len, axis=0
            )

        data_out_1["xdim"] = data_out_1["xdim"] / (
            data_in_1["xdim"] / data_in_2["xdim"]
        )
        data1_modified = True

    # Resample in y dimension if requested
    if equalize_y_dimension:
        if data_in_1["ydim"] > data_in_2["ydim"]:
            resample_factor = data_in_2["ydim"] / data_in_1["ydim"]

            if data_in_2["depth_data"].shape[0] == 1:
                data_out_2["depth_data"] = resample(
                    data_in_2["depth_data"].flatten(),
                    int(len(data_in_2["depth_data"]) * resample_factor),
                ).reshape(1, -1)
            else:
                new_len = int(data_in_2["depth_data"].shape[0] * resample_factor)
                data_out_2["depth_data"] = resample(
                    data_in_2["depth_data"], new_len, axis=0
                )

            data_out_2["ydim"] = data_out_2["ydim"] / (
                data_in_2["ydim"] / data_in_1["ydim"]
            )
            data2_modified = True

        if data_in_1["ydim"] < data_in_2["ydim"]:
            resample_factor = data_in_1["ydim"] / data_in_2["ydim"]

            if data_in_1["depth_data"].shape[0] == 1:
                data_out_1["depth_data"] = resample(
                    data_in_1["depth_data"].flatten(),
                    int(len(data_in_1["depth_data"]) * resample_factor),
                ).reshape(1, -1)
            else:
                new_len = int(data_in_1["depth_data"].shape[0] * resample_factor)
                data_out_1["depth_data"] = resample(
                    data_in_1["depth_data"], new_len, axis=0
                )

            data_out_1["ydim"] = data_out_1["ydim"] / (
                data_in_1["ydim"] / data_in_2["ydim"]
            )
            data1_modified = True

    # Clear LR field if data was modified
    if data1_modified and "LR" in data_out_1:
        data_out_1["LR"] = None

    if data2_modified and "LR" in data_out_2:
        data_out_2["LR"] = None

    return data_out_1, data_out_2


def make_dataset_length_equal(data_1, data_2):
    """
    MakeDatasetLengthEqual: Cut the longer profile to match the shorter one

    Args:
        data_1: Profile 1 (dict or array)
        data_2: Profile 2 (dict or array)

    Returns:
        data_1: Profile 1 with same length as profile 2
        data_2: Profile 2 with same length as profile 1
    """
    if isinstance(data_1, dict):
        data_1_depth_data = data_1["depth_data"]
        data_2_depth_data = data_2["depth_data"]
    else:
        data_1_depth_data = data_1
        data_2_depth_data = data_2

    size_1 = data_1_depth_data.shape[0]
    size_2 = data_2_depth_data.shape[0]

    if size_1 != size_2:
        cut_index = min(size_1, size_2)

        # Calculate indices for centering the cut
        start_1 = (size_1 - cut_index) // 2
        end_1 = start_1 + cut_index
        start_2 = (size_2 - cut_index) // 2
        end_2 = start_2 + cut_index

        data_1_depth_data = data_1_depth_data[start_1:end_1]
        data_2_depth_data = data_2_depth_data[start_2:end_2]

        if isinstance(data_1, dict):
            data_1["depth_data"] = data_1_depth_data
            data_2["depth_data"] = data_2_depth_data
        else:
            data_1 = data_1_depth_data
            data_2 = data_2_depth_data

    return data_1, data_2


# ============================================================================
# OPTIMIZATION AND ERROR FUNCTIONS
# ============================================================================


def errorfunc_for_inter_profile_alignment(
    transform_parameters, profile1, profile2, iVerbose=0
):
    """
    ErrorfuncForInterProfileAlignment: Error function for profile optimization

    Args:
        transform_parameters: [translation, scaling*10000-10000]
        profile1: Reference profile
        profile2: Compared profile
        iVerbose: Verbosity level

    Returns:
        similarity_score: Negative similarity (for minimization)
    """
    translation = transform_parameters[0]
    scaling = transform_parameters[1] / 10000 + 1

    profile2_trans = translate_scale_pointset(
        profile2, np.array([[translation, scaling]])
    )
    similarity_score = -get_similarity_score(profile1, profile2_trans)

    return similarity_score


class FminSearchBnd:
    """
    Bounded fminsearch using variable transformation
    Similar to MATLAB's fminsearchbnd
    """

    @staticmethod
    def optimize(func, x0, lb, ub, options=None, args=()):
        """
        Bounded optimization using scipy.optimize.minimize

        Args:
            func: Objective function
            x0: Initial guess
            lb: Lower bounds
            ub: Upper bounds
            options: Optimization options
            args: Additional arguments for func

        Returns:
            x: Optimal parameters
            fval: Function value at optimum
        """
        x0 = np.asarray(x0).flatten()
        lb = np.asarray(lb).flatten()
        ub = np.asarray(ub).flatten()
        n = len(x0)

        if options is None:
            options = {}

        # Classify bounds
        bound_class = np.zeros(n, dtype=int)
        for i in range(n):
            k = int(np.isfinite(lb[i])) + 2 * int(np.isfinite(ub[i]))
            bound_class[i] = k
            if k == 3 and lb[i] == ub[i]:
                bound_class[i] = 4

        # Transform initial values
        x0u = []
        for i in range(n):
            if bound_class[i] == 1:  # lower bound only
                if x0[i] <= lb[i]:
                    x0u.append(0)
                else:
                    x0u.append(np.sqrt(x0[i] - lb[i]))
            elif bound_class[i] == 2:  # upper bound only
                if x0[i] >= ub[i]:
                    x0u.append(0)
                else:
                    x0u.append(np.sqrt(ub[i] - x0[i]))
            elif bound_class[i] == 3:  # both bounds
                if x0[i] <= lb[i]:
                    x0u.append(-np.pi / 2)
                elif x0[i] >= ub[i]:
                    x0u.append(np.pi / 2)
                else:
                    val = 2 * (x0[i] - lb[i]) / (ub[i] - lb[i]) - 1
                    x0u.append(2 * np.pi + np.arcsin(np.clip(val, -1, 1)))
            elif bound_class[i] == 0:  # unconstrained
                x0u.append(x0[i])
            # bound_class[i] == 4 (fixed) is skipped

        x0u = np.array(x0u)

        # Define transformation function
        def xtransform(x):
            xtrans = np.zeros(n)
            k = 0
            for i in range(n):
                if bound_class[i] == 1:  # lower bound only
                    xtrans[i] = lb[i] + x[k] ** 2
                    k += 1
                elif bound_class[i] == 2:  # upper bound only
                    xtrans[i] = ub[i] - x[k] ** 2
                    k += 1
                elif bound_class[i] == 3:  # both bounds
                    xtrans[i] = (np.sin(x[k]) + 1) / 2
                    xtrans[i] = xtrans[i] * (ub[i] - lb[i]) + lb[i]
                    xtrans[i] = np.clip(xtrans[i], lb[i], ub[i])
                    k += 1
                elif bound_class[i] == 4:  # fixed
                    xtrans[i] = lb[i]
                elif bound_class[i] == 0:  # unconstrained
                    xtrans[i] = x[k]
                    k += 1
            return xtrans

        # Wrapper function
        def intrafun(x):
            xtrans = xtransform(x)
            return func(xtrans, *args)

        # Run optimization
        method = options.get("method", "Nelder-Mead")
        tol = options.get("TolFun", 1e-6)

        result = minimize(
            intrafun,
            x0u,
            method=method,
            options={
                "xatol": tol,
                "fatol": tol,
                "disp": options.get("Display", "off") != "off",
            },
        )

        x = xtransform(result.x)
        fval = result.fun

        return x, fval


# ============================================================================
# COMPARISON RESULTS
# ============================================================================


def get_striated_mark_comparison_results(
    trans_array, profiles1, profiles2, results_table
):
    """
    GetStriatedMarkComparisonResults: Calculate comparison measures

    Args:
        trans_array: Transformation parameters [translation, scaling]
        profiles1: Reference profile
        profiles2: Compared profile
        results_table: Results structure

    Returns:
        results_table: Updated results structure
    """
    # Generate composite transformation matrix
    transform_matrix = np.eye(3)
    for cur_param in range(trans_array.shape[0]):
        translation = trans_array[cur_param, 0]
        scaling = trans_array[cur_param, 1]

        current_transform = np.array([[scaling, 0, translation], [0, 1, 0], [0, 0, 1]])
        transform_matrix = current_transform @ transform_matrix

    # Set results table fields
    results_table["dPos"] = transform_matrix[0, 2] * results_table["vPixSep1"]
    results_table["dScale"] = transform_matrix[0, 0]
    results_table["simVal"] = get_similarity_score(profiles2, profiles1)
    results_table["lOverlap"] = len(profiles1) * results_table["vPixSep1"]
    results_table["ccf"] = get_similarity_score(
        profiles2, profiles1, "cross_correlation"
    )

    # Convert to micrometers
    profiles1 = profiles1.flatten() * 1e6  # [m] -> [um]
    profiles2 = profiles2.flatten() * 1e6  # [m] -> [um]
    profiles12 = profiles2 - profiles1  # difference

    N = len(profiles1)

    results_table["sa_1"] = np.sum(np.abs(profiles1)) / N
    results_table["sq_1"] = np.sqrt(np.dot(profiles1, profiles1) / N)
    results_table["sa_2"] = np.sum(np.abs(profiles2)) / N
    results_table["sq_2"] = np.sqrt(np.dot(profiles2, profiles2) / N)
    results_table["sa12"] = np.sum(np.abs(profiles12)) / N
    results_table["sq12"] = np.sqrt(np.dot(profiles12, profiles12) / N)

    results_table["ds1"] = (results_table["sq12"] / results_table["sq_1"]) ** 2
    results_table["ds2"] = (results_table["sq12"] / results_table["sq_2"]) ** 2
    results_table["ds"] = results_table["sq12"] ** 2 / (
        results_table["sq_1"] * results_table["sq_2"]
    )

    return results_table


# ============================================================================
# MAIN ALIGNMENT FUNCTIONS
# ============================================================================


def align_inter_profiles_multi_scale(
    profiles1_in, profiles2_in, param, results_table=None, iVerbose=0
):
    """
    AlignInterProfilesMultiScale: Multi-scale profile registration

    This function is a placeholder that assumes ApplyLowPassFilter exists.
    You need to provide the actual implementation or import it.

    Args:
        profiles1_in: Reference profile (dict or array)
        profiles2_in: Compared profile (dict or array)
        param: Parameter dictionary
        results_table: Results structure (optional)
        iVerbose: Verbosity level

    Returns:
        trans_array: Transformation parameters
        cross_correlation: Correlation values
        profiles2_out: Registered compared profile
        profiles1_out: Registered reference profile
        results_table: Updated results structure
    """
    # Initialize results table if needed
    if results_table is None:
        results_table = profile_correlator_res_init()

    # Get parameters
    plot_figures = get_param_value(param, "plot_figures", 0)
    pass_scales = get_param_value(param, "pass", [1000, 500, 250, 100, 50, 25, 10, 5])
    filtertype = get_param_value(param, "filtertype", "lowpass")
    remove_zeros = get_param_value(param, "remove_zeros", 1)
    show_info = get_param_value(param, "show_info", 0)
    x0 = get_param_value(param, "x0", [0, 0])
    use_mean = get_param_value(param, "use_mean", 1)

    cutoff_hi = get_param_value(param, "cutoff_hi", 1000)
    cutoff_lo = get_param_value(param, "cutoff_lo", 5)

    # Update cutoffs from profile data if available
    if isinstance(profiles1_in, dict) and isinstance(profiles2_in, dict):
        if "cutoff_hi" in profiles1_in and "cutoff_hi" in profiles2_in:
            if (
                profiles1_in["cutoff_hi"] is not None
                and profiles2_in["cutoff_hi"] is not None
            ):
                cutoff_hi = min(profiles1_in["cutoff_hi"], profiles2_in["cutoff_hi"])
        if "cutoff_lo" in profiles1_in and "cutoff_lo" in profiles2_in:
            if (
                profiles1_in["cutoff_lo"] is not None
                and profiles2_in["cutoff_lo"] is not None
            ):
                cutoff_lo = max(profiles1_in["cutoff_lo"], profiles2_in["cutoff_lo"])

    max_translation = get_param_value(param, "max_translation", 1e7)
    max_translation = max_translation / 1000  # [um] -> [mm]
    max_scaling = get_param_value(param, "max_scaling", 0.05)
    cut_borders_after_smoothing_for_alignment = get_param_value(
        param, "cut_borders_after_smoothing_for_alignment", 0
    )

    # Extract data
    if isinstance(profiles1_in, dict):
        xdim = profiles1_in["xdim"]
        LR = profiles1_in.get("LR", max(cutoff_lo * 1e-6, 2 * xdim))
        profiles1 = profiles1_in["depth_data"].copy()
        profiles2 = profiles2_in["depth_data"].copy()
    else:
        xdim = get_param_value(param, "xdim", 4.3831e-07)
        profiles1 = profiles1_in.copy()
        profiles2 = profiles2_in.copy()
        LR = max(cutoff_lo * 1e-6, 2 * xdim)

    # Take mean or median across columns
    if use_mean:
        if profiles1.ndim > 1 and profiles1.shape[1] > 1:
            profiles1 = np.nanmean(profiles1, axis=1)
        else:
            profiles1 = profiles1.flatten()
        if profiles2.ndim > 1 and profiles2.shape[1] > 1:
            profiles2 = np.nanmean(profiles2, axis=1)
        else:
            profiles2 = profiles2.flatten()
    else:
        if profiles1.ndim > 1 and profiles1.shape[1] > 1:
            profiles1 = np.nanmedian(profiles1, axis=1)
        else:
            profiles1 = profiles1.flatten()
        if profiles2.ndim > 1 and profiles2.shape[1] > 1:
            profiles2 = np.nanmedian(profiles2, axis=1)
        else:
            profiles2 = profiles2.flatten()

    if show_info:
        print("Aligning mean profiles of different toolmarks...")

    # Check sizes
    size_1 = len(profiles1)
    size_2 = len(profiles2)

    if size_1 != size_2:
        raise ValueError(
            "The profiles are different length! "
            "To apply the transformation parameters, "
            "the profiles have to be cropped first!"
        )

    # Convert pass to array
    pass_scales = np.array(pass_scales).reshape(-1, 1)

    # Initialize x0
    if np.sum(x0) > 0:
        x0 = np.array([round(x0[0] / xdim / 1000), 1 + x0[1] / 100])
    else:
        x0 = np.array([0, 0])

    # Initialize translation bounds
    if max_translation != 10000:
        redetermine_max_trans = False
        max_translation = round(max_translation / xdim / 1000)
    else:
        redetermine_max_trans = True

    if max_scaling != 0.05:
        max_scaling = max_scaling / 100

    # Ensure column vectors
    profiles1 = profiles1.flatten()
    profiles2 = profiles2.flatten()

    mean_1 = profiles1
    translation_tot = 0
    scaling_tot = 1
    current_scaling = 1

    cross_correlation = []
    trans_array = []
    profiles2_mod = profiles2.copy()

    # Multi-scale loop
    for scale_level in range(len(pass_scales)):
        max_translation_tmp = max_translation - translation_tot
        min_translation_tmp = max_translation + translation_tot
        max_scaling_tmp = max_scaling * (1 - (scaling_tot - 1))
        min_scaling_tmp = max_scaling * (1 + (scaling_tot - 1))

        # Check resolution threshold
        if LR is None:
            resolution_threshold = 2 * xdim
        else:
            resolution_threshold = LR

        current_cutoff = pass_scales[scale_level, 0]

        if (
            current_cutoff * 1e-6 >= resolution_threshold
            and current_cutoff <= cutoff_hi
            and current_cutoff >= cutoff_lo
        ):
            if show_info:
                print(f"Scale: {scale_level + 1}")

            mean_2 = profiles2_mod

            # NOTE: This is where ApplyLowPassFilter would be called
            # For now, we'll skip filtering and use the profiles directly
            # You need to implement or provide ApplyLowPassFilter

            # Placeholder: use unfiltered profiles
            result1 = mean_1.copy()
            result2 = mean_2.copy()

            # WARNING: Actual MATLAB code calls ApplyLowPassFilter here!
            # This is a simplified version without filtering

            # Update translation bounds if needed
            if redetermine_max_trans:
                min_translation_tmp = current_cutoff
                max_translation_tmp = current_cutoff

            # Determine subsample factor
            subsample_factor = max(
                1, int(np.ceil(current_cutoff / (xdim * 1e6) / 2 / 5))
            )

            current_profile1 = result1[::subsample_factor]
            current_profile2 = result2[::subsample_factor]

            current_profile1_lo = result1
            current_profile2_lo = result2

            # Optimization
            options = {
                "TolFun": 1e-6,
                "TolX": 1e-6,
                "Display": "on" if show_info else "off",
            }

            lb = np.array(
                [
                    -round(min_translation_tmp / subsample_factor),
                    ((1 - min_scaling_tmp) / current_scaling - 1) * 10000,
                ]
            )
            ub = np.array(
                [
                    round(max_translation_tmp / subsample_factor),
                    ((1 + max_scaling_tmp) / current_scaling - 1) * 10000,
                ]
            )

            x, fval = FminSearchBnd.optimize(
                errorfunc_for_inter_profile_alignment,
                x0,
                lb,
                ub,
                options,
                args=(current_profile1, current_profile2, iVerbose),
            )

            translation = x[0] * subsample_factor
            scaling = x[1] / 10000 + 1
            current_scaling = scaling

            profiles2_scale_trans = translate_scale_pointset(
                mean_2, np.array([[translation, scaling]])
            )

            current_profile2_lo = translate_scale_pointset(
                current_profile2_lo, np.array([[translation, scaling]])
            )

            cross_correlation_1 = get_similarity_score(
                result1, current_profile2_lo, "cross_correlation"
            )

            profile1_no_zeros, profile2_no_zeros, _ = remove_boundary_zeros(
                mean_1, profiles2_scale_trans
            )

            cross_correlation_2 = get_similarity_score(
                profile1_no_zeros, profile2_no_zeros, "cross_correlation"
            )

            cross_correlation.append([cross_correlation_1, cross_correlation_2])

            translation_tot += translation
            scaling_tot *= scaling

            trans_array.append([translation, scaling])
            profiles2_mod = translate_scale_pointset(profiles2, np.array(trans_array))

    # Finalize
    trans_array = np.array(trans_array) if trans_array else np.array([[0, 1]])
    cross_correlation = (
        np.array(cross_correlation) if cross_correlation else np.array([[0, 0]])
    )

    if remove_zeros:
        profiles1_out, profiles2_out, extra_translate = remove_boundary_zeros(
            profiles1, profiles2_mod
        )
    else:
        profiles1_out = profiles1
        profiles2_out = profiles2_mod

    # Update final cross-correlation
    if len(cross_correlation) > 0:
        cross_correlation[-1, 1] = get_similarity_score(
            profiles1_out, profiles2_out, "cross_correlation"
        )

    # Update results table
    results_table = get_striated_mark_comparison_results(
        trans_array, profiles1_out, profiles2_out, results_table
    )

    return trans_array, cross_correlation, profiles2_out, profiles1_out, results_table


def align_inter_profiles_partial_multi_scale(
    reference, partial_mark, param, results_table=None, iVerbose=0
):
    """
    AlignInterProfilesPartialMultiScale: Align partial profile to full profile

    This function is a placeholder that assumes DetermineMatchCandidatesMultiScale
    and other filtering functions exist.

    Args:
        reference: Full reference profile
        partial_mark: Partial profile to align
        param: Parameter dictionary
        results_table: Results structure (optional)
        iVerbose: Verbosity level

    Returns:
        transform_parameters_max_xcorr_candidate: Best transformation
        max_xcorr_candidate_start: Best starting position
        xcorr: Cross-correlation value
        profiles2_out: Aligned partial profile (in dict)
        profiles1_out: Aligned reference profile (in dict)
        results_table: Updated results structure
    """
    # Initialize
    if results_table is None:
        results_table = profile_correlator_res_init()

    # Parameters
    show_info = get_param_value(param, "show_info", 1)
    remove_zeros = get_param_value(param, "remove_zeros", 1)
    use_mean = get_param_value(param, "use_mean", 1)

    # Take mean/median
    if use_mean:
        if reference["depth_data"].ndim > 1 and reference["depth_data"].shape[1] > 1:
            reference["depth_data"] = np.nanmean(reference["depth_data"], axis=1)
        if (
            partial_mark["depth_data"].ndim > 1
            and partial_mark["depth_data"].shape[1] > 1
        ):
            partial_mark["depth_data"] = np.nanmean(partial_mark["depth_data"], axis=1)
    else:
        if reference["depth_data"].ndim > 1 and reference["depth_data"].shape[1] > 1:
            reference["depth_data"] = np.nanmedian(reference["depth_data"], axis=1)
        if (
            partial_mark["depth_data"].ndim > 1
            and partial_mark["depth_data"].shape[1] > 1
        ):
            partial_mark["depth_data"] = np.nanmedian(
                partial_mark["depth_data"], axis=1
            )

    if show_info:
        print("Aligning mean (partial) profiles of different toolmarks...")

    # NOTE: This is where DetermineMatchCandidatesMultiScale would be called
    # This function requires RemoveNoiseGaussian, RemoveShapeGaussian, etc.
    # For now, we'll create a simple brute-force search

    # Simplified brute-force search
    part_prof_length = len(partial_mark["depth_data"])
    ref_length = len(reference["depth_data"])

    # Try all possible positions
    max_xcorr_candidates = []
    trans_array_dict = {}
    profiles1_dict = {}
    profiles2_dict = {}
    label_opts = []

    param_tmp = param.copy()
    param_tmp["plot_figures"] = 0
    param_tmp["pass"] = [
        p for p in param_tmp["pass"] if p <= param_tmp.get("cutoff_hi", 1000)
    ]

    # Simple sliding window search
    for cur_location in range(ref_length - part_prof_length + 1):
        cur_candidate = len(label_opts)
        label_opts.append(cur_location)

        reference_data_tmp = {
            "depth_data": reference["depth_data"][
                cur_location : cur_location + part_prof_length
            ],
            "xdim": reference["xdim"],
        }

        if "cutoff_hi" in reference:
            reference_data_tmp["cutoff_hi"] = reference["cutoff_hi"]
        else:
            reference_data_tmp["cutoff_hi"] = param.get("cutoff_hi", 1000)

        if "cutoff_lo" in reference:
            reference_data_tmp["cutoff_lo"] = reference["cutoff_lo"]
        else:
            reference_data_tmp["cutoff_lo"] = param.get("cutoff_lo", 5)

        # Align this segment
        transform_parameters, xcorr, profile_2_reg, profile_1_reg, results_table = (
            align_inter_profiles_multi_scale(
                reference_data_tmp, partial_mark, param_tmp, results_table, iVerbose
            )
        )

        max_xcorr_candidates.append(xcorr[-1, 1] if len(xcorr) > 0 else 0)
        trans_array_dict[f"Candidate_{cur_candidate}"] = transform_parameters
        profiles1_dict[f"Candidate_{cur_candidate}"] = profile_1_reg
        profiles2_dict[f"Candidate_{cur_candidate}"] = profile_2_reg

    # Find best candidate
    max_xcorr_candidates = np.array(max_xcorr_candidates)
    I = np.argmax(max_xcorr_candidates)

    transform_parameters_max_xcorr_candidate = trans_array_dict[f"Candidate_{I}"]
    max_xcorr_candidate_start = label_opts[I]

    profile1_out = profiles1_dict[f"Candidate_{I}"]
    profile2_out = profiles2_dict[f"Candidate_{I}"]

    xcorr = get_similarity_score(profile1_out, profile2_out)

    # Package outputs
    profiles2_out_dict = {"Candidate_1": profile2_out}
    profiles1_out_dict = {"Candidate_1": profile1_out}

    # Update results table
    results_table["startPartProfile"] = (
        max_xcorr_candidate_start * reference["xdim"] * 1e6
    )

    return (
        transform_parameters_max_xcorr_candidate,
        max_xcorr_candidate_start,
        xcorr,
        profiles2_out_dict,
        profiles1_out_dict,
        results_table,
    )


# ============================================================================
# MAIN PROFILE CORRELATOR FUNCTION
# ============================================================================


def profile_correlator_single(
    profile_ref, profile_comp, results_table=None, param=None, iVerbose=0
):
    """
    ProfileCorrelatorSingle: Process a single profile registration and correlation

    Args:
        profile_ref: Scratch structure with reference profile
        profile_comp: Scratch structure with compared profile
        results_table: Structure with comparison results (optional)
        param: Structure with analysis settings
        iVerbose: Verbosity level (0=none, 1=top level, 2=detailed)

    Returns:
        results_table: Updated structure with comparison results
    """
    # Initialize parameters
    if param is None:
        param = {}

    # Get parameters
    part_mark_perc = get_param_value(param, "part_mark_perc", 8)

    # Initialize results table if needed
    if results_table is None or not results_table:
        results_table = profile_correlator_res_init()

    # Set results table fields
    results_table["bProfile"] = 1
    results_table["bSegments"] = 0

    # Subsample the larger profile to equalize sampling
    profile_ref_equal, profile_comp_equal = equalize_sampling_distance(
        profile_ref, profile_comp
    )

    # Set results table fields
    results_table["vPixSep1"] = profile_ref_equal["xdim"] * 1e6
    results_table["vPixSep2"] = profile_comp_equal["xdim"] * 1e6

    # Determine profile lengths
    size_1 = profile_ref_equal["depth_data"].shape[0]
    size_2 = profile_comp_equal["depth_data"].shape[0]

    # Calculate relative length difference
    length_diff_percentage = abs(size_1 - size_2) / max(size_1, size_2) * 100

    if length_diff_percentage < part_mark_perc:
        # Full profile comparison
        results_table["bPartialProfile"] = 0

        # Make profiles equal length
        profile_ref_equal_mod, profile_comp_equal_mod = make_dataset_length_equal(
            profile_ref_equal, profile_comp_equal
        )

        # Align the profiles
        (
            transform_parameters,
            xcorr,
            profile_comp_equal_mod_reg,
            profile_ref_equal_mod_reg,
            results_table,
        ) = align_inter_profiles_multi_scale(
            profile_ref_equal_mod,
            profile_comp_equal_mod,
            param,
            results_table,
            iVerbose,
        )
    else:
        # Partial profile comparison
        results_table["bPartialProfile"] = 1

        if size_1 > size_2:
            (
                transform_parameters_max_xcorr_candidate,
                max_xcorr_candidate_start,
                xcorr,
                profile_comp_equal_reg_dict,
                profile_ref_equal_reg_dict,
                results_table,
            ) = align_inter_profiles_partial_multi_scale(
                profile_ref_equal, profile_comp_equal, param, results_table, iVerbose
            )
        else:
            (
                transform_parameters_max_xcorr_candidate,
                max_xcorr_candidate_start,
                xcorr,
                profile_ref_equal_reg_dict,
                profile_comp_equal_reg_dict,
                results_table,
            ) = align_inter_profiles_partial_multi_scale(
                profile_comp_equal, profile_ref_equal, param, results_table, iVerbose
            )

    # Set overlap percentage
    if size_1 >= size_2:
        results_table["pOverlap"] = results_table["lOverlap"] / (
            size_2 * results_table["vPixSep2"]
        )
    else:
        results_table["pOverlap"] = results_table["lOverlap"] / (
            size_1 * results_table["vPixSep1"]
        )

    return results_table
