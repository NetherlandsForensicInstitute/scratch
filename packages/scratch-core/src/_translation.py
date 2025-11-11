from typing import Any, Sequence
from copy import deepcopy

from _data_types import JavaStruct


def get_param_value(input_struct: Any, param_name: str, param_value: Any) -> Any:
    """
    Translation of the MATLAB function 'GetParamValue'

    %GetParamValue   checks if a certain PARAM_NAME is present in INPUT_STRUCT
    %                if yes, outputs INPUT_STRUCT.PARAM_VALUE
    %                otherwise outputs the given PARAM_VALUE
    % Syntax:
    %       param_value = GetParamValue(input_struct, param_name, param_value)
    """
    if hasattr(input_struct, param_name):
        return getattr(input_struct, param_name)
    elif isinstance(input_struct, dict):
        return input_struct.get(param_name, param_value)

    return param_value


def subsample_data(data_in: JavaStruct, step_size: int | Sequence[int]) -> JavaStruct:
    """
    TODO: write docstring
    """
    # Skip the GetParamValue and struct type check in the MATLAB implementation and just use step sizes directly as input arguments
    if isinstance(step_size, int):
        sub_x = sub_y = step_size
    else:
        if not isinstance(step_size, Sequence) or len(step_size) > 2:
            raise ValueError(
                "`step_size` is expected to be a single integer or a pair of two integers."
            )
        # For some reason the subsampling parameters are passed as (dy, dx)
        sub_x, sub_y = step_size[1], step_size[0]

    # Here we assume (y, x) indexing instead of the implementation in MATLAB which assumed (x, y) indexing
    depth_data_sub = data_in.depth_data[::sub_y, ::sub_x]
    texture_data_sub = (
        data_in.texture_data[::sub_y, ::sub_x, :]
        if data_in.texture_data is not None
        else None
    )
    quality_data_sub = (
        data_in.quality_data[::sub_y, ::sub_x]
        if data_in.quality_data is not None
        else None
    )

    # Create the output struct
    data_sub = deepcopy(data_in)
    data_sub.depth_data = depth_data_sub
    data_sub.texture_data = texture_data_sub
    data_sub.quality_data = quality_data_sub
    data_sub.xdim = data_in.xdim * sub_x
    data_sub.ydim = data_in.ydim * sub_y
    data_sub.xdim_orig = data_in.xdim  # In the MATLAB implementation it's data_in.xdim * sub_x here, but that seems like a mistake?
    data_sub.ydim_orig = data_in.ydim
    data_sub.subsampling = [
        sub_y,
        sub_x,
    ]  # For some reason the subsampling parameters are passed as (dy, dx)

    if sub_x != sub_y != 1.0:
        if data_sub.LR <= data_sub.xdim * 2:
            data_sub.LR = None

    return data_sub
