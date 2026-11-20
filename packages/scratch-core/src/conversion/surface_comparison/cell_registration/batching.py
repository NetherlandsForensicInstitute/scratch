import torch
from loguru import logger
import os
import functools

# Recommended batch sizes. Small batches are preferred to minimize memory usage
# without significantly impacting throughput.
DEFAULT_ANGLE_BATCH_SIZE = {"cuda": 8, "cpu": 2}
DEFAULT_TEMPLATE_BATCH_SIZE = {"cuda": 4, "cpu": 1}
# Refinement jobs (candidate pose x trial angle) scored together in one padded-crop batch.
DEFAULT_FINE_BATCH_SIZE = {"cuda": 256, "cpu": 64}


def resolve_device(device: torch.device | None) -> torch.device:
    """Return *device*, defaulting to CUDA when it is actually usable."""
    if device is not None:
        return device
    return _default_device()


@functools.cache
def _default_device() -> torch.device:
    """Pick the default device once per process; CUDA only if a context can be created."""
    if os.environ.get("SCRATCH_DEVICE", "").lower() == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")  # forces CUDA context creation
            return torch.device("cuda")
        except (RuntimeError, torch.AcceleratorError) as error:
            logger.warning(f"cuda unavailable, falling back to cpu: {error}")
    return torch.device("cpu")


def get_default_batch_size(
    device: torch.device, sizes: dict[str, int], fallback: int = 1
) -> int:
    """Return the default batch size for the given device."""
    return sizes.get(device.type, fallback)
