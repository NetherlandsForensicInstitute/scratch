import torch

# Recommended batch sizes. Small batches are preferred to minimize memory usage
# without significantly impacting throughput.
DEFAULT_ANGLE_BATCH_SIZE = {"cuda": 8, "cpu": 2}
DEFAULT_TEMPLATE_BATCH_SIZE = {"cuda": 4, "cpu": 1}
# Refinement jobs (candidate pose x trial angle) scored together in one padded-crop batch.
DEFAULT_FINE_BATCH_SIZE = {"cuda": 256, "cpu": 64}


def resolve_device(device: torch.device | None) -> torch.device:
    """Return *device*, defaulting to CUDA when it is available."""
    return device or torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_default_batch_size(
    device: torch.device, sizes: dict[str, int], fallback: int = 1
) -> int:
    """Return the default batch size for the given device."""
    return sizes.get(device.type, fallback)
