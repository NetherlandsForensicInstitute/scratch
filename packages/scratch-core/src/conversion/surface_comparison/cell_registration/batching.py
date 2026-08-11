import torch

#: Recommended batch sizes, used whenever the caller doesn't pin one explicitly. Small batches win
#: on both devices: a realistic sweep is already past the point where extra batching improves
#: throughput, so larger batches only inflate the working set. Measured on a 1500x1500 canvas,
#: 150px cells, 30 cells, 21 angles, 12-core host with CUDA available:
#:
#:   CUDA (angles=64, templates=32) -> 1.55 s, 16.63 GB   CPU (16, 4) -> 13.63 s
#:   CUDA (angles= 8, templates= 4) -> 1.21 s,  2.65 GB   CPU ( 2, 1) -> 13.75 s
#:   CUDA (angles= 4, templates= 2) -> 1.25 s,  0.80 GB   CPU ( 1, 1) -> 12.47 s
#:
#: The defaults trade ~3% GPU speed for ~3x less memory relative to (8, 4); CPU throughput is flat,
#: so its defaults only keep peak memory low. Re-measure for your own hardware and scan sizes.
DEFAULT_ANGLE_BATCH_SIZE = {"cuda": 8, "cpu": 2}
DEFAULT_TEMPLATE_BATCH_SIZE = {"cuda": 4, "cpu": 1}
#: Refinement jobs (candidate pose x trial angle) scored together in one padded-crop batch.
DEFAULT_FINE_BATCH_SIZE = {"cuda": 256, "cpu": 64}


def resolve_device(device: torch.device | None) -> torch.device:
    """Return *device*, defaulting to CUDA when it is available."""
    return device or torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_default_batch_size(
    device: torch.device, sizes: dict[str, int], fallback: int = 1
) -> int:
    """Look up a recommended batch size for *device*, from one of the ``DEFAULT_*`` tables above."""
    return sizes.get(device.type, fallback)
