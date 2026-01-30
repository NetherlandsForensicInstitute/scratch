"""Type definitions for the mutator registry system."""

from __future__ import annotations

from dataclasses import astuple, dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, Self, runtime_checkable

from loguru import logger


if TYPE_CHECKING:
    from container_models.scan_image import ScanImage


@runtime_checkable
class MutatorProtocol(Protocol):
    """Protocol defining the expected signature for image mutators.

    All mutators must accept a ScanImage as first argument and return a ScanImage.
    Additional keyword arguments are allowed for configuration.
    """

    def __call__(self, scan_image: ScanImage, **kwargs: Any) -> ScanImage: ...


@dataclass(frozen=True)
class MutatorMetadata:
    """Immutable metadata container for registered mutators.

    :param name: Unique identifier for the mutator (typically function name)
    :param description: Human-readable description (from docstring or explicit)
    """

    name: str
    description: str

    def __post_init__(self) -> None:
        if fields := ", ".join(field for field, value in astuple(self) if not value):
            raise ValueError(f"Mutator {fields} cannot be empty")


@dataclass
class RegisteredMutator:
    """A mutator function bundled with its registration metadata.

    :param func: The underlying mutator function
    :param metadata: Registration metadata
    """

    func: MutatorProtocol
    metadata: MutatorMetadata
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __call__(self, scan_image: ScanImage) -> ScanImage:
        """Execute the mutator function."""
        return self.func(scan_image, **self.kwargs)

    @property
    def name(self) -> str:
        return self.metadata.name

    def bind(self, **kwargs: Any) -> Self:
        """Create a bound mutator with pre-filled arguments.

        Example:
            blurred = gaussian_blur.bind(sigma=2.0)
            result = blurred(scan_image)
        """
        if not kwargs:
            logger.warning(f"bind called on {self.metadata.name} with no keyword value")
        self.kwargs |= kwargs
        return self


class MutatorAlreadyRegisteredError(Exception):
    """Raised when attempting to register a mutator with a name that already exists."""

    def __init__(self, mutator_name: str) -> None:
        self.mutator_name = mutator_name
        super().__init__(
            f"Mutator '{mutator_name}' is already registered. "
            f"Use a different name or unregister the existing one first."
        )
