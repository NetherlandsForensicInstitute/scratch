"""Type definitions for the mutator registry system."""

from __future__ import annotations

from dataclasses import astuple, dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable


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

    def __call__(self, scan_image: ScanImage, **kwargs: Any) -> ScanImage:
        """Execute the mutator function."""
        return self.func(scan_image, **kwargs)

    @property
    def name(self) -> str:
        return self.metadata.name


class MutatorAlreadyRegisteredError(Exception):
    """Raised when attempting to register a mutator with a name that already exists."""

    def __init__(self, mutator_name: str) -> None:
        self.mutator_name = mutator_name
        super().__init__(
            f"Mutator '{mutator_name}' is already registered. "
            f"Use a different name or unregister the existing one first."
        )
