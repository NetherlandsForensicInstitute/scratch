from datetime import date
from pathlib import Path

import pytest
from conversion.data_formats import MarkMetadata
from lir import FeatureData, InstanceData, LLRData
from lir.lrsystems import LRSystem

from tests.helper_function import make_cell

RESOURCES = Path(__file__).parent.parent.parent / "packages/scratch-core/tests/resources"


@pytest.fixture
def random_lr_system_path() -> Path:
    """Path to the pre-built random LR system pickle in test resources."""
    return RESOURCES / "random_lr_system.pkl"


class _IdentityLRSystem(LRSystem):
    """Minimal LRSystem that returns the input score as the LLR."""

    def __init__(self) -> None:
        pass

    def apply(self, instances: InstanceData) -> LLRData:
        assert isinstance(instances, FeatureData)
        return LLRData(features=instances.features[:, 0])


@pytest.fixture
def metadata_reference() -> MarkMetadata:
    return MarkMetadata(case_id="ref-001", firearm_id="fw-1", specimen_id="sp-1", measurement_id="ms-1", mark_id="mk-1")


@pytest.fixture
def metadata_compared() -> MarkMetadata:
    return MarkMetadata(
        case_id="comp-001", firearm_id="fw-2", specimen_id="sp-2", measurement_id="ms-2", mark_id="mk-2"
    )


@pytest.fixture
def base_kwargs(metadata_reference: MarkMetadata, metadata_compared: MarkMetadata) -> dict:
    """Return the common kwargs required by all CalculateLR schemas."""
    return {
        "user_id": "AAAAA",
        "date_report": date(2000, 1, 1),
        "metadata_reference": metadata_reference,
        "metadata_compared": metadata_compared,
    }


@pytest.fixture
def striation_kwargs(
    lr_system_path: Path,
    mark_dirs: tuple[Path, Path],
    base_kwargs: dict,
) -> dict:
    mark_dir_ref, mark_dir_comp = mark_dirs
    return {
        "mark_dir_ref": mark_dir_ref,
        "mark_dir_ref_aligned": mark_dir_ref,
        "mark_dir_comp": mark_dir_comp,
        "mark_dir_comp_aligned": mark_dir_comp,
        "score": 0.5,
        "lr_system_path": lr_system_path,
        **base_kwargs,
    }


@pytest.fixture
def impression_kwargs(
    lr_system_path: Path,
    mark_dirs: tuple[Path, Path],
    base_kwargs: dict,
) -> dict:
    mark_dir_ref, mark_dir_comp = mark_dirs
    return {
        "mark_dir_ref": mark_dir_ref,
        "mark_dir_comp": mark_dir_comp,
        "score": 3,
        "n_cells": 10,
        "lr_system_path": lr_system_path,
        "cells": [
            make_cell(
                center_reference=(i * 1e-3, 0.0),
                best_score=0.3,
                cell_size=(1e-3, 1e-3),
            ).model_dump()
            for i in range(5)
        ],
        **base_kwargs,
    }
