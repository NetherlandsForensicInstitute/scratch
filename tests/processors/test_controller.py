from pathlib import Path

import pytest
from computations.likelihood_ratio import get_lr_system
from PIL import Image

from processors.controller import save_lr_overview_plot

RESOURCES = Path(__file__).parent.parent.parent / "packages/scratch-core/tests/resources"


@pytest.fixture
def random_lr_system_path() -> Path:
    """Path to the pre-built random LR system pickle in test resources."""
    return RESOURCES / "random_lr_system.pkl"


class TestSaveLrOverviewPlot:
    """Tests for save_lr_overview_plot."""

    @pytest.mark.integration
    def test_saves_png(self, tmp_path: Path, random_lr_system_path: Path) -> None:
        """Output file is written and is a valid PNG."""
        output = tmp_path / "lr_plot.png"
        system = get_lr_system(random_lr_system_path)
        save_lr_overview_plot(system, score=0.5, lr=1.2, score_max=1.0, output_path=output)

        assert output.exists()
        assert Image.open(output).format == "PNG"
