from fastapi import APIRouter

from .schema import ProcessResultOneToMany, ProcessResultOneToOne

comparison_router = APIRouter(
    prefix="/comparison",
    tags=["comparison"],
)


@comparison_router.get(
    path="/proces_image_{id_key}",
    summary="check status of comparison proces",
    description="""check status of proces of a comparison, of the on to one comparison or the one to many,
    **maybe in 2 seperate urls? dunno yet**""",
)
async def proces_image(id_key: str) -> ProcessResultOneToOne | ProcessResultOneToMany:
    """Docstring"""
    ...


@comparison_router.get(
    path="/plot_results_{id_key}",
    summary="return the plot of a comparison",
    description="""return a plot of the selected processed scan\n
## MatLab functions:
- PlotResultsNFIProfile (caching or push to db when doing `proces_image`)
""",
)
async def plot_results(id_key: str) -> str:  # TODO: what is send here?
    """Docstring"""
    ...


@comparison_router.get(
    path="/scores_{id_key}",
    summary="return the scores of a comparison",
    description="""return the scores of the selected processed scan\n
## MatLab functions:
- ScoreCalculationScadular  (caching or push to db when doing `proces_image`)
""",
)
async def scores(id_key: str) -> int:
    """Docstring"""
    return 80


@comparison_router.get(
    path="/distribution_plot_{id_key}",
    summary="return the distribution_plot of a comparison",
    description="""return the distribution_plot of the selected processed scan\n
## MatLab functions:
- GenerateScoreDistributionPlots  (caching or push to db when doing `proces_image`)
""",
)
async def distribution_plot(id_key: str) -> str:  # TODO: what is send back?
    """Docstring"""
    ...
