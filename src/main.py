from fastapi import APIRouter, FastAPI

from comparators.router import comparison_router
from pre_processors.router import pre_processors
from processors.router import processors

app = FastAPI()
prefix_router = APIRouter()

prefix_router.include_router(pre_processors)
prefix_router.include_router(processors)
prefix_router.include_router(comparison_router)
app.include_router(prefix_router)


@app.get("/")
async def root() -> dict[str, str]:
    """Return a hallo NFI message at the root as placeholder.

    Eventually this will be removed or replaced with another endpoint.

    """
    return {"message": "Hello NFI Scratch"}
