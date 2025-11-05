from fastapi import APIRouter, FastAPI
from uvicorn import run

from comparators.router import comparison_router
from pre_processors.router import pre_processors
from processors.router import processors

app = FastAPI()
prefix_router = APIRouter()

prefix_router.include_router(pre_processors)
prefix_router.include_router(processors)
prefix_router.include_router(comparison_router)
app.include_router(prefix_router)


if __name__ == "__main__":
    run(app, host="127.0.0.1", port=8000, reload=False, workers=1)
