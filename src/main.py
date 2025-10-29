from fastapi import APIRouter, FastAPI

from src.comparators.router import comparison_router
from src.pre_processors.router import pre_processors
from src.processors.router import processors

app = FastAPI()
prefix_router = APIRouter()

prefix_router.include_router(pre_processors)
prefix_router.include_router(processors)
prefix_router.include_router(comparison_router)
app.include_router(prefix_router)


@app.get("/")
async def root():
    return {"message": "Hello NFI Scratch"}
