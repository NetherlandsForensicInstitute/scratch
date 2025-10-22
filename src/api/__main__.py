import uvicorn
from fastapi import APIRouter, FastAPI

from .comparison.router import comparison_router
from .processing.router import processing_router
from .scans.router import scan_router

app = FastAPI()
prefix_router = APIRouter()

prefix_router.include_router(scan_router)
prefix_router.include_router(processing_router)
prefix_router.include_router(comparison_router)
app.include_router(prefix_router)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
