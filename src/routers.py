from fastapi import APIRouter

from comparators.router import comparison_router
from extractors.router import extractor_route
from preprocessors import preprocessor_route
from processors.router import processors

prefix_router = APIRouter()

prefix_router.include_router(preprocessor_route)
prefix_router.include_router(processors)
prefix_router.include_router(comparison_router)
prefix_router.include_router(extractor_route)
