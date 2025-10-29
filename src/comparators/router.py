from fastapi import APIRouter

comparison_router = APIRouter(
    prefix="/comparator",
    tags=["comparator"],
)


@comparison_router.get(
    path="/",
    summary="check status of comparison proces",
    description="""Some description of comparison endpoint, you can use basic **markup**""",
)
async def comparison_root():
    return {"message": "Hello from the comparator"}
