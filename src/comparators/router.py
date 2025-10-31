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
async def comparison_root() -> dict[str, str]:
    """Fetch a simple message from the REST API.

    Here is some more information about the function some notes what is expected.
    Special remarks what the function is doing.

    return: dict[str,str] but, use as much as possible Pydantic for return types
    """
    return {"message": "Hello from the comparator"}
