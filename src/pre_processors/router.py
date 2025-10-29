from fastapi import APIRouter

pre_processors = APIRouter(
    prefix="/pre-processor",
    tags=["pre-processor"],
)


@pre_processors.get(
    path="/",
    summary="check status of comparison proces",
    description="""Some description of pre-processors endpoint, you can use basic **markup**""",
)
async def comparison_root():
    return {"message": "Hello from the pre-processors"}
