from fastapi import APIRouter

processors = APIRouter(
    prefix="/processor",
    tags=["processor"],
)


@processors.get(
    path="/",
    summary="check status of comparison proces",
    description="""Some description of processors endpoint, you can use basic **markup**""",
)
async def comparison_root():
    return {"message": "Hello from the processors"}
