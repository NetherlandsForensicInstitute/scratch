from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    """Write a docstring."""
    return {"message": "Hello NFI Scratch"}
