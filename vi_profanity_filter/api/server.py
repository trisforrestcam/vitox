"""FastAPI server cho Vietnamese Profanity Filter."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..filter import ViProfanityFilter


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager cho startup/shutdown events."""
    print("Vietnamese Profanity Filter API started")
    yield
    print("Vietnamese Profanity Filter API shutting down")


app = FastAPI(title="Vietnamese Profanity Filter API", lifespan=lifespan)

# Singleton filter instance
_filter = ViProfanityFilter()


class CheckRequest(BaseModel):
    """Request body cho endpoint ``/check``."""

    text: str


class CensorRequest(BaseModel):
    """Request body cho endpoint ``/censor``."""

    text: str


@app.post("/check")
async def check(request: CheckRequest) -> dict:
    """Kiểm tra text có chứa profanity hay không.

    Trả về kết quả phân tích đầy đủ gồm label, confidence score,
    matched words và layer đưa ra quyết định.
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")
    return _filter.check(request.text)


@app.post("/censor")
async def censor(request: CensorRequest) -> dict:
    """Trả về text với các từ profane được thay bằng dấu *."""
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")
    return {"censored_text": _filter.censor(request.text)}


@app.get("/health")
async def health() -> dict:
    """Health-check endpoint."""
    return {"status": "ok"}
