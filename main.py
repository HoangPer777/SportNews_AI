"""FastAPI application for the Sports Weekly Intelligence Agent."""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import logging

logging.basicConfig(level=logging.INFO)

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from graph import run_pipeline
from models.schemas import ReportResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="Sports Weekly Intelligence Agent")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch any unhandled exception and return HTTP 500."""
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ReportResponse(status="error", error=str(exc)).model_dump(mode="json"),
    )


@app.post("/generate-report", status_code=202, response_model=ReportResponse)
async def generate_report() -> ReportResponse:
    """Trigger the full pipeline and return the completed report."""
    final_state = run_pipeline()
    return ReportResponse(status="success", report=final_state["report"])
