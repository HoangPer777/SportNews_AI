"""FastAPI application for the Sports Weekly Intelligence Agent."""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import logging
import os

logging.basicConfig(level=logging.INFO)

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from graph import PipelineStageError, run_pipeline
from models.schemas import GenerateReportRequest, ReportMetadata, ReportResponse
from tools.email_sender import send_report_email

logger = logging.getLogger(__name__)

app = FastAPI(title="Sports Weekly Intelligence Agent")


def build_error_metadata(period_type: str, lookback_days: int, stage: str) -> ReportMetadata:
    from datetime import datetime, timedelta, timezone

    today = datetime.now(tz=timezone(timedelta(hours=7))).date()
    if period_type == "weekly":
        period_start = today - timedelta(days=today.weekday())
        period_end = period_start + timedelta(days=6)
    else:
        period_end = today
        period_start = today - timedelta(days=lookback_days - 1)
    return ReportMetadata(
        period_type=period_type,
        period_start=period_start,
        period_end=period_end,
        lookback_days=lookback_days,
        stage=stage,
    )

# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

scheduler = BackgroundScheduler(timezone="Asia/Ho_Chi_Minh")


def scheduled_report_job() -> None:
    """Run the pipeline and email the report. Called by the scheduler."""
    logger.info("Scheduled job: starting weekly report generation...")
    try:
        period_type = os.getenv("REPORT_PERIOD", "weekly")
        lookback_days = int(os.getenv("REPORT_LOOKBACK_DAYS", "1" if period_type == "daily" else "7"))
        final_state = run_pipeline(period_type=period_type, lookback_days=lookback_days)
        report = final_state.get("report")
        if report is None:
            error = final_state.get("error", "Unknown error")
            logger.error("Scheduled job: pipeline produced no report. Error: %s", error)
            return

        # Read the saved markdown file and send it
        output_path = os.getenv("REPORT_OUTPUT_PATH", "outputs/weekly_report.md")
        if os.path.exists(output_path):
            with open(output_path, encoding="utf-8") as f:
                markdown = f.read()
            send_report_email(markdown)
        else:
            logger.warning("Scheduled job: report file not found at %s", output_path)

        logger.info("Scheduled job: weekly report sent successfully.")
    except Exception as exc:
        logger.error("Scheduled job failed: %s", exc, exc_info=True)


@app.on_event("startup")
def start_scheduler() -> None:
    """Start the weekly scheduler on app startup."""
    # Default: every Monday at 08:00 Vietnam time
    # Override via env: SCHEDULE_DAY_OF_WEEK (0=Mon..6=Sun), SCHEDULE_HOUR, SCHEDULE_MINUTE
    day_of_week = os.getenv("SCHEDULE_DAY_OF_WEEK", "mon")
    hour = int(os.getenv("SCHEDULE_HOUR", "8"))
    minute = int(os.getenv("SCHEDULE_MINUTE", "0"))

    scheduler.add_job(
        scheduled_report_job,
        trigger=CronTrigger(day_of_week=day_of_week, hour=hour, minute=minute),
        id="weekly_report",
        replace_existing=True,
    )
    scheduler.start()
    logger.info(
        "Scheduler started: weekly report every %s at %02d:%02d (Asia/Ho_Chi_Minh)",
        day_of_week, hour, minute,
    )


@app.on_event("shutdown")
def stop_scheduler() -> None:
    scheduler.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ReportResponse(status="error", error=str(exc)).model_dump(mode="json"),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def verify_internal_api_key(x_internal_api_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("INTERNAL_API_KEY", "").strip()
    if not expected:
        return
    if x_internal_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid internal API key")


@app.post("/generate-report", status_code=202, response_model=ReportResponse)
async def generate_report(
    payload: GenerateReportRequest | None = None,
    _auth: None = Depends(verify_internal_api_key),
) -> ReportResponse:
    """Trigger the full pipeline manually and return the completed report."""
    payload = payload or GenerateReportRequest()
    lookback_days = payload.lookback_days
    if lookback_days is None:
        lookback_days = 1 if payload.period_type == "daily" else 7
    try:
        final_state = run_pipeline(period_type=payload.period_type, lookback_days=lookback_days)
    except PipelineStageError as exc:
        error_msg = str(exc)
        logger.error("Pipeline failed at stage %s: %s", exc.stage, exc.cause, exc_info=True)
        return ReportResponse(
            status="error",
            report=None,
            metadata=build_error_metadata(payload.period_type, lookback_days, exc.stage),
            error=error_msg,
        )
    report = final_state.get("report")
    metadata = final_state.get("metadata")
    pipeline_error = final_state.get("error")

    if report is None:
        error_msg = pipeline_error or "Pipeline completed but no report was generated."
        logger.error("Pipeline produced no report. Error: %s", error_msg)
        return ReportResponse(status="error", report=None, metadata=metadata, error=error_msg)

    # Optionally send email on manual trigger too
    if os.getenv("EMAIL_ON_MANUAL_TRIGGER", "false").lower() == "true":
        output_path = os.getenv("REPORT_OUTPUT_PATH", "outputs/weekly_report.md")
        if os.path.exists(output_path):
            with open(output_path, encoding="utf-8") as f:
                markdown = f.read()
            try:
                send_report_email(markdown)
            except Exception as exc:
                logger.warning("Email send failed (non-fatal): %s", exc)

    return ReportResponse(status="success", report=report, metadata=metadata)
