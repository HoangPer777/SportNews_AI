"""Email sender for the weekly sports report."""

from __future__ import annotations

import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)


def _markdown_to_html(markdown: str) -> str:
    """Convert basic markdown to HTML for email."""
    import re

    lines = markdown.split("\n")
    html_lines = []
    for line in lines:
        # Headers
        if line.startswith("# "):
            line = f"<h1>{line[2:]}</h1>"
        elif line.startswith("## "):
            line = f"<h2>{line[3:]}</h2>"
        elif line.startswith("### "):
            line = f"<h3>{line[4:]}</h3>"
        # Horizontal rule
        elif line.strip() == "---":
            line = "<hr>"
        # List items
        elif line.startswith("- "):
            line = f"<li>{line[2:]}</li>"
        # Bold
        line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
        # Italic
        line = re.sub(r"\*(.+?)\*", r"<em>\1</em>", line)
        # URLs as links
        line = re.sub(r"(https?://\S+)", r'<a href="\1">\1</a>', line)
        # Empty line → paragraph break
        if line.strip() == "":
            line = "<br>"
        html_lines.append(line)

    body = "\n".join(html_lines)
    return f"""
    <html><body style="font-family: Arial, sans-serif; max-width: 800px; margin: auto; padding: 20px;">
    {body}
    </body></html>
    """


def send_report_email(report_markdown: str, subject: str | None = None) -> None:
    """Send the weekly report as an HTML email.

    Required env vars:
        SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD,
        EMAIL_SENDER, EMAIL_RECIPIENTS (comma-separated)
    """
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_password = os.getenv("SMTP_PASSWORD", "")
    sender = os.getenv("EMAIL_SENDER", smtp_user)
    recipients_raw = os.getenv("EMAIL_RECIPIENTS", "")

    if not recipients_raw:
        logger.warning("EMAIL_RECIPIENTS not set — skipping email send.")
        return

    recipients = [r.strip() for r in recipients_raw.split(",") if r.strip()]
    subject = subject or "Báo Cáo Thể Thao Tuần"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)

    # Plain text fallback
    msg.attach(MIMEText(report_markdown, "plain", "utf-8"))
    # HTML version
    msg.attach(MIMEText(_markdown_to_html(report_markdown), "html", "utf-8"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(sender, recipients, msg.as_string())
        logger.info("Report email sent to: %s", recipients)
    except Exception as exc:
        logger.error("Failed to send report email: %s", exc)
        raise
