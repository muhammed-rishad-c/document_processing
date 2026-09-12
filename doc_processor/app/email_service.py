import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone

from dotenv import load_dotenv

from .models import Company, Lead

load_dotenv()

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_FROM_ADDRESS = os.getenv("SMTP_FROM_ADDRESS", SMTP_USER)
SMTP_USE_SSL = os.getenv("SMTP_USE_SSL", "false").lower() == "true"


def _build_message(company: Company, lead: Lead, department_email: str) -> MIMEMultipart:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    category_display = lead.category_name or "General"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"New lead for {company.name} — {category_display}"
    msg["From"] = SMTP_FROM_ADDRESS
    msg["To"] = department_email

    body_lines = [
        f"New lead captured via the {company.name} chat widget.",
        "",
        f"Category: {category_display}",
        f"Name: {lead.name}",
        f"Email: {lead.email}",
        f"Phone: {lead.phone or 'Not provided'}",
        "",
        "Original question:",
        lead.question or "(not recorded)",
        "",
        f"Captured: {timestamp}",
    ]
    msg.attach(MIMEText("\n".join(body_lines), "plain"))
    return msg


def send_lead_notification(company: Company, lead: Lead, department_email: str) -> bool:
    """Sends a lead notification email to the resolved department address.
    Never raises — any SMTP/config failure is logged and returns False, so a
    flaky mail server can never break the visitor-facing chat response.
    department_email is passed in already-resolved (see widget.py's
    _resolve_department) rather than re-derived here, so there is a single
    source of truth for the fallback-to-default logic."""
    if not SMTP_HOST or not SMTP_USER or not SMTP_PASSWORD:
        print("[send_lead_notification] SMTP not configured, skipping send.")
        return False

    if not department_email:
        print(f"[send_lead_notification] No destination email for lead {lead.id}, skipping send.")
        return False

    try:
        msg = _build_message(company, lead, department_email)

        if SMTP_USE_SSL:
            server = smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=10)
        else:
            server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10)

        with server:
            if not SMTP_USE_SSL:
                server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_FROM_ADDRESS, [department_email], msg.as_string())

        return True
    except Exception as e:
        print(f"[send_lead_notification] Failed to send lead notification for lead {lead.id}: {e}")
        return False