"""HealthcareGuard FastAPI application.

Run locally with: uvicorn server:app --reload
Deploy on Render with: uvicorn server:app --host 0.0.0.0 --port $PORT
"""
from __future__ import annotations

import os
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from healthcareguard.guard import Guard
from healthcareguard.policy import ActionContext, ROLES, SOURCES
from healthcareguard.patient_risk import detect_patient_risk

APP_VERSION = "2.0.0-prototype"
ROOT = __import__("pathlib").Path(__file__).resolve().parent
app = FastAPI(
    title="HealthcareGuard API",
    description="Research prototype for screening healthcare prompts for safeguard-bypass and patient-risk signals.",
    version=APP_VERSION,
    docs_url="/api/docs",
    redoc_url=None,
)

allowed_origins = [x.strip() for x in os.getenv("ALLOWED_ORIGINS", "https://healthcaregaurd.netlify.app").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"chrome-extension://[a-p]{32}",
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=600,
)

guard = Guard.load_or_train()


class ScreenRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    role: str = "unknown"
    verified: bool = False
    source: str = "user"
    flags: List[str] = Field(default_factory=list)


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "service": "HealthcareGuard",
        "version": APP_VERSION,
        "model_loaded": guard is not None,
        "safeguard_bypass_threshold": guard.threshold,
        "patient_risk_screening": "transparent_rules_v1",
        "note": "Research prototype; not clinically validated.",
    }


@app.post("/api/screen")
def screen(req: ScreenRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Enter text to screen.")

    context = ActionContext(
        actor_role=req.role if req.role in ROLES else "unknown",
        authorisation_verified=req.verified,
        content_source=req.source if req.source in SOURCES else "user",
        patient_flags=req.flags[:5],
    )
    safeguard_result = guard.screen(text, context).as_dict()
    patient_risk = detect_patient_risk(text)

    urgent = patient_risk["level"] == "urgent"
    block = safeguard_result["decision"] == "BLOCK"
    high = patient_risk["level"] == "high" or safeguard_result["risk"] == "High"
    review = patient_risk["level"] == "review" or safeguard_result["decision"] == "REVIEW"
    should_alert = bool(patient_risk["detected"] or safeguard_result["flagged"] or safeguard_result["decision"] in ("BLOCK", "REVIEW"))

    if urgent:
        alert_level, alert_title = "urgent", "Urgent healthcare risk signal"
        alert_summary = "This prompt may describe an emergency or overdose. Do not wait for an AI response; seek urgent real-world help."
    elif block:
        alert_level, alert_title = "high", "Clinical safeguard bypass detected"
        alert_summary = safeguard_result["explanation"]
    elif high:
        alert_level, alert_title = "high", "Potential healthcare risk detected"
        alert_summary = "Review the medication or allergy concern with an appropriate healthcare professional."
    elif should_alert or review:
        alert_level, alert_title = "review", "Healthcare prompt needs review"
        alert_summary = "A pattern associated with a medication or clinical safety concern was detected."
    else:
        alert_level, alert_title = "clear", "No listed risk pattern detected"
        alert_summary = "No configured pattern was matched. This does not establish that the prompt or a later AI response is safe."

    safeguard_result["patient_risk"] = patient_risk
    safeguard_result["combined_alert"] = {
        "alert": should_alert,
        "level": alert_level,
        "title": alert_title,
        "summary": alert_summary,
    }
    safeguard_result["screening_disclaimer"] = (
        "HealthcareGuard is an experimental prompt-screening aid. It is not a medical device, "
        "does not provide diagnosis or triage, does not verify drug interactions, and cannot guarantee safety. "
        "Do not enter identifiable patient information."
    )
    return safeguard_result


@app.get("/")
def index():
    return FileResponse(ROOT / "index.html")


@app.get("/extension")
def extension_info():
    return FileResponse(ROOT / "install.html")
