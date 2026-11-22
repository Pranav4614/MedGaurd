"""HealthcareGuard FastAPI application."""
from __future__ import annotations
import os
from pathlib import Path
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from healthcareguard.guard import Guard
from healthcareguard.policy import ActionContext, ROLES, SOURCES
from healthcareguard.patient_risk import detect_patient_risk
from healthcareguard.patient_risk_ml import PatientRiskML
from healthcareguard.substance_risk_rules import detect_substance_risk

APP_VERSION = "2.2.0-prototype"
app = FastAPI(title="HealthcareGuard API", description="Research prototype for screening healthcare prompts for safeguard-bypass and patient-risk signals.", version=APP_VERSION, docs_url="/api/docs", redoc_url=None)
allowed_origins = [x.strip() for x in os.getenv("ALLOWED_ORIGINS", "https://secframework.vercel.app").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_origin_regex=r"chrome-extension://[a-p]{32}", allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type", "Authorization"], max_age=600)

guard = None
guard_load_error = None
model_inference_error = None
try:
    guard = Guard.load()
    guard.screen("What is diabetes?", ActionContext())
except Exception as exc:
    guard = None
    guard_load_error = type(exc).__name__
    model_inference_error = type(exc).__name__

patient_risk_ml_engine = None
patient_risk_ml_load_error = None
try:
    patient_risk_ml_engine = PatientRiskML()
except Exception as exc:
    patient_risk_ml_load_error = type(exc).__name__


class ScreenRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    role: str = "unknown"
    verified: bool = False
    source: str = "user"
    flags: List[str] = Field(default_factory=list)


@app.get("/api/health")
def health():
    return {
        "ok": True, "status": "ready" if guard is not None else "degraded",
        "service": "HealthcareGuard", "version": APP_VERSION,
        "model_loaded": guard is not None,
        "safeguard_bypass_threshold": guard.threshold if guard is not None else None,
        "detectors": {
            "clinical_safeguard_model": "ready" if guard is not None else "unavailable",
            "patient_risk_rules": "ready", "substance_risk_rules": "ready",
            "patient_risk_ml": "ready" if patient_risk_ml_engine is not None else "unavailable",
        },
        "model_load_error": guard_load_error,
        "model_inference": "ready" if guard is not None and model_inference_error is None else "unavailable",
        "patient_risk_screening": "transparent_rules_v1",
        "substance_risk_screening": "transparent_substance_rules_v1",
        "patient_risk_ml_version": "patient-risk-substance-v2.1" if patient_risk_ml_engine is not None else None,
        "patient_risk_ml_load_error": patient_risk_ml_load_error,
        "note": "Research prototype; not clinically validated.",
    }


@app.post("/api/screen")
def screen(req: ScreenRequest):
    if guard is None:
        raise HTTPException(status_code=503, detail="The clinical safeguard model is unavailable. This prompt was not screened.")
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Enter text to screen.")
    context = ActionContext(actor_role=req.role if req.role in ROLES else "unknown",
        authorisation_verified=req.verified, content_source=req.source if req.source in SOURCES else "user",
        patient_flags=req.flags[:5])
    safeguard_result = guard.screen(text, context).as_dict()
    patient_risk = detect_patient_risk(text)
    substance_risk = detect_substance_risk(text)

    if patient_risk_ml_engine is not None:
        try:
            patient_risk_ml_result = patient_risk_ml_engine.predict(text)
        except Exception as exc:
            patient_risk_ml_result = {"available": False, "detected": False, "category": None,
                "predicted_category": None, "risk_score": None, "risk_threshold": None,
                "category_confidence": None, "suggested_severity": "unavailable",
                "label": "Experimental ML layer unavailable",
                "message": "Other screening layers completed; the experimental model failed for this request.",
                "category_source": "unavailable", "error": type(exc).__name__, "synthetic_training_only": True}
    else:
        patient_risk_ml_result = {"available": False, "detected": False, "category": None,
            "predicted_category": None, "risk_score": None, "risk_threshold": None,
            "category_confidence": None, "suggested_severity": "unavailable",
            "label": "Experimental ML layer unavailable",
            "message": "Other screening layers remain available; the experimental model did not load.",
            "category_source": "unavailable", "synthetic_training_only": True}

    # Deterministic emergency rules alone can set urgent. ML can add review only.
    urgent = patient_risk["level"] == "urgent" or substance_risk["level"] == "urgent"
    block = safeguard_result["decision"] == "BLOCK"
    high = patient_risk["level"] == "high" or safeguard_result["risk"] == "High"
    ml_review = bool(patient_risk_ml_result.get("detected", False))
    should_alert = bool(patient_risk["detected"] or substance_risk["detected"] or ml_review
        or safeguard_result["flagged"] or safeguard_result["decision"] in ("BLOCK", "REVIEW"))

    if urgent:
        alert_level, alert_title = "urgent", "Urgent healthcare risk signal"
        alert_summary = "The deterministic rule layer found wording that may describe a medical emergency. Do not wait for an AI response; seek urgent real-world medical help."
    elif block:
        alert_level, alert_title = "high", "Clinical safeguard bypass detected"
        alert_summary = safeguard_result["explanation"]
    elif high:
        alert_level, alert_title = "high", "Potential healthcare risk detected"
        alert_summary = "Review the medication or clinical safety concern with an appropriate healthcare professional."
    elif should_alert or safeguard_result["decision"] == "REVIEW":
        alert_level, alert_title = "review", "Healthcare prompt needs review"
        alert_summary = "A patient-risk rule or experimental ML wording signal was detected. This is not a diagnosis; review the findings."
    else:
        alert_level, alert_title = "clear", "No listed risk pattern detected"
        alert_summary = "No configured risk pattern was matched. This does not establish that the prompt or a later AI response is safe."

    sources = []
    if safeguard_result["flagged"] or safeguard_result["decision"] in ("BLOCK", "REVIEW"): sources.append("clinical_safeguard_model")
    if patient_risk["detected"]: sources.append("patient_risk_rules")
    if substance_risk["detected"]: sources.append("substance_risk_rules")
    if ml_review: sources.append("patient_risk_ml")
    safeguard_result["patient_risk"] = patient_risk
    safeguard_result["substance_risk"] = substance_risk
    safeguard_result["patient_risk_ml"] = patient_risk_ml_result
    safeguard_result["combined_alert"] = {"alert": should_alert, "level": alert_level,
        "title": alert_title, "summary": alert_summary, "sources": sources}
    safeguard_result["screening_disclaimer"] = (
        "HealthcareGuard is an experimental prompt-screening aid, not a medical device. "
        "It does not diagnose substance use disorder, provide clinical triage, verify drug interactions, "
        "or guarantee safety. All ML training examples are synthetic and model scores are not calibrated "
        "probabilities. Do not enter identifiable patient information.")
    return safeguard_result


@app.get("/")
def index():
    dashboard = Path(__file__).parent / "index.html"
    if dashboard.is_file(): return FileResponse(dashboard)
    return {"service": "HealthcareGuard API", "docs": "/api/docs", "health": "/api/health"}


@app.get("/extension")
def extension_info():
    guide = Path(__file__).parent / "install.html"
    if guide.is_file(): return FileResponse(guide)
    return {"message": "Extension install guide is not included in this deployment."}
