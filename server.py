"""HealthcareGuard FastAPI application with GitHub/Render folder-layout detection.

This replacement fixes deployments where GitHub contains an extra
`healthcareguard_project/` directory above the actual application files.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Locate the actual project root before importing the healthcareguard package.
# Supports either of these layouts:
#   repository/server.py + repository/healthcareguard/
#   repository/server.py + repository/healthcareguard_project/healthcareguard/
# ---------------------------------------------------------------------------
SERVER_DIR = Path(__file__).resolve().parent


def _is_project_root(path: Path) -> bool:
    return (
        (path / "healthcareguard" / "__init__.py").is_file()
        and (path / "healthcareguard" / "guard.py").is_file()
    )


_candidates = [
    SERVER_DIR,
    SERVER_DIR / "healthcareguard_project",
    SERVER_DIR / "HealthcareGuard",
    SERVER_DIR / "app",
]

# Also inspect one directory level below the server file. This covers a
# repository where the extracted project folder was uploaded as a subfolder.
try:
    _candidates.extend(p for p in SERVER_DIR.iterdir() if p.is_dir() and not p.name.startswith("."))
except OSError:
    pass

PROJECT_ROOT = next((p.resolve() for p in _candidates if _is_project_root(p)), None)

if PROJECT_ROOT is None:
    raise ModuleNotFoundError(
        "HealthcareGuard package folder not found. Expected a 'healthcareguard' "
        "folder containing __init__.py and guard.py beside server.py, or inside "
        "a 'healthcareguard_project' subfolder. In Render, set Root Directory to "
        "'healthcareguard_project' if that is where server.py and healthcareguard/ live."
    )

# Ensure Python can import the package even when the repository has an extra folder.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from healthcareguard.guard import Guard  # noqa: E402
from healthcareguard.policy import ActionContext, ROLES, SOURCES  # noqa: E402
from healthcareguard.patient_risk import detect_patient_risk  # noqa: E402


APP_VERSION = "2.0.0-prototype"
app = FastAPI(
    title="HealthcareGuard API",
    description=(
        "Research prototype for screening healthcare prompts for safeguard-bypass "
        "and patient-risk signals."
    ),
    version=APP_VERSION,
    docs_url="/api/docs",
    redoc_url=None,
)

allowed_origins = [
    item.strip()
    for item in os.getenv(
        "ALLOWED_ORIGINS", "https://healthcaregaurd.netlify.app"
    ).split(",")
    if item.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"chrome-extension://[a-p]{32}",
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=600,
)

# Load the existing trained safeguard-bypass model once at startup.
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
    should_alert = bool(
        patient_risk["detected"]
        or safeguard_result["flagged"]
        or safeguard_result["decision"] in ("BLOCK", "REVIEW")
    )

    if urgent:
        alert_level = "urgent"
        alert_title = "Urgent healthcare risk signal"
        alert_summary = (
            "This prompt may describe an emergency or overdose. Do not wait for an "
            "AI response; seek urgent real-world help."
        )
    elif block:
        alert_level = "high"
        alert_title = "Clinical safeguard bypass detected"
        alert_summary = safeguard_result["explanation"]
    elif high:
        alert_level = "high"
        alert_title = "Potential healthcare risk detected"
        alert_summary = (
            "Review the medication or allergy concern with an appropriate "
            "healthcare professional."
        )
    elif should_alert or review:
        alert_level = "review"
        alert_title = "Healthcare prompt needs review"
        alert_summary = (
            "A pattern associated with a medication or clinical safety concern "
            "was detected."
        )
    else:
        alert_level = "clear"
        alert_title = "No listed risk pattern detected"
        alert_summary = (
            "No configured pattern was matched. This does not establish that "
            "the prompt or a later AI response is safe."
        )

    safeguard_result["patient_risk"] = patient_risk
    safeguard_result["combined_alert"] = {
        "alert": should_alert,
        "level": alert_level,
        "title": alert_title,
        "summary": alert_summary,
    }
    safeguard_result["screening_disclaimer"] = (
        "HealthcareGuard is an experimental prompt-screening aid. It is not a "
        "medical device, does not provide diagnosis or triage, does not verify "
        "drug interactions, and cannot guarantee safety. Do not enter identifiable "
        "patient information."
    )
    return safeguard_result


@app.get("/")
def index():
    page = PROJECT_ROOT / "static" / "index.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="Dashboard file not found.")
    return FileResponse(page)


@app.get("/extension")
def extension_info():
    page = PROJECT_ROOT / "extension" / "install.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="Extension installation page not found.")
    return FileResponse(page)
