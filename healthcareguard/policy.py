"""Context-aware policy layer.

The text classifier answers "does this look like an attempt to subvert a clinical safeguard?".
It cannot answer "is this actor allowed to do this?". That is a property of the *deployment
context* (who is acting, is authorisation verified by an external system, where did the text
come from), so it is passed in as structured input and never inferred from claims inside the text.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
from .taxonomy import RX, DOSE, INTER, EVID, DX, ADD, REC, PREG, GENERAL

ROLES = ["unknown", "patient", "nurse", "clinician", "prescriber", "pharmacist", "records_officer",
         "specialist", "administrator", "system"]
SOURCES = ["user", "retrieved_document", "tool_output", "email_attachment"]
CLINICAL_ROLES = {"clinician", "prescriber", "pharmacist", "records_officer", "specialist"}
HIGH_RISK_FLAGS = ["pregnant", "paediatric", "frail_elderly", "renal_impairment", "controlled_drug"]

ROLE_PERMISSIONS = {
    RX: {"prescriber", "clinician"},
    DOSE: {"prescriber", "clinician", "pharmacist"},
    INTER: {"prescriber", "clinician", "pharmacist"},
    EVID: set(),                      # nobody is authorised to fabricate evidence
    DX: {"clinician", "prescriber", "specialist"},
    ADD: {"prescriber", "clinician", "specialist"},
    REC: {"clinician", "records_officer"},
    PREG: {"prescriber", "clinician", "specialist"},
    GENERAL: set(),
}


@dataclass
class ActionContext:
    actor_role: str = "unknown"
    authorisation_verified: bool = False      # verified by an external system, NOT claimed in the text
    audit_trail_enabled: bool = True
    content_source: str = "user"              # user | retrieved_document | tool_output | email_attachment
    patient_flags: List[str] = field(default_factory=list)
    authoriser: Optional[str] = None


@dataclass
class PolicyOutcome:
    decision: str            # ALLOW | ALLOW_WITH_AUDIT | REVIEW | BLOCK
    notes: List[str]


def apply_policy(*, score: float, threshold: float, category: str, feats: dict, ctx: ActionContext) -> PolicyOutcome:
    flagged = score >= threshold
    action = bool(feats["state_x_target"]) and not feats["info_request"]
    hiding = bool(feats["conceal"] or feats["falsify"] or feats["exfil"])
    permitted = (ctx.authorisation_verified and ctx.actor_role in ROLE_PERMISSIONS.get(category, set())
                 and ctx.audit_trail_enabled)
    clinical_authorised = ctx.authorisation_verified and ctx.actor_role in CLINICAL_ROLES and ctx.audit_trail_enabled
    notes: List[str] = []

    # 1. text that arrived via a document/tool must not carry instructions
    if ctx.content_source != "user" and (flagged or action or feats["embedded"]):
        return PolicyOutcome("BLOCK", ["instruction_in_untrusted_content: text from a document or tool output "
                                       "must not issue clinical instructions"])
    # 2. legitimate, authorised changes do not need to be hidden or falsified
    if flagged and hiding:
        return PolicyOutcome("BLOCK", ["concealment_or_falsification: authorised changes do not need to be hidden"])
    if flagged and category == EVID:
        return PolicyOutcome("BLOCK", ["evidence_fabrication: no role may fabricate clinical evidence"])

    if flagged:
        if permitted and feats["authorised"]:
            decision = "ALLOW_WITH_AUDIT"
            notes.append(f"verified {ctx.actor_role} authorisation + documented justification in the request; log for audit")
        elif permitted:
            decision = "REVIEW"
            notes.append("verified authorised actor, but the request bypasses a safeguard without a documented reason")
        elif score >= 0.8 or feats["unauthorised"] or feats["bypass"]:
            decision = "BLOCK"
            notes.append("safeguard-bypass pattern with no verified authorisation for this actor/category")
        else:
            decision = "REVIEW"
            notes.append("adversarial score above threshold; no verified authorisation")
    elif action:
        if clinical_authorised:
            decision = "ALLOW_WITH_AUDIT"
            notes.append("state-changing clinical action by a verified clinical actor; log for audit")
        else:
            decision = "REVIEW"
            notes.append("state-changing clinical action requires verified authorisation (none provided)")
    else:
        decision = "ALLOW"

    if decision == "ALLOW_WITH_AUDIT" and action and ctx.patient_flags:
        decision = "REVIEW"
        notes.append("patient has high-risk flags (" + ", ".join(ctx.patient_flags) + "): route to specialist verification")
    return PolicyOutcome(decision, notes)
