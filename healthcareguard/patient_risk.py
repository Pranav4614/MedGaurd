"""Patient-facing healthcare risk screening.

This is a transparent, deterministic phrase/context screening layer. It is NOT a
clinical model, diagnosis system, drug-interaction database, or emergency triage
service. It deliberately reports matched signals rather than claiming accuracy.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Pattern:
    category: str
    label: str
    severity: str
    regex: str
    message: str
    needs_patient_context: bool = False


# Patterns focus on expressed intent/symptoms, not isolated medical vocabulary.
PATTERNS: tuple[Pattern, ...] = (
    Pattern("possible_emergency", "Possible emergency symptoms", "urgent",
            r"\b(chest pain|chest pressure|difficulty breathing|trouble breathing|shortness of breath|struggling to breathe|gasping for air|can't breathe|cannot breathe|not breathing|severe bleeding|uncontrolled bleeding|unconscious|passed out|cannot wake|blue lips|choking|face drooping|facial droop|sudden weakness|one-sided weakness|slurred speech|sudden confusion|sudden severe headache|heart attack|signs? of stroke|seizure|severe allergic reaction|anaphylaxis)\b",
            "The wording may describe a time-critical medical problem. Do not wait for an AI response; contact local emergency services or urgent medical care now.",
            True),
    Pattern("possible_overdose_or_poisoning", "Possible overdose or poisoning", "urgent",
            r"\b(took too many (pills?|tablets?|doses?)|taken too many (pills?|tablets?|doses?)|accidentally took too many|overdosed|overdose on|swallowed (poison|bleach|a toxic)|drank (poison|bleach|a toxic)|poisoned myself|took too much medication)\b",
            "Possible overdose or poisoning language detected. Seek urgent help from local emergency services or a poison-control service; do not wait for an AI response."),
    Pattern("self_harm_medication", "Possible intentional self-poisoning", "urgent",
            r"\b(want to (overdose|take all my pills)|planning to (overdose|poison myself)|take (all|the rest of) my (pills|medication)|kill myself with (pills|medication|drugs)|how many.{0,45}(pills?|tablets?|medication|medicine|drugs?).{0,30}(kill|die|fatal|lethal)|what (amount|dose).{0,45}(medication|medicine|drug|pills?).{0,30}(kill|die|fatal|lethal))\b",
            "This may indicate immediate self-harm danger. Contact local emergency services or a trusted person who can stay with you now."),
    Pattern("medication_dose_misuse", "Potential medication dose misuse", "high",
            r"\b(can i|should i|is it safe to|i want to|i plan to|i am going to|i took|i have taken|i am taking|i accidentally took|what happens if i|what if i)\b.{0,75}\b(double|twice|triple|three times|extra|more than prescribed|two|2|three|3)\b.{0,55}\b(dose|dosage|pills?|tablets?|medication|medicine)\b|\b(overdose|overdosed|higher dose than prescribed|took too much (of my )?(medication|medicine|drug))\b",
            "The prompt may involve taking more medication than prescribed. Do not change a dose based on an AI response; contact the prescriber or pharmacist."),
    Pattern("unsafe_medication_change", "Potentially unsafe medication change", "review",
            r"\b(?:(?:can|should) i|is it safe to|i want to|i plan to|i am going to)\s+(?:stop|skip|discontinue|quit|reduce|increase|change|adjust|pause|withhold)\b.{0,55}\b(medication|medicine|prescription|pills?|tablets?|antibiotics?|insulin|blood pressure medicine|antidepressants?|dose|dosage)\b|\b(i stopped|i have stopped|i am stopping|stopped|stopping|skipped|skipping|discontinued|quit|reduced|increased|changed|adjusted|withheld|paused)\b.{0,55}\b(medication|medicine|prescription|pills?|tablets?|antibiotics?|insulin|blood pressure medicine|antidepressants?|dose|dosage)\b",
            "Stopping, skipping or changing prescribed treatment may carry risks. Confirm any medication change with the prescriber or pharmacist."),
    Pattern("allergy_contraindication_question", "Possible allergy or contraindication concern", "high",
            r"\b(allergic to|allergy to|severe allergy|anaphylaxis)\b.{0,90}\b(can i take|should i take|safe to take|prescribed|taking|take)\b|\b(can i take|should i take|safe to take)\b.{0,70}\b(despite|although|even though)\b.{0,35}\b(allerg|reaction)\b",
            "The prompt mentions a possible allergy or contraindication. Do not use an AI response to override an allergy warning; check with a clinician or pharmacist."),
    Pattern("medication_interaction_question", "Medication interaction question", "review",
            r"\b(can i|should i|is it safe to|safe to)\b.{0,65}\b(take|mix|combine|use)\b.{0,65}\b(with|together|at the same time)\b|\b(interact|interaction)\b.{0,55}\b(medication|medicine|drug|supplement)\b",
            "A possible drug or supplement interaction is being asked about. This checker does not contain a complete interaction database; confirm with a pharmacist or prescriber."),
    Pattern("high_risk_patient_context", "Higher-risk patient context", "review",
            r"\b(pregnan(t|cy)|breastfeeding|my (child|baby|infant)|my (mother|father) is (over|aged)|kidney disease|liver disease|renal impairment|on blood thinners|taking warfarin|taking insulin|immunocompromised)\b.{0,100}\b(dose|dosage|medicine|medication|drug|treatment|symptom|pain)\b",
            "The prompt combines a potentially higher-risk patient context with a treatment or symptom question. Seek advice from an appropriate healthcare professional."),
)

NEGATION_RE = re.compile(r"\b(no|not|never|denies|without|isn't|wasn't|aren't|don't|doesn't|didn't|haven't|hasn't|hadn't)\b", re.I)
BOUNDARY_RE = re.compile(r"[,.!?;\n]|\bbut\b|\bhowever\b|\balthough\b", re.I)
EDUCATIONAL_CONTEXT_RE = re.compile(r"\b(define|definition|explain|what is|what are|symptoms|signs of|risk factors|causes of|for an exam|in general|for a school project|overview of)\b", re.I)


def _not_negated(text: str, start: int) -> bool:
    """Check the local clause immediately before a phrase for explicit negation."""
    prefix = text[max(0, start - 85):start]
    boundaries = list(BOUNDARY_RE.finditer(prefix))
    if boundaries:
        prefix = prefix[boundaries[-1].end():]
    return not bool(NEGATION_RE.search(prefix))


def _patient_context_required(text: str, match: re.Match[str]) -> bool:
    # Educational or definitional wording should not be treated as a current
    # symptom report unless the prompt also describes a real person/current state.
    if EDUCATIONAL_CONTEXT_RE.search(text) and not re.search(
        r"\b(i have|i'm having|i am having|my symptoms|right now|currently|someone is|patient has)\b", text, re.I
    ):
        return False
    # For sparse symptom prompts (for example, just "chest pain"), prefer a
    # conservative warning over silently treating it as harmless.
    return True


def detect_patient_risk(text: str) -> dict:
    """Return transparent patient-risk screening findings for a prompt.

    No probability/accuracy score is emitted: the detector is a rule-based prototype.
    """
    if not isinstance(text, str):
        text = ""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return {
            "detected": False, "level": "none_detected", "categories": [],
            "findings": [], "disclaimer": "No text was supplied.",
            "method": "transparent_rules_v1",
        }

    findings: list[dict] = []
    for pattern in PATTERNS:
        matches = [m for m in re.finditer(pattern.regex, text, re.I) if _not_negated(text, m.start())]
        if not matches:
            continue
        match = matches[0]
        if pattern.needs_patient_context and not _patient_context_required(text, match):
            continue
        findings.append({
            "category": pattern.category,
            "label": pattern.label,
            "severity": pattern.severity,
            "matched_phrase": match.group(0),
            "message": pattern.message,
        })

    order = {"urgent": 0, "high": 1, "review": 2}
    findings.sort(key=lambda f: order.get(f["severity"], 9))
    levels = {f["severity"] for f in findings}
    level = "urgent" if "urgent" in levels else "high" if "high" in levels else "review" if findings else "none_detected"
    return {
        "detected": bool(findings),
        "level": level,
        "categories": list(dict.fromkeys(f["category"] for f in findings)),
        "findings": findings,
        "disclaimer": "Pattern-based screening only. It is not a diagnosis, triage decision, medication-interaction check, or guarantee of safety.",
        "method": "transparent_rules_v1",
    }
