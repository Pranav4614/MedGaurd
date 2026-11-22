"""Inference wrapper for experimental HealthcareGuard substance-risk ML v2.1.

It screens wording only. Scores are not calibrated probabilities. It must not diagnose
addiction, infer character/intent, assess intoxication/withdrawal, or replace urgent rules.
"""
from __future__ import annotations
from pathlib import Path
import re
import joblib

MODEL_PATH = Path(__file__).with_name("patient_risk_substance_model.joblib")

# Transparent supplemental overrides handle a few high-value distinctions where
# broad ML categories overlap. These are intentionally narrow and auditable.
MEDICATION_REDOSING_RE = re.compile(
    r"\b(urge|tempted|want|thinking of|feel like|plan to)\b.{0,65}\b(take|repeat|redose|another|extra)\b.{0,55}\b(medicine|medication|tablet|pill|dose|prescription)\b"
    r"|\b(take|taking|repeat|redose)\b.{0,55}\b(medicine|medication|tablet|pill|dose)\b.{0,35}\b(again|another|extra|early|before)\b",
    re.I,
)
MISINFORMATION_ACTION_RE = re.compile(
    r"(?:\b(viral post|social media|influencer|online claim|online rumou?r|forwarded message|forum post|unverified claim|video says|website says)\b.{0,140}\b(stop|stopping|ignore|replace|refuse|avoid|skip|discontinue|dismiss)\b.{0,85}\b(treatment|medicine|medication|doctor|clinician|vaccine|care|prescription)\b)"
    r"|(?:\b(stop|stopping|ignore|replace|refuse|avoid|skip|discontinue|dismiss)\b.{0,90}\b(treatment|medicine|medication|doctor|clinician|vaccine|care|prescription)\b.{0,100}\b(because|after|based on|following|since)\b.{0,90}\b(viral post|social media|influencer|online claim|online rumou?r|forwarded message|forum post|unverified claim|video|website)\b)",
    re.I,
)
STIGMA_COERCION_RE = re.compile(
    r"\b(sham(e|ing)|humiliat(e|ing)|threaten|punish|mock|force|coerce|embarrass)\b.{0,100}\b(addict|addiction|substance|drug use|relapse|recovery|rehab|treatment)\b"
    r"|\b(addict|addiction|substance|drug use|relapse|recovery|rehab|treatment)\b.{0,100}\b(sham(e|ing)|humiliat(e|ing)|threaten|punish|mock|force|coerce|embarrass)\b",
    re.I,
)
RECOVERY_SEEKING_RE = re.compile(
    r"\b(?:i want help|i need help|help me find|looking for|seeking|want to find|trying to find)\b.{0,120}\b(?:support|treatment|recovery|rehab|counselling|counseling|professional|service|help)\b"
    r"|\b(?:trying to quit|trying to stop|want to quit|want to stop|want to reduce)\b.{0,70}\b(?:drinking|alcohol|substance|drug use|using)\b", re.I)
EDUCATIONAL_QUERY_RE = re.compile(
    r"\b(?:what is|what are|define|definition of|explain|describe|in general|for a class|for an assignment|textbook|study guide|learn about|academic research|research paper|overview of)\b", re.I)
CURRENT_EVENT_RE = re.compile(
    r"\b(?:i have|i am having|i took|i used|i stopped|i am stopping|my (?:friend|partner|parent|family member) is|my (?:friend|partner|parent|family member) has|someone is|someone has|right now|currently)\b", re.I)
ACTIVE_SUBSTANCE_RISK_RE = re.compile(
    r"\b(?:cravings?|urge to (?:drink|use|take (?:the )?(?:drug|substance|opioid|recreational substance))|tempted to use|relapse|lapse|withdrawal|withdrawing|overdose|overdosed|poisoning|mixing substances|mixed substances|took too much|taking more than prescribed|self[- ]harm|kill myself|hurt myself|shaking and sweating after stopping|seizure|hallucinations?|unresponsive|barely responsive)\b", re.I)

SUPPORTING_OTHER_RE = re.compile(
    r"\b(my friend|my partner|my brother|my sister|my parent|my relative|my colleague|someone i care about|a family member|a loved one)\b.{0,150}\b(how can i (support|help|speak|talk)|what (can|should) i say|supportive thing to say|encourage them|help them find support)\b",
    re.I,
)
IMMEDIATE_DANGER_RE = re.compile(
    r"\b(overdose|poison(ed|ing)?|unconscious|unresponsive|barely responsive|cannot be woken|can't be woken|difficult to wake|blue lips|slow breathing|not breathing|seizure|collapsed|suicid(al|e)|kill myself|hurt myself|severe withdrawal)\b",
    re.I,
)

class PatientRiskML:
    def __init__(self, model_path: str | Path = MODEL_PATH):
        bundle = joblib.load(model_path)
        self.risk_model = bundle["risk_model"]
        self.category_model = bundle["category_model"]
        self.metadata = bundle.get("metadata", {})
        self.threshold = float(self.metadata.get("risk_threshold", 0.60))
        self.severity_map = dict(self.metadata.get("severity_map", {}))
        self.safe_categories = set(self.metadata.get("safe_categories", []))
        self.labels = dict(self.metadata.get("labels", {}))
        self.messages = dict(self.metadata.get("messages", {}))

    def predict(self, text: str) -> dict:
        text = str(text or "").strip()
        if not text:
            return {
                "available": True, "detected": False, "category": None,
                "predicted_category": None, "risk_score": 0.0,
                "risk_threshold": self.threshold, "category_confidence": 0.0,
                "suggested_severity": "none", "label": None, "message": None,
                "category_source": "none", "method": "binary_risk_plus_category_tfidf_v2",
                "synthetic_training_only": True,
            }

        classes = list(self.risk_model.classes_)
        risk_index = classes.index(1)
        risk_score = float(self.risk_model.predict_proba([text])[0][risk_index])
        category_probs = self.category_model.predict_proba([text])[0]
        category_idx = int(category_probs.argmax())
        predicted_category = str(self.category_model.classes_[category_idx])
        category_confidence = float(category_probs[category_idx])
        detected = risk_score >= self.threshold
        category_source = "category_model"
        override_reason = None

        if MEDICATION_REDOSING_RE.search(text):
            predicted_category = "medication_redosing_or_nonadherence"
            category_source = "supplemental_rule"
            override_reason = "explicit_medication_repeat_dose_language"
        elif MISINFORMATION_ACTION_RE.search(text):
            predicted_category = "acting_on_health_misinformation"
            category_source = "supplemental_rule"
            override_reason = "explicit_unverified_claim_plus_treatment_action"
        elif STIGMA_COERCION_RE.search(text):
            predicted_category = "stigmatizing_or_coercive_request"
            category_source = "supplemental_rule"
            override_reason = "explicit_stigma_or_coercion_language"

        # Explicit requests to support another person are not a danger alert by
        # themselves. Never apply this override to prompts with immediate-danger cues.
        supportive_other = bool(SUPPORTING_OTHER_RE.search(text)) and not bool(IMMEDIATE_DANGER_RE.search(text))
        recovery_seeking = bool(RECOVERY_SEEKING_RE.search(text)) and not bool(ACTIVE_SUBSTANCE_RISK_RE.search(text)) and not bool(IMMEDIATE_DANGER_RE.search(text))
        educational_safe = bool(EDUCATIONAL_QUERY_RE.search(text)) and not bool(CURRENT_EVENT_RE.search(text)) and not bool(ACTIVE_SUBSTANCE_RISK_RE.search(text)) and not bool(IMMEDIATE_DANGER_RE.search(text))
        if supportive_other:
            detected = False
            predicted_category = "supporting_someone_with_substance_use"
            category_source = "supplemental_safe_intent_rule"
            override_reason = "explicit_support_seeking_for_another_person_without_emergency_cues"
        elif recovery_seeking:
            detected = False
            predicted_category = "recovery_support_or_harm_reduction"
            category_source = "supplemental_safe_intent_rule"
            override_reason = "explicit_recovery_help_seeking_without_active_danger_cues"
        elif educational_safe:
            detected = False
            predicted_category = "health_misinformation_verification" if re.search(r"\b(claim|rumou?r|misinformation|post|viral|social media)\b", text, re.I) else "benign_general_health"
            category_source = "supplemental_safe_intent_rule"
            override_reason = "general_educational_question_without_current_patient_event"

        if detected and predicted_category in self.safe_categories:
            category = "uncertain_risk_signal"
            suggested_severity = "review"
            label = "Potential health-safety concern — review"
            message = "The wording may contain a safety concern, but the category is uncertain. Review the prompt rather than assuming it is safe."
        elif detected:
            category = predicted_category
            suggested_severity = self.severity_map.get(predicted_category, "review")
            label = self.labels.get(predicted_category, "Potential patient-safety concern")
            message = self.messages.get(predicted_category)
        else:
            category = None
            suggested_severity = "none"
            label = None
            message = self.messages.get(predicted_category) if predicted_category in self.safe_categories else None

        return {
            "available": True,
            "detected": bool(detected),
            "category": category,
            "predicted_category": predicted_category,
            "risk_score": round(risk_score, 4),
            "risk_threshold": self.threshold,
            "category_confidence": round(category_confidence, 4),
            "suggested_severity": suggested_severity,
            "label": label,
            "message": message,
            "category_source": category_source,
            "override_reason": override_reason,
            "method": "binary_risk_plus_category_tfidf_v2",
            "synthetic_training_only": True,
            "disclaimer": "Experimental wording classifier trained on synthetic prompts only. Its score is not a calibrated probability. It cannot diagnose addiction, determine intent, assess intoxication or withdrawal, or establish whether someone is safe.",
        }