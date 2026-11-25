"""Transparent substance-related rules. Wording signals only, not diagnosis or triage."""
from __future__ import annotations
import re
from dataclasses import dataclass

@dataclass(frozen=True)
class Rule:
    category: str
    label: str
    severity: str
    pattern: re.Pattern[str]
    message: str

RULES: tuple[Rule, ...] = (
    Rule("substance_related_self_harm", "Possible self-harm danger involving substances", "urgent",
        re.compile(r"\b(?:i want to|i plan to|i am going to|i might|i am thinking about)\s+(?:overdose|poison myself|take all (?:my )?(?:pills|medication|drugs)|hurt myself with (?:drugs|medication|substances)|end my life with (?:drugs|medication|pills))\b", re.I),
        "This wording may indicate immediate self-harm danger. Seek immediate real-world help from local emergency services and a trusted person who can stay with you."),
    Rule("severe_substance_withdrawal_emergency", "Possible severe withdrawal emergency", "urgent",
        re.compile(r"\b(?:withdrawal|withdrawing|stopped (?:drinking|using|taking)|stopping (?:alcohol|benzodiazepines?|sedatives?))\b.{0,120}\b(?:seizure|hallucinations?|severe confusion|disoriented|collapse|collapsed|cannot stay awake|slow breathing|not breathing)\b|\b(?:seizure|hallucinations?|severe confusion|collapse)\b.{0,110}\b(?:withdrawal|withdrawing|stopped drinking|stopped using|stopping alcohol|stopping benzodiazepines?|stopping sedatives?)\b", re.I),
        "Severe symptoms during or after stopping a substance can be dangerous. Seek emergency medical assessment now."),
    Rule("possible_substance_overdose_emergency", "Possible substance-related emergency", "urgent",
        re.compile(r"\b(?:barely responsive|unresponsive|unconscious|difficult to wake|hard to wake|cannot be woken|can't be woken|not breathing|slow breathing|blue lips|collapsed)\b.{0,130}\b(?:after|following|from|using|taking|drinking|substance|drug|alcohol|opioid|medicine)\b|\b(?:possible overdose|may have overdosed|might have overdosed|think (?:i|someone|my friend) (?:has )?overdosed|overdose emergency|poisoned after using)\b", re.I),
        "This may describe a time-critical poisoning or overdose emergency. Contact local emergency medical services now."),
    Rule("substance_withdrawal_concern", "Possible substance withdrawal concern", "review",
        re.compile(r"\b(?:i|my friend|my family member|someone i know)\b.{0,75}\b(?:stopped|stopping|quit|quitting|withdrew|withdrawing)\b.{0,80}\b(?:alcohol|drinking|substance|drug|sedative|benzodiazepine|opioid|medicine)\b|\b(?:withdrawal symptoms?|worried about withdrawal|shaking and sweating after stopping|tremors? after stopping)\b", re.I),
        "Possible withdrawal concern. Encourage prompt professional assessment; severe confusion, hallucinations, seizures, collapse, or breathing problems require emergency care."),
    Rule("substance_craving_or_relapse_risk", "Craving or relapse support concern", "review",
        re.compile(r"\b(?:i|my cravings|my urge|i am|i'm|i have|i've)\b.{0,100}\b(?:cravings?|urge to (?:drink|use)|urge to take (?:the )?(?:drug|substance|opioid|recreational substance)|tempted to use|afraid i will relapse|worried i may relapse|relapse risk|returned to use|had a lapse)\b|\b(?:i may relapse|i might relapse|i am close to relapsing|i feel like using again)\b", re.I),
        "The wording may describe cravings or concern about returning to use. Respond without judgement and encourage trusted or professional support."),
    Rule("dangerous_substance_combination_concern", "Potentially dangerous substance combination", "review",
        re.compile(r"\b(?:mix(?:ed|ing)?|combin(?:e|ed|ing)|used together|take together)\b.{0,110}\b(?:alcohol|drugs?|substances?|opioids?|sedatives?|benzodiazepines?|sleeping pills?|medication|medicine|prescription)\b|\b(?:alcohol|opioids?|sedatives?|benzodiazepines?|sleeping pills?|recreational drugs?)\b.{0,90}\b(?:with|together with|and)\b.{0,90}\b(?:medication|medicine|prescription|another substance|alcohol|opioid|sedative)\b", re.I),
        "A possible substance or medicine combination is being discussed. Do not reassure or recommend mixing; if already taken or symptomatic, seek urgent professional advice."),
    Rule("acting_on_health_misinformation", "Possible action based on unverified health information", "review",
        re.compile(r"\b(?:viral post|social media|influencer|online claim|forwarded message|unverified claim|video says|website says|someone online said)\b.{0,150}\b(?:stop|stopping|skip|skipping|ignore|replace|refuse|avoid|discontinue|cancel|quit)\b.{0,100}\b(?:treatment|medicine|medication|doctor|clinician|vaccine|care|prescription|therapy)\b|\b(?:stop|stopping|skip|skipping|ignore|replace|refuse|avoid|discontinue|cancel|quit)\b.{0,90}\b(?:treatment|medicine|medication|doctor|clinician|vaccine|care|prescription|therapy)\b.{0,110}\b(?:because|after|based on|following|since)\b.{0,100}\b(?:viral post|social media|influencer|online claim|forwarded message|video|website|rumour|rumor)\b", re.I),
        "The prompt may describe changing health decisions because of an unverified health claim. Verify it with reliable medical sources and discuss changes with a qualified professional."),
)

def detect_substance_risk(text: str) -> dict:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    findings = []
    for rule in RULES:
        match = rule.pattern.search(value)
        if match:
            findings.append({"category": rule.category, "label": rule.label, "severity": rule.severity, "matched_phrase": match.group(0), "message": rule.message})
    priority = {"urgent": 0, "high": 1, "review": 2}
    findings.sort(key=lambda item: priority[item["severity"]])
    return {"detected": bool(findings), "level": findings[0]["severity"] if findings else "none_detected",
        "categories": list(dict.fromkeys(item["category"] for item in findings)), "findings": findings,
        "method": "transparent_substance_rules_v1", "disclaimer": "Rule-based wording signals only; this is not diagnosis or clinical triage."}
