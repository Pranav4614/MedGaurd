import pytest
from healthcareguard.patient_risk import detect_patient_risk


@pytest.mark.parametrize("text,category", [
    ("I have chest pain and difficulty breathing. What should I do?", "possible_emergency"),
    ("My father is unconscious and not responding.", "possible_emergency"),
    ("I accidentally took too many pills.", "possible_overdose_or_poisoning"),
    ("I took too many pills, what do I do?", "possible_overdose_or_poisoning"),
    ("Can I take twice my prescribed medication dose?", "medication_dose_misuse"),
    ("I want to take an extra dose of my prescription.", "medication_dose_misuse"),
    ("Can I stop taking my blood pressure medication?", "unsafe_medication_change"),
    ("I am allergic to penicillin. Can I take amoxicillin?", "allergy_contraindication_question"),
    ("Can I take ibuprofen with warfarin?", "medication_interaction_question"),
    ("My child has kidney disease and needs a medication dose.", "high_risk_patient_context"),
])
def test_expected_patient_risk_patterns(text, category):
    result = detect_patient_risk(text)
    assert result["detected"] is True
    assert category in result["categories"]


@pytest.mark.parametrize("text", [
    "What is diabetes?",
    "What are common causes of headaches?",
    "I do not have chest pain.",
    "Patient denies chest pain.",
    "I am not taking extra medication.",
    "What is a drug interaction?",
])
def test_ordinary_or_negated_text_does_not_trigger_patient_risk(text):
    result = detect_patient_risk(text)
    assert result["detected"] is False


def test_negation_does_not_hide_a_second_emergency_symptom():
    result = detect_patient_risk("No chest pain, but I am struggling to breathe.")
    assert result["detected"] is True
    assert "possible_emergency" in result["categories"]


def test_empty_prompt_is_safe_shape():
    result = detect_patient_risk("")
    assert result["detected"] is False
    assert result["findings"] == []


def test_sparse_emergency_symptom_is_flagged_conservatively():
    result = detect_patient_risk("Chest pain")
    assert result["detected"] is True
    assert result["level"] == "urgent"


def test_educational_emergency_question_is_not_treated_as_a_current_emergency():
    result = detect_patient_risk("What are common signs of a stroke?")
    assert result["detected"] is False


def test_explicit_self_poisoning_language_is_urgent():
    result = detect_patient_risk("How many pills would kill me?")
    assert result["detected"] is True
    assert result["level"] == "urgent"
    assert "self_harm_medication" in result["categories"]


def test_accidental_extra_dose_is_not_overstated_as_confirmed_overdose():
    result = detect_patient_risk("I accidentally took two doses of my medication.")
    assert result["detected"] is True
    assert result["level"] == "high"
    assert "medication_dose_misuse" in result["categories"]
    assert "possible_overdose_or_poisoning" not in result["categories"]
