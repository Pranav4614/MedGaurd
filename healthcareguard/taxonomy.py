"""Attack taxonomy for HealthcareGuard."""

RX = "Prescription modification"
DOSE = "Dosage safety override"
INTER = "Drug interaction / contraindication bypass"
EVID = "Clinical evidence fabrication"
DX = "Diagnosis manipulation"
ADD = "Addiction / withdrawal manipulation"
REC = "Patient-record manipulation"
PREG = "Pregnancy / high-risk treatment bypass"
GENERAL = "General / unclear"

HEALTHCARE_CATEGORIES = [RX, DOSE, INTER, EVID, DX, ADD, REC, PREG]
ALL_CATEGORIES = HEALTHCARE_CATEGORIES + [GENERAL]
HIGH_SEVERITY = {RX, DOSE, INTER, REC, PREG, ADD}

DESCRIPTIONS = {
    RX: "Changing, switching or re-issuing a prescription without prescriber sign-off.",
    DOSE: "Overriding dose limits, skipping dose verification, or falsifying inputs to dose checks.",
    INTER: "Hiding or dismissing interaction, allergy or contraindication warnings; false reassurance.",
    EVID: "Inventing or misattributing studies, guidelines, citations or endorsements.",
    DX: "Forcing or falsifying a diagnosis, assessment, triage level or diagnosis code.",
    ADD: "Unsafe withdrawal/detox advice, hiding withdrawal risks, or obtaining controlled drugs deceptively.",
    REC: "Altering, deleting, backdating or disabling audit of patient records and results.",
    PREG: "Bypassing pregnancy, paediatric, frailty or other high-risk safeguards.",
    GENERAL: "Generic prompt injection, jailbreaks, data exfiltration and social engineering.",
}
