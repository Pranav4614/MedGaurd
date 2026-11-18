# Model and evaluation notes

## Layer 1: safeguard bypass

The existing `models/guard.joblib` artifact and existing `healthcareguard/guard.py` model/policy pipeline are retained. Do not change their threshold based on a few hand-picked examples. The reported score is a model score from this existing classifier, not a clinical risk probability.

## Layer 2: patient-facing signals

`healthcareguard/patient_risk.py` is a deterministic rules-based prototype. It is deliberately not described as a trained model. It detects configured text patterns and uses limited local negation/context handling. It does not understand all clinical context, validate a named drug pair against a reliable drug database, infer patient-specific severity, or reliably distinguish every educational example from a real patient scenario.

## Evaluation status

A small automated suite lives in `tests/`. Its tests are developer-authored checks that guard against obvious regressions; they are not a representative clinical dataset and are not a clinical validation study. No sensitivity, specificity, accuracy, PPV, NPV or safety claim is made.

## Work required before real-world healthcare deployment

1. Define intended use and out-of-scope use with clinical and regulatory experts.
2. Create a representative, de-identified and independently labelled dataset with appropriate consent/governance.
3. Have multiple qualified reviewers label each category and resolve disagreements.
4. Evaluate false positives and false negatives separately by category, language, demographic subgroup and prompt style.
5. Add versioned evaluation, threshold governance, monitoring, data retention controls, access control, rate limiting, audit/security review and incident handling.
6. Never let this prototype replace emergency services, professional judgement, or a validated clinical decision-support control.
