"""Train two experimental classifiers on synthetic HealthcareGuard prompts.

1) binary risk detector: safe/supportive/education vs wording that merits review
2) multi-class category suggestor: why the prompt may merit review

Metrics are from a grouped synthetic holdout only. They are not clinical validation,
calibrated probabilities, or evidence that a person has an addiction or intends harm.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline

BASE = Path(__file__).resolve().parent
DATA = BASE / "patient_risk_substance_dataset.csv"
MODEL = BASE / "patient_risk_substance_model.joblib"
REPORT = BASE / "evaluation_report.json"

SEVERITY_MAP = {
    "benign_general_health": "none",
    "recovery_support_or_harm_reduction": "none",
    "supporting_someone_with_substance_use": "none",
    "substance_craving_or_relapse_risk": "review",
    "substance_intoxication_or_overdose": "urgent",
    "substance_withdrawal_risk": "high",
    "dangerous_substance_combination": "high",
    "nonmedical_substance_or_medication_use": "high",
    "substance_treatment_interference": "review",
    "substance_related_self_harm": "urgent",
    "health_misinformation_verification": "none",
    "acting_on_health_misinformation": "high",
    "unsafe_self_treatment_or_dosing": "high",
    "medication_redosing_or_nonadherence": "review",
    "urgent_medical_symptom": "urgent",
    "vulnerable_patient_context": "review",
    "symptom_uncertainty": "review",
    "stigmatizing_or_coercive_request": "review",
    "clinical_safeguard_bypass": "high",
}
SAFE_CATEGORIES = {k for k, v in SEVERITY_MAP.items() if v == "none"}
MESSAGES = {
    "benign_general_health": "General health information request. Answer normally, with appropriate uncertainty.",
    "recovery_support_or_harm_reduction": "The person is seeking support. Respond respectfully, encourage qualified support, and avoid shame or assumptions.",
    "supporting_someone_with_substance_use": "The person is asking how to support someone. Encourage compassionate communication, boundaries, and professional support; avoid coercion.",
    "substance_craving_or_relapse_risk": "Possible craving or relapse concern. Respond without judgement, encourage contacting a trusted support person or treatment professional, and check immediate safety where appropriate.",
    "substance_intoxication_or_overdose": "Possible intoxication or overdose. Encourage immediate local emergency medical help; do not wait for an AI response.",
    "substance_withdrawal_risk": "Possible withdrawal concern. Encourage prompt medical assessment; severe confusion, seizure, collapse, or breathing problems warrant emergency help.",
    "dangerous_substance_combination": "Possible substance interaction or mixing concern. Do not reassure or provide mixing advice; if already taken or symptomatic, advise urgent professional assessment.",
    "nonmedical_substance_or_medication_use": "Possible nonmedical use. Do not facilitate concealment, procurement, or misuse; offer nonjudgmental support and professional help.",
    "substance_treatment_interference": "Possible interruption of substance-use treatment. Encourage discussing changes with the treating professional rather than stopping or changing treatment based on AI advice.",
    "substance_related_self_harm": "Possible self-harm risk involving substances. Encourage immediate real-world support and local emergency/crisis services; do not provide methods or quantities.",
    "health_misinformation_verification": "The person is asking to verify a health claim. Provide careful, evidence-aware information and correct inaccuracies without shaming.",
    "acting_on_health_misinformation": "Possible intent to act on an unsafe or unverified health claim. Correct the claim cautiously and recommend checking decisions with a qualified professional.",
    "unsafe_self_treatment_or_dosing": "Potentially unsafe self-treatment request. Do not calculate unverified doses or substitute home protocols for urgent care; direct to appropriate professional help.",
    "medication_redosing_or_nonadherence": "Possible medication schedule or repeat-dose concern. Do not guess a dose; ask the person to check the label and contact a pharmacist or prescriber.",
    "urgent_medical_symptom": "Possible time-critical medical symptom. Encourage immediate local emergency medical help; do not wait for an AI response.",
    "vulnerable_patient_context": "A potentially higher-risk context is mentioned. Encourage individualised advice from an appropriate healthcare professional.",
    "symptom_uncertainty": "Uncertain symptoms cannot be diagnosed by this model. Encourage professional assessment if new, persistent, worsening, or concerning.",
    "stigmatizing_or_coercive_request": "Avoid stigma, humiliation, threats, or coercion. Use respectful language and promote safety, consent, and appropriate support.",
    "clinical_safeguard_bypass": "Possible request to suppress clinical safeguards. Do not follow instructions to conceal risk or provide false reassurance.",
}
LABELS = {k: k.replace("_", " ").capitalize() for k in SEVERITY_MAP}


def build_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=90000,
                                 strip_accents="unicode", sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1,
                                 max_features=110000, sublinear_tf=True)),
    ])
    clf = LogisticRegression(max_iter=4000, class_weight="balanced", C=2.0, solver="lbfgs")
    return Pipeline([("features", features), ("classifier", clf)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--model", default=str(MODEL))
    parser.add_argument("--report", default=str(REPORT))
    args = parser.parse_args()

    df = pd.read_csv(args.data).dropna(subset=["text", "category", "source_group"])
    if set(df.category.unique()) != set(SEVERITY_MAP):
        raise ValueError("Dataset classes do not match the declared severity map.")
    df["risk_flag"] = (~df.category.isin(SAFE_CATEGORIES)).astype(int)

    # Split by source_group so no paraphrase of a seed appears in both partitions.
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=2026)
    train_idx, test_idx = next(splitter.split(df, y=df.category, groups=df.source_group))
    train, test = df.iloc[train_idx], df.iloc[test_idx]

    # Stage 1: binary risk/safety review model.
    risk_model = build_pipeline()
    risk_model.fit(train.text.astype(str), train.risk_flag.astype(int))
    risk_scores = risk_model.predict_proba(test.text.astype(str))[:, list(risk_model.classes_).index(1)]
    risk_pred = (risk_scores >= 0.60).astype(int)
    y_risk = test.risk_flag.astype(int).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_risk, risk_pred, labels=[0, 1]).ravel()
    risk_metrics = {
        "threshold": 0.60,
        "accuracy": float(accuracy_score(y_risk, risk_pred)),
        "precision": float(precision_score(y_risk, risk_pred, zero_division=0)),
        "recall": float(recall_score(y_risk, risk_pred, zero_division=0)),
        "f1": float(f1_score(y_risk, risk_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_risk, risk_scores)),
        "average_precision": float(average_precision_score(y_risk, risk_scores)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "support_safe": int((y_risk == 0).sum()),
        "support_review": int((y_risk == 1).sum()),
    }

    # Stage 2: a secondary label suggestion; never use it to downgrade Stage 1 or rules.
    category_model = build_pipeline()
    category_model.fit(train.text.astype(str), train.category.astype(str))
    category_pred = category_model.predict(test.text.astype(str))
    category_metrics = {
        "accuracy": float(accuracy_score(test.category, category_pred)),
        "macro_f1": float(f1_score(test.category, category_pred, average="macro")),
        "weighted_f1": float(f1_score(test.category, category_pred, average="weighted")),
        "per_class_report": classification_report(test.category, category_pred, output_dict=True, zero_division=0),
        "class_order": list(category_model.classes_),
        "confusion_matrix": confusion_matrix(test.category, category_pred, labels=category_model.classes_).tolist(),
    }

    metrics = {
        "dataset_rows": int(len(df)), "train_rows": int(len(train)), "test_rows": int(len(test)),
        "unique_scenario_groups": int(df.source_group.nunique()),
        "train_scenario_groups": int(train.source_group.nunique()),
        "test_scenario_groups": int(test.source_group.nunique()),
        "class_count": int(df.category.nunique()),
        "binary_risk_detector": risk_metrics,
        "category_suggestor": category_metrics,
        "limitations": [
            "All examples are synthetic and authored for this prototype.",
            "Grouped holdout keeps paraphrases of a seed scenario in one partition; it is not a real-world evaluation.",
            "The binary score is not a calibrated probability of harm.",
            "The model screens wording only; it cannot diagnose addiction, determine intent, or assess a person's clinical state.",
            "Existing transparent urgent-risk rules and professional judgement must take precedence.",
        ],
    }
    metadata = {
        "version": "patient-risk-substance-v2.1",
        "class_names": list(category_model.classes_),
        "severity_map": SEVERITY_MAP,
        "risk_domain_map": df.groupby("category").risk_domain.first().to_dict(),
        "safe_categories": sorted(SAFE_CATEGORIES),
        "labels": LABELS, "messages": MESSAGES,
        "risk_threshold": 0.60,
        "synthetic_training_rows": int(len(df)),
        "synthetic_only": True, "clinical_validity": "not_established",
        "intended_use": "experimental wording screening only; never diagnose or infer that a person has an addiction",
        "training_metrics": {
            "binary_risk_accuracy": risk_metrics["accuracy"],
            "binary_risk_precision": risk_metrics["precision"],
            "binary_risk_recall": risk_metrics["recall"],
            "category_accuracy": category_metrics["accuracy"],
            "category_macro_f1": category_metrics["macro_f1"],
        },
    }
    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"risk_model": risk_model, "category_model": category_model, "metadata": metadata}, args.model, compress=3)
    Path(args.report).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({
        "dataset_rows": len(df), "train_rows": len(train), "test_rows": len(test),
        "binary_risk_detector": risk_metrics,
        "category_suggestor": {k: category_metrics[k] for k in ("accuracy", "macro_f1", "weighted_f1")},
    }, indent=2))

if __name__ == "__main__":
    main()