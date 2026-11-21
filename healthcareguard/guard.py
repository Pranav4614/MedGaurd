"""High-level API: Guard.screen(text, context) -> Verdict."""
from __future__ import annotations
import joblib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional
import pandas as pd
from .lexicon import extract_features, find_evidence, FEATURE_NAMES
from .models import fit_binary, fit_category, proba, choose_threshold
from .policy import ActionContext, apply_policy
from .taxonomy import GENERAL, HIGH_SEVERITY

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "guard.joblib"
if not MODEL_PATH.exists():
    MODEL_PATH = ROOT / "guard.joblib"

REASON_NAMES = {
    "conceal": "concealment_cue", "bypass": "safety_control_bypass_cue", "falsify": "falsification_cue",
    "false_reassurance": "false_reassurance_cue", "unauthorised": "missing_authorisation_cue",
    "authority_claim": "unverified_authority_claim", "embedded": "embedded_instruction_pattern",
    "exfil": "data_exfiltration_pattern",
}


@dataclass
class Verdict:
    text: str
    decision: str
    flagged: bool
    score: float
    threshold: float
    category: str
    category_confidence: float
    risk: str
    reasons: List[str]
    evidence: List[dict]
    policy_notes: List[str]
    rule_contributions: List[dict]
    explanation: str

    def as_dict(self):
        return asdict(self)


class Guard:
    def __init__(self, detector, category_model, threshold: float, meta: Optional[dict] = None):
        self.detector, self.category_model, self.threshold = detector, category_model, float(threshold)
        self.meta = meta or {}

    # ------------------------------------------------------------------ build / persist
    @classmethod
    def train(cls, syn: pd.DataFrame, threshold: Optional[float] = None, tune_on: Optional[pd.DataFrame] = None):
        det = fit_binary("hybrid", syn)
        cat = fit_category(syn)
        if threshold is None:
            if tune_on is not None and len(tune_on):
                threshold = choose_threshold(tune_on.label.values, proba(det, tune_on.text))
            else:
                threshold = 0.5
        return cls(det, cat, threshold, {"trained_rows": int(len(syn)), "threshold_tuned_on": "challenge-dev" if tune_on is not None else "default"})

    def save(self, path=MODEL_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"detector": self.detector, "category_model": self.category_model,
                     "threshold": self.threshold, "meta": self.meta}, path)

    @classmethod
    def load(cls, path=MODEL_PATH):
        d = joblib.load(path)
        return cls(d["detector"], d["category_model"], d["threshold"], d.get("meta"))

    @classmethod
    def load_or_train(cls, path=MODEL_PATH):
        from .data_gen import generate
        from .challenge import load_challenge, split
        if Path(path).exists():
            try:
                return cls.load(path)
            except Exception:
                pass
        dev, _ = split(load_challenge())
        g = cls.train(generate(), tune_on=dev)
        try:
            g.save(path)
        except Exception:
            pass
        return g

    # ------------------------------------------------------------------ inference
    def _rule_contributions(self, text, k=6):
        clf = self.detector.named_steps["clf"]
        n = len(FEATURE_NAMES)
        w = clf.coef_[0][-n:]
        f = extract_features(text)
        c = [(name, float(w[i] * f[name])) for i, name in enumerate(FEATURE_NAMES) if f[name]]
        c.sort(key=lambda x: -abs(x[1]))
        return [{"feature": a, "contribution": round(b, 3)} for a, b in c[:k]]

    def score(self, texts):
        return proba(self.detector, texts)

    def screen(self, text, context: Optional[ActionContext] = None) -> Verdict:
        ctx = context or ActionContext()
        text = "" if text is None else str(text).strip()
        if not text:
            return Verdict("", "NO_INPUT", False, 0.0, self.threshold, GENERAL, 0.0, "Low", [], [], [], [],
                           "Enter text to screen.")
        s = float(self.score([text])[0])
        cp = self.category_model.predict_proba([text])[0]
        classes = list(self.category_model.classes_)
        cat, conf = classes[int(cp.argmax())], float(cp.max())
        feats = extract_features(text)
        flagged = s >= self.threshold
        out = apply_policy(score=s, threshold=self.threshold, category=cat, feats=feats, ctx=ctx)
        reasons = [REASON_NAMES[k] for k in REASON_NAMES if feats.get(k)]
        if flagged:
            reasons.insert(0, "ml_score_above_threshold")
        if feats["state_x_target"] and not feats["info_request"]:
            reasons.append("state_changing_action_on_clinical_target")
        if ctx.content_source != "user":
            reasons.append("untrusted_content_source")
        if out.decision in ("BLOCK",) or (flagged and cat in HIGH_SEVERITY):
            risk = "High"
        elif flagged or out.decision == "REVIEW":
            risk = "Medium"
        else:
            risk = "Low"
        ev = [{"cue": c, "text": t, "kind": k} for c, t, k in find_evidence(text)]
        try:
            rc = self._rule_contributions(text)
        except Exception:
            rc = []
        expl = {
            "BLOCK": "Blocked: pattern consistent with an attempt to subvert a clinical safeguard.",
            "REVIEW": "Hold for human review before any action is taken.",
            "ALLOW_WITH_AUDIT": "No adversarial pattern that overrides the verified context; proceed and log.",
            "ALLOW": "No strong adversarial pattern found. This does not prove the text is safe or clinically correct.",
        }[out.decision]
        return Verdict(text, out.decision, flagged, s, self.threshold, cat if flagged or out.decision in ("BLOCK", "REVIEW") else GENERAL,
                       conf, risk, reasons, ev, out.notes, rc, expl)

    def screen_df(self, df: pd.DataFrame, text_col: str, role_col=None, verified_col=None, source_col=None) -> pd.DataFrame:
        rows = []
        for _, r in df.iterrows():
            ctx = ActionContext(
                actor_role=str(r[role_col]) if role_col else "unknown",
                authorisation_verified=bool(r[verified_col]) if verified_col else False,
                content_source=str(r[source_col]) if source_col else "user")
            v = self.screen(r[text_col], ctx)
            rows.append({"decision": v.decision, "score": round(v.score, 3), "category": v.category,
                         "risk": v.risk, "reasons": "; ".join(v.reasons), "policy_notes": "; ".join(v.policy_notes)})
        return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
