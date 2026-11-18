"""Evaluation harness. Run:  python -m healthcareguard.cli evaluate

Experiments
  1. Grouped CV            - template-level split of the synthetic data (inflated: same author/style)
  2. Leave-one-category-out- train without an attack category, test on it (novel attack types)
  3. Challenge set         - hand-written, held out; dev tunes threshold, test reported once
  4. Subtype breakdown     - recall by attack style, FPR by benign style (keyword traps etc.)
  5. Policy layer          - text-only vs text + simulated deployment context
  6. General vs healthcare - does a general jailbreak detector transfer? (proxy corpus or Concardia)
  7. Leakage audit         - nearest-neighbour similarity of challenge items to training data
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer

from .data_gen import generate
from .challenge import load_challenge, split
from .general_data import generate_proxy
from .guard import Guard
from .models import make_pipeline, fit_binary, proba, binary_metrics, bootstrap_ci, choose_threshold
from .policy import ActionContext
from .taxonomy import ALL_CATEGORIES, HEALTHCARE_CATEGORIES

ROOT = Path(__file__).resolve().parents[1]


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else round(float(o), 4)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


# ------------------------------------------------------------------ 1. grouped CV
def grouped_cv(syn, kinds=("v01", "tfidf", "rules", "hybrid"), k=5):
    out = {}
    for kind in kinds:
        p = cross_val_predict(make_pipeline(kind), syn.text, syn.label, groups=syn.group,
                              cv=GroupKFold(n_splits=k), method="predict_proba")[:, 1]
        out[kind] = binary_metrics(syn.label.values, p, 0.5)
    return out


# ------------------------------------------------------------------ 2. leave-one-category-out
def loco(syn, kinds=("tfidf", "rules", "hybrid")):
    out = {}
    for kind in kinds:
        per = {}
        for cat in ALL_CATEGORIES:
            tr, te = syn[syn.category != cat], syn[syn.category == cat]
            m = fit_binary(kind, tr)
            p = proba(m, te.text)
            mm = binary_metrics(te.label.values, p, 0.5)
            per[cat] = {"recall": mm["recall"], "fpr": mm["fpr"], "n_attack": int((te.label == 1).sum()),
                        "n_benign": int((te.label == 0).sum())}
        per["_macro_recall"] = float(np.mean([v["recall"] for k, v in per.items() if not k.startswith("_")]))
        per["_macro_fpr"] = float(np.mean([v["fpr"] for k, v in per.items() if not k.startswith("_")]))
        out[kind] = per
    return out


# ------------------------------------------------------------------ 3/4. challenge set
def _breakdown(df, p, thr, col):
    res = {}
    att = df[df.label == 1]
    for k, g in att.groupby(col):
        res[k] = {"n": int(len(g)), "recall": float((p[g.index] >= thr).mean())}
    return res


def _fpr_breakdown(df, p, thr):
    res = {}
    ben = df[df.label == 0]
    for k, g in ben.groupby("subtype"):
        res[k] = {"n": int(len(g)), "fpr": float((p[g.index] >= thr).mean())}
    return res


def challenge_eval(models: dict, dev, test, ch):
    out = {}
    for name, m in models.items():
        p_dev, p_test, p_all = proba(m, dev.text), proba(m, test.text), proba(m, ch.text)
        thr = choose_threshold(dev.label.values, p_dev)
        entry = {"threshold_tuned_on_dev": thr,
                 "dev_at_tuned": binary_metrics(dev.label.values, p_dev, thr),
                 "test_at_0.5": binary_metrics(test.label.values, p_test, 0.5),
                 "test_at_tuned": binary_metrics(test.label.values, p_test, thr),
                 "test_ci": {"recall": bootstrap_ci(test.label.values, p_test, thr, "recall"),
                             "fpr": bootstrap_ci(test.label.values, p_test, thr, "fpr")},
                 "recall_by_category_all": _breakdown(ch.reset_index(drop=True), p_all, thr, "category"),
                 "recall_by_subtype_all": _breakdown(ch.reset_index(drop=True), p_all, thr, "subtype"),
                 "fpr_by_subtype_all": _fpr_breakdown(ch.reset_index(drop=True), p_all, thr)}
        out[name] = entry
    return out


def errors(model, df, thr, k=12):
    p = proba(model, df.text)
    rows = []
    for i, r in df.iterrows():
        pred = int(p[i] >= thr)
        if pred != r.label:
            rows.append({"text": r.text, "label": int(r.label), "score": round(float(p[i]), 3),
                         "type": "missed attack" if r.label == 1 else "false alarm", "subtype": r.subtype})
    return rows[:k * 3]


# ------------------------------------------------------------------ 5. policy layer
def policy_eval(guard: Guard, ch: pd.DataFrame):
    res = {}
    for scope, df in (("all", ch), ("test", ch[ch.split == "test"])):
        df = df.reset_index(drop=True)
        y = df.label.values
        text_only = np.array([guard.score([t])[0] >= guard.threshold for t in df.text])
        with_ctx, no_ctx = [], []
        for _, r in df.iterrows():
            ctx = ActionContext(actor_role=r.ctx_role, authorisation_verified=bool(r.ctx_verified_auth),
                                content_source=r.ctx_source)
            with_ctx.append(guard.screen(r.text, ctx).decision in ("REVIEW", "BLOCK"))
            no_ctx.append(guard.screen(r.text, ActionContext()).decision in ("REVIEW", "BLOCK"))

        def rates(pred):
            pred = np.asarray(pred)
            return {"attack_recall": float(pred[y == 1].mean()), "benign_intervention_rate": float(pred[y == 0].mean())}
        res[scope] = {"n": int(len(df)), "text_classifier_only": rates(text_only),
                      "policy_no_context": rates(no_ctx), "policy_with_simulated_context": rates(with_ctx)}
    return res


# ------------------------------------------------------------------ 7. leakage audit
def leakage_audit(syn, ch):
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(pd.concat([syn.text, ch.text]))
    sim = (vec.transform(ch.text) @ vec.transform(syn.text).T).toarray().max(axis=1)
    return {"exact_overlap": int(len(set(syn.text) & set(ch.text))),
            "challenge_items": int(len(ch)),
            "max_cosine_mean": float(sim.mean()),
            "items_with_cosine_gt_0.6": int((sim > 0.6).sum()),
            "items_with_cosine_gt_0.8": int((sim > 0.8).sum())}


# ------------------------------------------------------------------ 6. general vs healthcare
def general_vs_healthcare(general_df: pd.DataFrame, hybrid, ch, dev, test, label="proxy"):
    gen = fit_binary("tfidf", general_df)
    res = challenge_eval({"general": gen, "healthcare_hybrid": hybrid}, dev, test, ch)
    res["general_training_rows"] = int(len(general_df))
    res["general_source"] = label
    return res


# ------------------------------------------------------------------ orchestration
def run_all(general_df=None, general_label="synthetic generic-jailbreak PROXY (not Concardia)", write=True, out_dir=None):
    out_dir = Path(out_dir or ROOT / "reports")
    syn = generate()
    ch = load_challenge()
    dev, test = split(ch)
    v01 = pd.read_csv(ROOT / "data" / "v0_1_seed.csv")
    general_df = generate_proxy() if general_df is None else general_df

    models = {
        "v0.1 prototype (30 rows)": fit_binary("v01", v01),
        "general jailbreak detector": fit_binary("tfidf", general_df),
        "tfidf (healthcare data)": fit_binary("tfidf", syn),
        "rules only": fit_binary("rules", syn),
        "hybrid (healthcare data)": fit_binary("hybrid", syn),
    }
    results = {
        "meta": {"synthetic_rows": int(len(syn)), "templates": int(syn.group.nunique()),
                 "challenge_dev": int(len(dev)), "challenge_test": int(len(test)),
                 "general_source": general_label, "general_rows": int(len(general_df))},
        "grouped_cv": grouped_cv(syn),
        "loco": loco(syn),
        "challenge": challenge_eval(models, dev, test, ch),
        "leakage": leakage_audit(syn, ch),
    }
    guard = Guard.train(syn, tune_on=dev)
    results["policy"] = policy_eval(guard, ch)
    results["errors_hybrid_test"] = errors(models["hybrid (healthcare data)"], test,
                                            results["challenge"]["hybrid (healthcare data)"]["threshold_tuned_on_dev"])
    # category accuracy on challenge attacks (hybrid category model)
    att = ch[ch.label == 1].reset_index(drop=True)
    pred = guard.category_model.predict(att.text)
    results["category_accuracy_challenge_attacks"] = float((pred == att.category.values).mean())
    results = _clean(results)
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results.json").write_text(json.dumps(results, indent=2))
        (out_dir / "REPORT.md").write_text(render_report(results))
        guard.save()
    return results


# ------------------------------------------------------------------ report
def _pct(x):
    return "n/a" if x is None else f"{100 * x:.0f}%"


def _ci(c):
    return f"{_pct(c[0])}-{_pct(c[1])}"


def md_table(headers, rows):
    s = "| " + " | ".join(headers) + " |\n|" + "---|" * len(headers) + "\n"
    for r in rows:
        s += "| " + " | ".join(str(x) for x in r) + " |\n"
    return s


def render_report(r) -> str:
    m = r["meta"]
    s = ["# HealthcareGuard v0.2 - evaluation report\n",
         "> Auto-generated by `python -m healthcareguard.cli evaluate`. Read the caveats before the numbers.\n",
         "## Caveats (read first)\n",
         f"- Training data: **{m['synthetic_rows']} synthetic rows from {m['templates']} templates**, written by the same author as the detector.",
         f"- Challenge set: **{m['challenge_dev']+m['challenge_test']} hand-written examples** ({m['challenge_dev']} dev / {m['challenge_test']} test), also written by the detector's author, who drafted the cue lexicon with that style of attack in mind. It is a stress test, **not independent validation**. Dev was used only to choose the decision threshold; **no cue or model edits were made after the first full run**, so test numbers are from that single run.",
         f"- General-detector source: **{m['general_source']}** ({m['general_rows']} rows). If this says PROXY, it is not Concardia and the transfer result below is only indicative.",
         "- With 44 test items, a single example moves recall/FPR by ~5 points. Look at the confidence intervals.",
         "- Nothing here shows the system is clinically safe. No clinician reviewed the data or labels.\n",
         "## 1. Template-grouped cross-validation (synthetic data, unseen templates per fold)\n"]
    s.append(md_table(["model", "precision", "recall", "FPR", "F1", "AUROC"],
                      [[k, _pct(v["precision"]), _pct(v["recall"]), _pct(v["fpr"]), f"{v['f1']:.3f}", f"{v['auroc']:.3f}"]
                       for k, v in r["grouped_cv"].items()]))
    s.append("Every fold is scored on templates absent from its training folds. All templates share an author and style, and the synthetic benign examples include deliberate keyword traps, so FPR here is a stress number, not a deployment estimate. Compare with section 3: if the hand-written set is *no harder* than template-held-out CV, it is not a real out-of-distribution test.\n")
    s.append("## 2. Leave-one-attack-category-out (novel attack types)\n")
    cats = [c for c in r["loco"]["hybrid"] if not c.startswith("_")]
    rows = []
    for c in cats:
        rows.append([c] + [f"{_pct(r['loco'][k][c]['recall'])} / {_pct(r['loco'][k][c]['fpr'])}" for k in ("tfidf", "rules", "hybrid")])
    rows.append(["**macro**"] + [f"{_pct(r['loco'][k]['_macro_recall'])} / {_pct(r['loco'][k]['_macro_fpr'])}" for k in ("tfidf", "rules", "hybrid")])
    s.append(md_table(["held-out category", "tfidf recall / FPR", "rules recall / FPR", "hybrid recall / FPR"], rows))
    s.append("## 3. Hand-written challenge set (held out)\n")
    rows = []
    for k, v in r["challenge"].items():
        t = v["test_at_tuned"]
        rows.append([k, f"{v['threshold_tuned_on_dev']:.2f}", _pct(t["recall"]), _ci(v["test_ci"]["recall"]),
                     _pct(t["fpr"]), _ci(v["test_ci"]["fpr"]), f"{t['f1']:.3f}", f"{t['auroc']:.3f}",
                     f"{_pct(v['test_at_0.5']['recall'])} / {_pct(v['test_at_0.5']['fpr'])}"])
    s.append(md_table(["model", "thr (tuned on dev)", "test recall", "95% CI", "test FPR", "95% CI", "F1", "AUROC", "at thr 0.5: recall / FPR"], rows))
    flat = [k for k, v in r["challenge"].items() if v["test_at_tuned"]["fpr"] is not None and v["test_at_tuned"]["fpr"] >= 0.9]
    if flat:
        s.append("Note: " + ", ".join(f"**{k}**" for k in flat) + " is flagging (almost) everything at its dev-tuned threshold, i.e. it has no usable signal on healthcare text; use the 'at thr 0.5' column (its native operating point) and AUROC for that row.\n")
    s.append("## 4. Where does each model fail? (all 89 challenge items; thresholds tuned on dev)\n")
    names = list(r["challenge"].keys())
    subs = sorted(r["challenge"][names[0]]["recall_by_subtype_all"])
    s.append("**Attack recall by attack style**\n")
    s.append(md_table(["attack style (n)"] + names,
                      [[f"{x} ({r['challenge'][names[0]]['recall_by_subtype_all'][x]['n']})"] +
                       [_pct(r["challenge"][n]["recall_by_subtype_all"][x]["recall"]) for n in names] for x in subs]))
    cats = sorted(r["challenge"][names[0]]["recall_by_category_all"])
    s.append("**Attack recall by category**\n")
    s.append(md_table(["category (n)"] + names,
                      [[f"{x} ({r['challenge'][names[0]]['recall_by_category_all'][x]['n']})"] +
                       [_pct(r["challenge"][n]["recall_by_category_all"][x]["recall"]) for n in names] for x in cats]))
    fs = sorted(r["challenge"][names[0]]["fpr_by_subtype_all"])
    s.append("**False-alarm rate on benign items by style**\n")
    s.append(md_table(["benign style (n)"] + names,
                      [[f"{x} ({r['challenge'][names[0]]['fpr_by_subtype_all'][x]['n']})"] +
                       [_pct(r["challenge"][n]["fpr_by_subtype_all"][x]["fpr"]) for n in names] for x in fs]))
    s.append("## 5. Policy layer with simulated deployment context\n")
    s.append("The policy layer is default-deny for *state-changing actions on clinical objects* when no verified authorisation is supplied, which is why attack interception is high and why benign interventions rise when context is missing. Contexts below are *simulated* from the challenge labels (clinician-authorised rows get a verified clinician context; embedded-injection rows arrive as a retrieved document). This shows what the layer can do **if** the host system supplies trustworthy context; it does not show that real systems will.\n")
    rows = []
    for scope in ("all", "test"):
        p = r["policy"][scope]
        for k, label in (("text_classifier_only", "text classifier only"), ("policy_no_context", "policy, no context supplied"),
                         ("policy_with_simulated_context", "policy + simulated context")):
            rows.append([f"{scope} (n={p['n']})", label, _pct(p[k]["attack_recall"]), _pct(p[k]["benign_intervention_rate"])])
    s.append(md_table(["scope", "configuration", "attacks intercepted", "benign requests held/blocked"], rows))
    s.append(f"Category label accuracy on challenge attacks (9-way, hybrid): **{_pct(r['category_accuracy_challenge_attacks'])}**. Categories overlap (e.g. 'chart it as verified' is both dose and record manipulation), so treat the category as a hint.\n")
    s.append("## 6. Leakage audit\n")
    L = r["leakage"]
    s.append(f"Exact text overlap train/challenge: **{L['exact_overlap']}**. Challenge items with TF-IDF cosine > 0.6 to some training row: **{L['items_with_cosine_gt_0.6']}/{L['challenge_items']}**; > 0.8: **{L['items_with_cosine_gt_0.8']}**. Mean max cosine: {L['max_cosine_mean']:.2f}.\n")
    s.append("## 7. Remaining errors of the hybrid model on the test split\n")
    s.append(md_table(["type", "score", "style", "text"], [[e["type"], e["score"], e["subtype"], e["text"]] for e in r["errors_hybrid_test"]]) if r["errors_hybrid_test"] else "None.\n")
    return "\n".join(s)
