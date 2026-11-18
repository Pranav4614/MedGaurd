"""Command line: python -m healthcareguard.cli <command>"""
import argparse, json, sys
import pandas as pd
from .data_gen import generate
from .challenge import load_challenge
from .general_data import generate_proxy, load_labelled_csv
from . import evaluate as ev
from .guard import Guard, ROOT
from .policy import ActionContext


def _build():
    d = ROOT / "data"
    d.mkdir(exist_ok=True)
    generate().to_csv(d / "synthetic_healthcare_instructions.csv", index=False)
    load_challenge().to_csv(d / "challenge_set.csv", index=False)
    generate_proxy().to_csv(d / "general_jailbreak_proxy.csv", index=False)
    print("wrote data/synthetic_healthcare_instructions.csv, challenge_set.csv, general_jailbreak_proxy.csv")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="healthcareguard")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-data", help="write the generated CSVs to data/")
    e = sub.add_parser("evaluate", help="run all experiments, write reports/REPORT.md + results.json, save model")
    e.add_argument("--concardia", help="CSV of a labelled general jailbreak corpus (e.g. Concardia)")
    e.add_argument("--text-col"); e.add_argument("--label-col")
    e.add_argument("--attack-values", nargs="+", help="label values meaning jailbreak/attack, e.g. 1 jailbreak")
    s = sub.add_parser("screen", help="screen one instruction")
    s.add_argument("text")
    s.add_argument("--role", default="unknown"); s.add_argument("--verified", action="store_true")
    s.add_argument("--source", default="user")
    a = ap.parse_args(argv)
    if a.cmd == "build-data":
        _build()
    elif a.cmd == "evaluate":
        gdf, label = None, "synthetic generic-jailbreak PROXY (not Concardia)"
        if a.concardia:
            if not (a.text_col and a.label_col and a.attack_values):
                sys.exit("--concardia needs --text-col, --label-col and --attack-values")
            gdf = load_labelled_csv(pd.read_csv(a.concardia), a.text_col, a.label_col, a.attack_values)
            label = f"user-supplied corpus: {a.concardia}"
        r = ev.run_all(general_df=gdf, general_label=label)
        print((ROOT / "reports" / "REPORT.md").read_text())
    elif a.cmd == "screen":
        g = Guard.load_or_train()
        v = g.screen(a.text, ActionContext(actor_role=a.role, authorisation_verified=a.verified, content_source=a.source))
        print(json.dumps(v.as_dict(), indent=2))


if __name__ == "__main__":
    main()
