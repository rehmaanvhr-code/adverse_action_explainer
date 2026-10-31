"""Produces adverse action reasons for denied applicants using SHAP values.

For every denied applicant the four features that pushed the score down
the most are mapped to reason codes in data/reason_codes.json. The result is
a table of reasons per applicant and a handful of sample notices.
"""

import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from train import FEATURES

THRESHOLD = 0.5
TOP_K = 4


def load_reason_codes(path="data/reason_codes.json"):
    with open(path) as f:
        return json.load(f)


def shap_values(model, X):
    explainer = shap.TreeExplainer(model)
    vals = explainer.shap_values(X)
    if isinstance(vals, list):
        vals = vals[1]
    return np.asarray(vals)


def top_reasons(shap_row, k=TOP_K):
    # most negative contributions are the reasons for denial
    order = np.argsort(shap_row)
    reasons = []
    for idx in order[:k]:
        if shap_row[idx] < 0:
            reasons.append(FEATURES[idx])
    return reasons


def explain_denials(model, X, codes):
    scores = model.predict_proba(X)[:, 1]
    denied = X[scores < THRESHOLD]
    vals = shap_values(model, denied)
    rows = []
    for i, (idx, _) in enumerate(denied.iterrows()):
        reasons = top_reasons(vals[i])
        row = {"applicant_id": idx, "score": scores[X.index.get_loc(idx)]}
        for j in range(TOP_K):
            feat = reasons[j] if j < len(reasons) else ""
            row[f"reason_{j + 1}"] = codes[feat]["code"] if feat else ""
            row[f"reason_{j + 1}_feature"] = feat
        rows.append(row)
    return pd.DataFrame(rows), vals, denied


def write_notices(reasons_df, applicants, codes, path, n=5):
    lines = ["# Sample adverse action notices", ""]
    lines.append("These notices follow the structure of an ECOA Regulation B adverse action notice. The principal reasons are listed in order of importance as the model saw them.")
    lines.append("")
    for _, r in reasons_df.head(n).iterrows():
        a = applicants.loc[r["applicant_id"]]
        lines.append(f"## Applicant {r['applicant_id']}")
        lines.append("")
        lines.append("Your application for credit has been declined. The principal reasons for this decision are:")
        lines.append("")
        for j in range(1, TOP_K + 1):
            feat = r[f"reason_{j}_feature"]
            if feat:
                lines.append(f"{j}. {codes[feat]['text']} (code {codes[feat]['code']})")
        lines.append("")
        lines.append(f"Model score {r['score']:.2f} against a threshold of {THRESHOLD}. Key inputs: income {a['income']:.0f}, debt to income {a['dti']:.2f}, utilisation {a['utilisation']:.2f}, months since delinquency {a['months_since_delinquency']:.0f}, inquiries {a['inquiries']:.0f}.")
        lines.append("")
        lines.append("You have the right to a statement of specific reasons and to request a copy of the information used in this decision.")
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def plot_reason_frequency(reasons_df, codes, path):
    counts = {}
    for j in range(1, TOP_K + 1):
        for feat in reasons_df[f"reason_{j}_feature"]:
            if feat:
                counts[feat] = counts.get(feat, 0) + 1
    labels = [f"{codes[f]['code']} {f}" for f in counts]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(labels, list(counts.values()))
    ax.set_xlabel("Number of denied applicants citing this reason")
    ax.set_title("Adverse action reason frequency (top four reasons per denial)")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    codes = load_reason_codes()
    df = pd.read_csv("data/applicants.csv")
    with open("outputs/model.pkl", "rb") as f:
        model = pickle.load(f)
    X = df[FEATURES]
    reasons_df, vals, denied = explain_denials(model, X, codes)
    reasons_df.to_csv("outputs/reasons.csv", index=False)
    write_notices(reasons_df, df, codes, "outputs/adverse_action_notices.md")
    plot_reason_frequency(reasons_df, codes, "outputs/reason_frequency.png")
    print(f"{len(reasons_df)} denials explained out of {len(df)} applicants")
    print(reasons_df["reason_1_feature"].value_counts())
