"""Checks whether the adverse action reasons are stable, faithful and useful.

Consistency: do denied applicants who look alike get the same top reason?
Faithfulness: if the top reason feature is improved, does the decision flip?
Coverage: what share of denials cite at least one actionable reason first?
"""

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from explain import THRESHOLD, load_reason_codes, explain_denials
from train import FEATURES

# direction that improves the score for each feature
IMPROVE = {
    "income": 1,
    "dti": -1,
    "utilisation": -1,
    "months_since_delinquency": 1,
    "inquiries": -1,
    "loan_amount": -1,
    "employment_years": 1,
}


def consistency(denied, reasons_df, k=5):
    scaler = StandardScaler()
    Z = scaler.fit_transform(denied[FEATURES])
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    _, idx = nn.kneighbors(Z)
    top = reasons_df["reason_1_feature"].to_numpy()
    agree = []
    for i in range(len(Z)):
        neighbours = idx[i][1:]
        agree.append(np.mean(top[neighbours] == top[i]))
    return float(np.mean(agree))


def faithfulness(model, denied, reasons_df, full_df):
    flipped = 0
    tested = 0
    for i, (_, r) in enumerate(reasons_df.iterrows()):
        feat = r["reason_1_feature"]
        if not feat:
            continue
        row = denied.iloc[[i]].copy()
        # move the top reason feature to the 75th or 25th percentile of approved applicants
        approved = full_df[full_df["repaid"] == 1][feat]
        target = approved.quantile(0.75) if IMPROVE[feat] > 0 else approved.quantile(0.25)
        row[feat] = target
        new_score = model.predict_proba(row[FEATURES])[0, 1]
        tested += 1
        if new_score >= THRESHOLD:
            flipped += 1
    return flipped / tested if tested else 0.0


def coverage(reasons_df, codes):
    first_actionable = reasons_df["reason_1_feature"].map(lambda f: codes[f]["actionable"] if f else False)
    any_actionable = np.zeros(len(reasons_df), dtype=bool)
    for j in range(1, 5):
        col = reasons_df[f"reason_{j}_feature"]
        any_actionable |= col.map(lambda f: codes[f]["actionable"] if f else False).to_numpy()
    return float(first_actionable.mean()), float(any_actionable.mean())


def write_report(results, path):
    lines = ["# Adverse action reason quality report", ""]
    lines.append(f"Denials explained: {results['n_denied']}")
    lines.append("")
    lines.append(f"Consistency (share of 5 nearest neighbours sharing the top reason): {results['consistency']:.3f}")
    lines.append("")
    lines.append(f"Faithfulness (share of denials that flip to approval when the top reason feature is moved to an approved applicant level): {results['faithfulness']:.3f}")
    lines.append("")
    lines.append(f"Coverage, first reason actionable: {results['coverage_first']:.3f}")
    lines.append("")
    lines.append(f"Coverage, any of the four reasons actionable: {results['coverage_any']:.3f}")
    lines.append("")
    lines.append("Reading the numbers. Consistency near 1 means similar applicants get similar explanations, which is what a reviewer expects. Faithfulness measures whether the stated reason actually drives the decision; a low number would mean the notices point at features the model does not really depend on. Coverage asks whether the applicant can do anything about the reasons given, which matters because a notice built from reasons the applicant cannot change (such as length of employment) is accurate but not useful.")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def run(model, df, codes):
    X = df[FEATURES]
    reasons_df, _, denied = explain_denials(model, X, codes)
    cons = consistency(denied, reasons_df)
    faith = faithfulness(model, denied, reasons_df, df)
    cov_first, cov_any = coverage(reasons_df, codes)
    return {
        "n_denied": len(reasons_df),
        "consistency": cons,
        "faithfulness": faith,
        "coverage_first": cov_first,
        "coverage_any": cov_any,
    }


if __name__ == "__main__":
    codes = load_reason_codes()
    df = pd.read_csv("data/applicants.csv")
    with open("outputs/model.pkl", "rb") as f:
        model = pickle.load(f)
    results = run(model, df, codes)
    write_report(results, "outputs/quality_report.md")
    with open("outputs/quality_results.json", "w") as f:
        json.dump(results, f, indent=2)
    for k, v in results.items():
        print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")
