# Adverse action explainer

This repo trains a credit decision model on synthetic applications and then generates the explanation a declined applicant is entitled to receive. For each denial it uses SHAP values from a gradient boosting model to find the four features that pushed the score down the most, maps them to plain language reason codes in the style of ECOA Regulation B adverse action notices, and writes sample notices. It then tests whether those explanations are any good along three lines: consistency (do similar applicants get the same reasons), faithfulness (does changing the stated reason actually change the decision) and coverage (can the applicant do anything about the reasons given).

The repo grows out of "The Governance Lag: Why Slow AI Adoption Is the Larger Risk" (Rehman, 2026). One of the arguments there is that explainability obligations are often treated as a reason to delay deploying models in credit, when the harder and more useful question is whether the explanations a model gives are faithful and stable. A model that produces a reason code is easy. A model whose reason codes survive scrutiny is the actual governance requirement, and that is something you can measure.

In financial services this is a live obligation. Regulation B requires creditors to state the principal reasons for an adverse action, and the CFPB has said that using a complex model is not an excuse for vague reasons. The EU AI Act treats creditworthiness scoring as high risk and expects meaningful explanation. Model risk teams therefore need a way to validate explanations, not only accuracy. The quality checks in this repo are a first draft of that validation.

## How to run

```
pip install -r requirements.txt
python train.py
python explain.py
python quality.py
pytest
```

train.py builds data/applicants.csv (8,000 synthetic applicants, fixed seed) and saves the model. explain.py writes outputs/reasons.csv (one row per denied applicant with four reason codes), outputs/adverse_action_notices.md (five sample notices) and outputs/reason_frequency.png. quality.py writes outputs/quality_report.md and outputs/quality_results.json.

## Reading the outputs

The notices list the principal reasons in the order the model weighted them. The reason frequency chart shows which codes dominate across all denials; in the shipped run debt to income, delinquency history and utilisation account for most first reasons. The quality report has three numbers. Consistency is the share of an applicant's five nearest neighbours (in standardised feature space) who received the same top reason, about 0.69 in the shipped run. Faithfulness is the share of denials that flip to approval when the top reason feature is moved to the level of a typical approved applicant, about 0.85, which means the stated reason is usually a real driver. Coverage is the share of denials whose first reason is one the applicant can act on (0.77) and whose four reasons include at least one actionable item (1.0). Delinquency history and length of employment are marked as not actionable in data/reason_codes.json because the applicant cannot change them in the short term.

## Data and sources

All applicants are synthetic. Feature names and the reason code wording follow the sample reason list in the Regulation B model forms (Appendix C). SHAP values come from the shap package's TreeExplainer. No real credit data is included.

## Limitations

SHAP attributions are relative to the training distribution, so the same applicant could receive different reasons from a model trained on a different population. The faithfulness test moves one feature at a time, which ignores interactions. Reason codes are assigned by feature, while real notices sometimes combine several features into one reason. The approval threshold of 0.5 is arbitrary.

Author: Saif Rehman, MMAI, Smith School of Business, Queen's University.
