"""Synthetic credit applications and a gradient boosting decision model."""

import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

SEED = 11
FEATURES = [
    "income",
    "dti",
    "utilisation",
    "months_since_delinquency",
    "inquiries",
    "loan_amount",
    "employment_years",
]


def make_applicants(n=8000, seed=SEED):
    rng = np.random.default_rng(seed)
    income = np.clip(rng.lognormal(np.log(58000), 0.45, size=n), 12000, 400000)
    dti = np.clip(rng.normal(0.34, 0.13, size=n), 0.02, 0.98)
    utilisation = np.clip(rng.beta(2, 3.5, size=n), 0, 1)
    # 60 is used for "never delinquent" so higher is better
    delinquent = rng.random(n) < 0.3
    months_since_delinquency = np.where(delinquent, rng.integers(1, 48, size=n), 60)
    inquiries = rng.poisson(1.6, size=n)
    loan_amount = np.clip(rng.normal(20000, 10000, size=n), 1000, 90000)
    employment_years = np.clip(rng.normal(6, 4.5, size=n), 0, 40)

    logit = (
        0.4
        + 1.1 * np.log(income / 58000)
        - 4.5 * (dti - 0.34)
        - 2.8 * (utilisation - 0.36)
        + 0.025 * (months_since_delinquency - 42)
        - 0.3 * (inquiries - 1.6)
        - 0.00003 * (loan_amount - 20000)
        + 0.05 * (employment_years - 6)
    )
    p = 1 / (1 + np.exp(-logit))
    repaid = rng.binomial(1, p)

    return pd.DataFrame({
        "income": income.round(0),
        "dti": dti.round(3),
        "utilisation": utilisation.round(3),
        "months_since_delinquency": months_since_delinquency,
        "inquiries": inquiries,
        "loan_amount": loan_amount.round(0),
        "employment_years": employment_years.round(1),
        "repaid": repaid,
    })


def train(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df["repaid"], test_size=0.25, random_state=SEED
    )
    model = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=SEED)
    model.fit(X_train, y_train)
    return model, X_test, y_test


if __name__ == "__main__":
    df = make_applicants()
    df.to_csv("data/applicants.csv", index=False)
    model, X_test, y_test = train(df)
    print(f"test accuracy {model.score(X_test, y_test):.3f}, approval base rate {df['repaid'].mean():.3f}")
    with open("outputs/model.pkl", "wb") as f:
        pickle.dump(model, f)
