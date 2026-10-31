import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import make_applicants, train, FEATURES
from explain import load_reason_codes, explain_denials, top_reasons, THRESHOLD
from quality import coverage, consistency

CODES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "reason_codes.json")


def small_setup():
    df = make_applicants(n=1500, seed=3)
    model, _, _ = train(df)
    codes = load_reason_codes(CODES_PATH)
    reasons_df, vals, denied = explain_denials(model, df[FEATURES], codes)
    return df, model, codes, reasons_df, vals, denied


def test_top_reasons_only_negative():
    row = np.array([0.3, -0.5, -0.1, 0.2, -0.05, 0.0, 0.1])
    reasons = top_reasons(row, k=4)
    assert reasons == ["dti", "utilisation", "inquiries"]


def test_every_feature_has_a_code():
    codes = load_reason_codes(CODES_PATH)
    assert set(codes) == set(FEATURES)


def test_denials_all_below_threshold():
    df, model, codes, reasons_df, vals, denied = small_setup()
    assert (reasons_df["score"] < THRESHOLD).all()
    assert len(reasons_df) == len(denied)


def test_first_reason_present_for_all():
    df, model, codes, reasons_df, vals, denied = small_setup()
    assert (reasons_df["reason_1_feature"] != "").all()


def test_quality_metrics_in_range():
    df, model, codes, reasons_df, vals, denied = small_setup()
    c = consistency(denied, reasons_df)
    first, any_ = coverage(reasons_df, codes)
    assert 0 <= c <= 1
    assert 0 <= first <= any_ <= 1
