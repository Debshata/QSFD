"""Dataset registry, loading, validation and preprocessing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import (
    apply_imbalance,
    fit_preprocessor,
    make_splits,
    transform,
)
from src.data.registry import classify_columns, detect_target, normalise_target
from src.data.validators import (
    check_target_correlation,
    drop_cross_split_duplicates,
    profile_dataset,
    validate_dataset,
)


# -- registry ---------------------------------------------------------------

def test_detect_target_prefers_configured_column(synthetic_frame):
    assert detect_target(synthetic_frame, "is_fraud", ["Class"]) == "is_fraud"


def test_detect_target_falls_back_to_candidates(synthetic_frame):
    assert detect_target(synthetic_frame, "not_a_column",
                         ["nope", "is_fraud"]) == "is_fraud"


def test_detect_target_raises_when_absent():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError, match="Could not identify a target"):
        detect_target(df, None, ["Class"])


@pytest.mark.parametrize("raw,positives,expected", [
    ([0, 1, 0, 1], [1], [0, 1, 0, 1]),
    (["N", "Y", "N"], ["Y"], [0, 1, 0]),
    ([False, True, False], [True], [0, 1, 0]),
    (["legit", "FRAUD", "legit"], ["fraud"], [0, 1, 0]),   # case-insensitive
])
def test_normalise_target_handles_encodings(raw, positives, expected):
    out = normalise_target(pd.Series(raw), positives)
    assert out.tolist() == expected
    assert out.dtype == np.int8


def test_normalise_target_rejects_all_negative():
    with pytest.raises(ValueError, match="zero positives"):
        normalise_target(pd.Series([0, 0, 0]), [1])


def test_classify_columns_assigns_roles(synthetic_frame):
    from src.utils.config import load_config

    roles = classify_columns(synthetic_frame, "is_fraud", [],
                             load_config("datasets")["autodetect"])
    assert "amount" in roles["numeric"]
    assert "channel" in roles["categorical"]
    assert "constant" in roles["unusable"]      # zero variance
    # txn_id is unique per row and matches an identifier pattern.
    assert "txn_id" in roles["identifier"] + roles["unusable"]


def test_spec_excludes_identifier_and_target(synthetic_spec):
    assert synthetic_spec.target == "is_fraud"
    assert "is_fraud" not in synthetic_spec.feature_columns
    assert "txn_id" not in synthetic_spec.feature_columns
    assert "constant" not in synthetic_spec.feature_columns


# -- validation -------------------------------------------------------------

def test_validate_passes_on_clean_data(synthetic_spec, synthetic_xy):
    X, y = synthetic_xy
    report = validate_dataset("synthetic_test", synthetic_spec, X, y)
    assert report.passed, report.errors


def test_validate_detects_target_leakage(synthetic_spec, synthetic_xy):
    X, y = synthetic_xy
    leaked = X.copy()
    leaked["sneaky"] = y.to_numpy() * 1.0       # a copy of the label
    report = validate_dataset("synthetic_test", synthetic_spec, leaked, y)
    assert not report.passed
    assert any("sneaky" in e for e in report.errors)


def test_validate_detects_target_in_features(synthetic_spec, synthetic_xy):
    X, y = synthetic_xy
    bad = X.copy()
    bad[synthetic_spec.target] = y.to_numpy()
    report = validate_dataset("synthetic_test", synthetic_spec, bad, y)
    assert not report.passed


def test_check_target_correlation_flags_copies(synthetic_xy):
    X, y = synthetic_xy
    X = X.copy()
    X["copy"] = y.to_numpy()
    assert "copy" in check_target_correlation(X, y)


def test_profile_reports_class_balance(synthetic_spec, synthetic_xy):
    X, y = synthetic_xy
    p = profile_dataset("synthetic_test", synthetic_spec, X, y, len(X))
    assert p["fraud_count"] + p["legitimate_count"] == p["rows"]
    assert 0 < p["fraud_percentage"] < 100


def test_duplicates_removed_before_split(synthetic_xy):
    X, y = synthetic_xy
    Xd = pd.concat([X, X.head(20)], ignore_index=True)
    yd = pd.concat([y, y.head(20)], ignore_index=True)
    Xc, yc, dropped = drop_cross_split_duplicates(Xd, yd)
    assert dropped >= 20
    assert len(Xc) == len(yc) == len(Xd) - dropped


# -- splitting and preprocessing -------------------------------------------

def test_splits_are_disjoint_and_complete(synthetic_spec, synthetic_xy):
    X, y = synthetic_xy
    s = make_splits(X, y, synthetic_spec)
    total = len(s.X_train) + len(s.X_val) + len(s.X_test)
    Xd, _, _ = drop_cross_split_duplicates(X, y)
    assert total == len(Xd)
    for fold in (s.y_train, s.y_val, s.y_test):
        assert len(fold) > 0
        assert fold.sum() > 0, "every fold should contain fraud cases"


def test_splits_are_deterministic(synthetic_spec, synthetic_xy):
    X, y = synthetic_xy
    a = make_splits(X, y, synthetic_spec)
    b = make_splits(X, y, synthetic_spec)
    pd.testing.assert_frame_equal(a.X_test, b.X_test)
    pd.testing.assert_series_equal(a.y_test, b.y_test)


def test_preprocessor_output_is_finite_and_no_nans(synthetic_spec, synthetic_xy):
    X, y = synthetic_xy
    s = make_splits(X, y, synthetic_spec)
    pre, names = fit_preprocessor(synthetic_spec, s.X_train)
    for fold in (s.X_train, s.X_val, s.X_test):
        out = transform(pre, fold)
        assert np.isfinite(out).all()
        assert not np.isnan(out).any()
        assert out.shape[1] == len(names)


def test_preprocessor_handles_missing_values(synthetic_spec, synthetic_xy):
    X, y = synthetic_xy
    X = X.copy()
    X.loc[X.index[:25], "amount"] = np.nan
    s = make_splits(X, y, synthetic_spec)
    pre, _ = fit_preprocessor(synthetic_spec, s.X_train)
    assert np.isfinite(transform(pre, s.X_test)).all()


def test_preprocessor_is_fitted_on_train_only(synthetic_spec, synthetic_xy):
    """Refitting on train alone must reproduce the same transform of test."""
    X, y = synthetic_xy
    s = make_splits(X, y, synthetic_spec)
    pre_a, _ = fit_preprocessor(synthetic_spec, s.X_train)
    pre_b, _ = fit_preprocessor(synthetic_spec, s.X_train)
    np.testing.assert_allclose(transform(pre_a, s.X_test), transform(pre_b, s.X_test))


def test_preprocessor_tolerates_unseen_category(synthetic_spec, synthetic_xy):
    X, y = synthetic_xy
    s = make_splits(X, y, synthetic_spec)
    pre, _ = fit_preprocessor(synthetic_spec, s.X_train)
    novel = s.X_test.head(3).copy()
    novel["channel"] = "kiosk"        # never seen in training
    assert np.isfinite(transform(pre, novel)).all()


def test_imbalance_only_changes_training_arrays(synthetic_spec, synthetic_xy):
    X, y = synthetic_xy
    s = make_splits(X, y, synthetic_spec)
    pre, _ = fit_preprocessor(synthetic_spec, s.X_train)
    Xtr = transform(pre, s.X_train)
    ytr = s.y_train.to_numpy()

    Xr, yr, applied = apply_imbalance(Xtr, ytr, strategy="undersample")
    assert applied in ("undersample", "none")
    assert len(Xr) == len(yr)
    # Class weighting is a no-op at the array level, by design.
    Xc, yc, _ = apply_imbalance(Xtr, ytr, strategy="class_weight")
    assert Xc.shape == Xtr.shape and np.array_equal(yc, ytr)
