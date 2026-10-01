import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import insights


def messy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "city": [" Pune", "Pune", "Delhi", None, "Delhi", "Delhi"],
            "sales": [10.0, 10.0, None, 7.0, 8.0, 8.0],
            "empty": [None] * 6,
        }
    )


def test_read_csv_parses_dates_and_trims_headers():
    df = insights.read_csv(b" day ,amount\n2025-01-01,5\n2025-01-02,7\n")
    assert list(df.columns) == ["day", "amount"]
    assert insights.column_kind(df["day"]) == insights.DATETIME
    assert insights.column_kind(df["amount"]) == insights.NUMERIC


def test_read_csv_handles_semicolons_and_latin1():
    df = insights.read_csv("name;price\nCafé;3\nThé;4\n".encode("latin-1"))
    assert list(df.columns) == ["name", "price"]
    assert df["name"].tolist() == ["Café", "Thé"]


def test_read_csv_keeps_a_single_column_file_intact():
    df = insights.read_csv(b"value\n" + b"\n".join(str(i).encode() for i in range(50)))
    assert list(df.columns) == ["value"]
    assert len(df) == 50


def test_read_csv_leaves_codes_that_are_not_dates_alone():
    df = insights.read_csv(b"code,qty\nA-12,1\nB-7,2\n")
    assert insights.column_kind(df["code"]) == insights.CATEGORY


def test_column_kind_detects_ids_as_text():
    ids = pd.Series([f"ORD-{i}" for i in range(500)])
    assert insights.column_kind(ids) == insights.TEXT
    assert insights.column_kind(pd.Series(["a", "b"] * 250)) == insights.CATEGORY


def test_profile_counts_missing_and_duplicates():
    summary, columns = insights.profile(messy_df())
    assert summary == {"rows": 6, "columns": 3, "missing_cells": 8, "duplicate_rows": 1}
    assert columns.set_index("column").loc["sales", "missing"] == 1


def test_clean_trims_before_removing_duplicates_and_fills_missing():
    cleaned, changes = insights.clean(messy_df())
    # " Pune" becomes "Pune", which makes rows 0 and 1 duplicates.
    assert len(cleaned) == 4
    assert "empty" not in cleaned.columns
    assert cleaned.isna().sum().sum() == 0
    assert cleaned["city"].tolist().count("Pune") == 1
    assert len(changes) == 5


def test_clean_with_everything_off_changes_nothing():
    df = messy_df()
    cleaned, changes = insights.clean(
        df, trim_text=False, drop_empty_columns=False, drop_duplicates=False, fill_missing=False
    )
    pd.testing.assert_frame_equal(cleaned, df)
    assert changes == []


def test_group_summary_sorts_largest_first():
    df = pd.DataFrame({"team": ["a", "b", "b", "c"], "pts": [1, 5, 5, 3]})
    table = insights.group_summary(df, "team", "pts", "sum")
    assert table["team"].tolist() == ["b", "c", "a"]
    assert table.columns[1] == "Total pts"
    assert insights.group_summary(df, "team", None, "count").iloc[0].tolist() == ["b", 2]


def test_top_counts_folds_the_tail_into_other():
    counts = insights.top_counts(pd.Series(list("aaabbc") + [None]), top=2)
    assert counts["value"].tolist() == ["a", "b", "Other"]
    assert counts["rows"].tolist() == [3, 2, 2]


def test_time_trend_picks_monthly_periods_for_a_year():
    dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    df = pd.DataFrame({"day": dates, "amount": 1.0})
    table, period = insights.time_trend(df, "day", "amount", "sum")
    assert period == "month"
    assert len(table) == 12
    assert table.iloc[0, 1] == 31


def _signal_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    noise = rng.normal(size=n)
    return pd.DataFrame(
        {
            "x": x,
            "noise": noise,
            "group": rng.choice(["a", "b"], n),
            "y": 3 * x + rng.normal(0, 0.1, n),
            "label": np.where(x > 0, "high", "low"),
            "id": [f"row-{i}" for i in range(n)],
        }
    )


def test_train_model_regression_beats_baseline():
    result = insights.train_model(_signal_df().drop(columns="label"), "y")
    assert result.task == "regression"
    assert result.score < result.baseline / 3
    assert result.importances.iloc[0]["feature"] == "x"
    assert result.ignored == ["id"]
    assert result.importances["importance"].sum() == pytest.approx(1)


def test_train_model_classification_beats_baseline():
    result = insights.train_model(_signal_df().drop(columns="y"), "label")
    assert result.task == "classification"
    assert result.score > 0.9 > result.baseline
    assert result.importances.iloc[0]["feature"] == "x"


def test_train_model_needs_enough_rows():
    with pytest.raises(ValueError, match="at least"):
        insights.train_model(_signal_df(n=10), "y")


def test_model_targets_skip_ids_and_dates():
    df = _signal_df()
    df["when"] = pd.Timestamp("2025-01-01")
    assert "id" not in insights.model_targets(df)
    assert "when" not in insights.model_targets(df)
    assert {"x", "label", "group"} <= set(insights.model_targets(df))


def test_app_runs_on_sample_data_without_errors():
    app = AppTest.from_file("../app.py", default_timeout=120).run()
    assert not app.exception
    assert [t.label for t in app.tabs] == ["Overview", "Cleaning", "Explore", "Predict"]
    assert app.metric[0].value == "1,215"
