"""Data logic for the CSV Insights Dashboard.

Everything here is plain pandas / scikit-learn with no Streamlit code,
so it can be tested on its own (see tests/test_insights.py).
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from pandas.api import types as ptypes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

NUMERIC, CATEGORY, DATETIME, TEXT = "number", "category", "date", "text"

AGGREGATIONS = {"sum": "Total", "mean": "Average", "median": "Median", "count": "Number of rows"}

MIN_MODEL_ROWS = 30
MAX_MODEL_ROWS = 20_000
MAX_CLASSES = 20

_DATE_LIKE = re.compile(r"^\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}([ T]\d{1,2}:\d{2}(:\d{2})?)?\s*$")


# ---------------------------------------------------------------- loading


def read_csv(source: str | Path | bytes) -> pd.DataFrame:
    """Read a CSV from a path or raw bytes, coping with common real-world quirks.

    Handles non-UTF-8 files (Excel exports are often Latin-1), semicolon or
    tab separated files, stray spaces in headers, and date columns stored as text.
    """
    def attempt(**kwargs) -> pd.DataFrame:
        data = io.BytesIO(source) if isinstance(source, bytes) else source
        try:
            return pd.read_csv(data, **kwargs)
        except UnicodeDecodeError:
            data = io.BytesIO(source) if isinstance(source, bytes) else source
            return pd.read_csv(data, encoding="latin-1", **kwargs)

    df = attempt()
    if df.shape[1] == 1 and any(sep in str(df.columns[0]) for sep in ";\t|"):
        # The whole header landed in one column, so the separator isn't a comma; let pandas sniff it.
        df = attempt(sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return parse_dates(df)


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert text columns that look like dates (2025-01-31, 31/01/2025, ...) to datetimes."""
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if not _is_text(s):
            continue
        sample = s.dropna().astype(str).head(200)
        if sample.empty or sample.str.match(_DATE_LIKE.pattern).mean() < 0.9:
            continue
        parsed = pd.to_datetime(s, errors="coerce", format="mixed")
        if parsed.notna().sum() >= 0.9 * s.notna().sum():
            out[col] = parsed
    return out


def _is_text(s: pd.Series) -> bool:
    return ptypes.is_object_dtype(s) or ptypes.is_string_dtype(s)


# ---------------------------------------------------------------- profiling


def column_kind(s: pd.Series) -> str:
    """Classify a column as number, category, date or text (free text / IDs)."""
    if ptypes.is_bool_dtype(s):
        return CATEGORY
    if ptypes.is_datetime64_any_dtype(s):
        return DATETIME
    if ptypes.is_numeric_dtype(s):
        return NUMERIC
    n_unique = s.nunique(dropna=True)
    if n_unique <= 50 or n_unique <= 0.05 * s.notna().sum():
        return CATEGORY
    return TEXT


def columns_of_kind(df: pd.DataFrame, *kinds: str) -> list[str]:
    return [c for c in df.columns if column_kind(df[c]) in kinds]


def profile(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Return headline numbers and a one-row-per-column summary table."""
    summary = {
        "rows": len(df),
        "columns": df.shape[1],
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    rows = []
    for col in df.columns:
        s = df[col]
        missing = int(s.isna().sum())
        example = s.dropna().iloc[0] if s.notna().any() else ""
        rows.append(
            {
                "column": col,
                "type": column_kind(s),
                "missing": missing,
                "missing %": round(100 * missing / len(df), 1) if len(df) else 0.0,
                "unique values": int(s.nunique(dropna=True)),
                "example": str(example),
            }
        )
    return summary, pd.DataFrame(rows)


# ---------------------------------------------------------------- cleaning


def clean(
    df: pd.DataFrame,
    *,
    trim_text: bool = True,
    drop_empty_columns: bool = True,
    drop_duplicates: bool = True,
    fill_missing: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Apply common cleaning steps and describe each change in plain English."""
    out = df.copy()
    changes: list[str] = []

    if trim_text:
        cells, cols = 0, 0
        for col in out.columns:
            s = out[col]
            if not _is_text(s):
                continue
            needs_trim = s.map(lambda v: isinstance(v, str) and v != v.strip())
            if needs_trim.any():
                out[col] = s.map(lambda v: v.strip() if isinstance(v, str) else v)
                cells += int(needs_trim.sum())
                cols += 1
        if cells:
            changes.append(f"Trimmed extra spaces in {cells:,} cells across {cols} text column(s).")

    if drop_empty_columns:
        empty = [c for c in out.columns if out[c].isna().all()]
        if empty:
            out = out.drop(columns=empty)
            changes.append(f"Removed {len(empty)} completely empty column(s): {', '.join(empty)}.")

    if drop_duplicates:
        n_dupes = int(out.duplicated().sum())
        if n_dupes:
            out = out.drop_duplicates().reset_index(drop=True)
            changes.append(f"Removed {n_dupes:,} duplicate row(s).")

    if fill_missing:
        for col in out.columns:
            s = out[col]
            n_missing = int(s.isna().sum())
            if not n_missing:
                continue
            kind = column_kind(s)
            if kind == NUMERIC:
                value = s.median()
                if ptypes.is_integer_dtype(s):  # nullable Int64 can't hold 2.5
                    value = round(value)
                out[col] = s.fillna(value)
                changes.append(f"Filled {n_missing:,} missing value(s) in '{col}' with the median ({value:g}).")
            elif kind in (CATEGORY, TEXT):
                modes = s.mode(dropna=True)
                if not modes.empty:
                    out[col] = s.fillna(modes.iloc[0])
                    changes.append(
                        f"Filled {n_missing:,} missing value(s) in '{col}' with the most common value ('{modes.iloc[0]}')."
                    )

    return out, changes


# ---------------------------------------------------------------- exploring


def top_counts(s: pd.Series, top: int = 15) -> pd.DataFrame:
    """Value counts, with everything past the top N folded into 'Other'."""
    counts = s.astype(str).where(s.notna(), "(missing)").value_counts()
    if len(counts) > top:
        other = counts.iloc[top:].sum()
        counts = pd.concat([counts.iloc[:top], pd.Series({"Other": other})])
    return counts.rename_axis("value").reset_index(name="rows")


def numeric_stats(s: pd.Series) -> dict[str, float]:
    return {
        "Average": s.mean(),
        "Median": s.median(),
        "Minimum": s.min(),
        "Maximum": s.max(),
        "Std. dev.": s.std(),
    }


def group_summary(df: pd.DataFrame, by: str, value: str | None, how: str) -> pd.DataFrame:
    """Aggregate a number column per group, largest first."""
    label = AGGREGATIONS[how] if how == "count" else f"{AGGREGATIONS[how]} {value}"
    grouped = df.groupby(by, dropna=False, observed=True)
    result = grouped.size() if how == "count" or value is None else grouped[value].agg(how)
    return (
        result.rename(label)
        .reset_index()
        .sort_values(label, ascending=False, kind="stable")
        .reset_index(drop=True)
    )


def time_trend(df: pd.DataFrame, date_col: str, value: str | None, how: str) -> tuple[pd.DataFrame, str]:
    """Aggregate per day, month or quarter (picked from the date range). Returns (table, period name)."""
    dates = df[date_col].dropna()
    span_days = (dates.max() - dates.min()).days if not dates.empty else 0
    freq, period = ("D", "day") if span_days <= 62 else ("MS", "month") if span_days <= 731 else ("QS", "quarter")

    resampled = df.dropna(subset=[date_col]).set_index(date_col).resample(freq)
    result = resampled.size() if how == "count" or value is None else resampled[value].agg(how)
    label = AGGREGATIONS[how] if how == "count" else f"{AGGREGATIONS[how]} {value}"
    return result.rename(label).rename_axis(date_col).reset_index(), period


def strongest_correlations(df: pd.DataFrame, top: int = 3) -> list[tuple[str, str, float]]:
    """The most strongly related pairs of number columns, strongest first."""
    corr = df[columns_of_kind(df, NUMERIC)].corr()
    pairs = [
        (a, b, corr.loc[a, b])
        for i, a in enumerate(corr.columns)
        for b in corr.columns[i + 1 :]
        if pd.notna(corr.loc[a, b])
    ]
    return sorted(pairs, key=lambda p: abs(p[2]), reverse=True)[:top]


# ---------------------------------------------------------------- predicting


@dataclass
class ModelResult:
    task: str  # "classification" or "regression"
    target: str
    metric_name: str
    score: float
    baseline: float
    higher_is_better: bool
    importances: pd.DataFrame  # columns: feature, importance (sums to 1)
    n_train: int
    n_test: int
    extra: dict[str, float] = field(default_factory=dict)
    ignored: list[str] = field(default_factory=list)


def model_targets(df: pd.DataFrame) -> list[str]:
    """Columns that make sense to predict: numbers, or categories with 2..MAX_CLASSES values."""
    targets = []
    for col in df.columns:
        kind = column_kind(df[col])
        n_unique = df[col].nunique(dropna=True)
        if (kind == NUMERIC and n_unique > 1) or (kind == CATEGORY and 2 <= n_unique <= MAX_CLASSES):
            targets.append(col)
    return targets


def _feature_matrix(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    """Turn every usable column into numbers. Returns (X, feature -> source column, ignored columns)."""
    parts, source, ignored = [], {}, []
    for col in df.columns:
        if col == target:
            continue
        s = df[col]
        kind = column_kind(s)
        if kind == NUMERIC:
            part = s.astype(float).fillna(s.median()).to_frame(col)
        elif kind == CATEGORY:
            filled = s.astype(str).where(s.notna(), "(missing)")
            part = pd.get_dummies(filled, prefix=col, prefix_sep="=", dtype=float)
        elif kind == DATETIME:
            part = pd.DataFrame(
                {f"{col} (month)": s.dt.month, f"{col} (weekday)": s.dt.dayofweek, f"{col} (year)": s.dt.year},
                index=s.index,
            ).astype(float)
            part = part.fillna(part.median())
        else:  # free text and ID columns don't generalise
            ignored.append(col)
            continue
        parts.append(part)
        source.update({feature: col for feature in part.columns})
    X = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)
    return X, source, ignored


def train_model(df: pd.DataFrame, target: str, seed: int = 42) -> ModelResult:
    """Train a random forest to predict `target` and compare it with a naive guess."""
    data = df[df[target].notna()]
    if len(data) > MAX_MODEL_ROWS:
        data = data.sample(MAX_MODEL_ROWS, random_state=seed)
    if len(data) < MIN_MODEL_ROWS:
        raise ValueError(f"Need at least {MIN_MODEL_ROWS} rows with a value in '{target}' (found {len(data)}).")

    X, source, ignored = _feature_matrix(data, target)
    if X.shape[1] == 0:
        raise ValueError("There are no other usable columns to learn from.")

    is_regression = column_kind(data[target]) == NUMERIC and data[target].nunique() > 10
    if is_regression:
        y = data[target].astype(float)
        stratify = None
    else:
        y = data[target].astype(str)
        if y.nunique() < 2:
            raise ValueError(f"'{target}' has only one value, so there is nothing to predict.")
        stratify = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=stratify
    )

    if is_regression:
        model = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        result = dict(
            task="regression",
            metric_name="Average error (MAE)",
            score=mean_absolute_error(y_test, pred),
            baseline=mean_absolute_error(y_test, [y_train.mean()] * len(y_test)),
            higher_is_better=False,
            extra={"R²": r2_score(y_test, pred)},
        )
    else:
        model = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
        model.fit(X_train, y_train)
        majority = y_train.mode().iloc[0]
        result = dict(
            task="classification",
            metric_name="Accuracy",
            score=accuracy_score(y_test, model.predict(X_test)),
            baseline=accuracy_score(y_test, [majority] * len(y_test)),
            higher_is_better=True,
        )

    importances = (
        pd.Series(model.feature_importances_, index=X.columns)
        .groupby(lambda feature: source[feature])
        .sum()
        .sort_values(ascending=False)
        .rename_axis("feature")
        .reset_index(name="importance")
    )
    return ModelResult(
        target=target,
        importances=importances,
        n_train=len(X_train),
        n_test=len(X_test),
        ignored=ignored,
        **result,
    )
