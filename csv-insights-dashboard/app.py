"""CSV Insights Dashboard: upload a CSV and get a profile, cleaning, charts and a quick prediction model.

Run locally:  streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

import charts
import insights

SAMPLE_PATH = Path(__file__).parent / "sample_data" / "sales.csv"

st.set_page_config(page_title="CSV Insights Dashboard", page_icon="📊", layout="wide")


@st.cache_data(show_spinner=False)
def load(data: bytes | None) -> pd.DataFrame:
    return insights.read_csv(SAMPLE_PATH if data is None else data)


@st.cache_data(show_spinner=False)
def cleaned(df: pd.DataFrame, **options) -> tuple[pd.DataFrame, list[str]]:
    return insights.clean(df, **options)


@st.cache_data(show_spinner=False)
def trained(df: pd.DataFrame, target: str) -> insights.ModelResult:
    return insights.train_model(df, target)


def theme_mode() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except AttributeError:  # older Streamlit versions
        return "light"


def fmt(value: float) -> str:
    return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:,.4g}"


# ---------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("1. Your data")
    uploaded = st.file_uploader("Upload a CSV file", type="csv")
    st.caption("No file yet? The app uses a sample sales dataset so you can try everything.")

    st.header("2. Cleaning")
    options = dict(
        trim_text=st.checkbox("Trim extra spaces in text", value=True),
        drop_empty_columns=st.checkbox("Remove empty columns", value=True),
        drop_duplicates=st.checkbox("Remove duplicate rows", value=True),
        fill_missing=st.checkbox(
            "Fill missing values", value=True, help="Numbers get the median, text gets the most common value."
        ),
    )
    st.caption("Explore and Predict use the cleaned data.")

try:
    raw = load(uploaded.getvalue() if uploaded else None)
except Exception as exc:  # any unreadable file should explain itself, not crash the app
    st.error(f"Couldn't read that file as a CSV: {exc}")
    st.stop()

if raw.empty:
    st.warning("That file has no rows.")
    st.stop()

df, changes = cleaned(raw, **options)
mode = theme_mode()
source_name = uploaded.name if uploaded else "sample sales data"

st.title("📊 CSV Insights Dashboard")
st.caption(f"Analysing **{source_name}**: {len(raw):,} rows × {raw.shape[1]} columns")

overview_tab, cleaning_tab, explore_tab, predict_tab = st.tabs(["Overview", "Cleaning", "Explore", "Predict"])

# ---------------------------------------------------------------- overview

with overview_tab:
    summary, column_table = insights.profile(raw)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{summary['rows']:,}")
    c2.metric("Columns", summary["columns"])
    c3.metric("Missing cells", f"{summary['missing_cells']:,}")
    c4.metric("Duplicate rows", f"{summary['duplicate_rows']:,}")

    st.subheader("Columns")
    st.dataframe(column_table, hide_index=True, width="stretch")
    st.subheader("First 100 rows")
    st.dataframe(raw.head(100), width="stretch")

# ---------------------------------------------------------------- cleaning

with cleaning_tab:
    st.subheader("What was fixed")
    if changes:
        st.markdown("\n".join(f"- {change}" for change in changes))
    else:
        st.success("Nothing needed fixing with the selected options.")

    c1, c2 = st.columns(2)
    c1.metric("Rows after cleaning", f"{len(df):,}", delta=f"{len(df) - len(raw):,}", delta_color="off")
    c2.metric("Missing cells after cleaning", f"{int(df.isna().sum().sum()):,}")

    st.download_button(
        "Download cleaned CSV",
        data=df.to_csv(index=False),
        file_name=f"cleaned_{uploaded.name if uploaded else 'sales.csv'}",
        mime="text/csv",
        type="primary",
    )
    st.dataframe(df.head(100), width="stretch")

# ---------------------------------------------------------------- explore

with explore_tab:
    numbers = insights.columns_of_kind(df, insights.NUMERIC)
    groups = insights.columns_of_kind(df, insights.CATEGORY)
    dates = insights.columns_of_kind(df, insights.DATETIME)

    st.subheader("Look at one column")
    first_chartable = next((i for i, c in enumerate(df.columns) if c in numbers + groups), 0)
    column = st.selectbox("Column", df.columns, index=first_chartable)
    kind = insights.column_kind(df[column])
    if kind == insights.NUMERIC:
        stats = insights.numeric_stats(df[column])
        for col, (name, value) in zip(st.columns(len(stats)), stats.items()):
            col.metric(name, fmt(value))
        st.plotly_chart(charts.histogram(df[column], mode))
    elif kind == insights.DATETIME:
        span = df[column].dropna()
        st.write(f"From **{span.min():%d %b %Y}** to **{span.max():%d %b %Y}**.")
        table, period = insights.time_trend(df, column, None, "count")
        st.caption(f"Rows per {period}")
        st.plotly_chart(charts.trend_line(table, column, table.columns[1], mode))
    elif kind == insights.TEXT:
        st.write(
            f"**{df[column].nunique():,}** different values in {len(df):,} rows, so this looks like an ID "
            "or free-text column. Charts and the model skip it."
        )
        st.dataframe(df[column].dropna().head(10), hide_index=True)
    else:
        counts = insights.top_counts(df[column])
        st.write(f"**{df[column].nunique():,}** different values." + (" Showing the 15 most common." if len(counts) > 15 else ""))
        st.plotly_chart(charts.ranked_bars(counts, "value", "rows", mode))

    if groups and numbers:
        st.subheader("Compare groups")
        c1, c2, c3 = st.columns(3)
        by = c1.selectbox("Group by", groups)
        how = c3.selectbox("Show", list(insights.AGGREGATIONS), format_func=insights.AGGREGATIONS.get, key="group_how")
        value = c2.selectbox("Of", numbers, disabled=how == "count", key="group_value")
        table = insights.group_summary(df, by, value, how).head(25)
        st.plotly_chart(charts.ranked_bars(table, by, table.columns[1], mode))
        with st.expander("See the numbers"):
            st.dataframe(table, hide_index=True, width="stretch")

    if dates and numbers:
        st.subheader("Trend over time")
        c1, c2, c3 = st.columns(3)
        date_col = c1.selectbox("Date column", dates)
        how = c3.selectbox("Show", list(insights.AGGREGATIONS), format_func=insights.AGGREGATIONS.get, key="trend_how")
        default_value = numbers.index("revenue") if "revenue" in numbers else 0
        value = c2.selectbox("Of", numbers, index=default_value, disabled=how == "count", key="trend_value")
        table, period = insights.time_trend(df, date_col, value, how)
        st.caption(f"{table.columns[1]} per {period}")
        st.plotly_chart(charts.trend_line(table, date_col, table.columns[1], mode))

    if len(numbers) >= 2:
        st.subheader("Which numbers move together?")
        st.plotly_chart(charts.correlation_heatmap(df[numbers].corr(), mode))
        for a, b, r in insights.strongest_correlations(df):
            direction = "rise together" if r > 0 else "move in opposite directions"
            strength = "strongly" if abs(r) >= 0.6 else "moderately" if abs(r) >= 0.3 else "weakly"
            st.markdown(f"- **{a}** and **{b}** {strength} {direction} (correlation {r:+.2f}).")

# ---------------------------------------------------------------- predict

with predict_tab:
    targets = insights.model_targets(df)
    if not targets:
        st.info("This file has no column that can be predicted (it needs a number or category column).")
    else:
        st.write(
            "Pick a column and a machine-learning model (random forest) learns to predict it from the "
            "other columns. It is tested on 25% of rows it never saw, and compared with a naive guess."
        )
        default = targets.index("returned") if "returned" in targets else len(targets) - 1
        target = st.selectbox("Predict this column", targets, index=default)
        try:
            with st.spinner("Training model..."):
                result = trained(df, target)
        except ValueError as exc:
            st.warning(str(exc))
        else:
            naive = "always guessing the most common value" if result.task == "classification" else "always guessing the average"
            better = result.score > result.baseline if result.higher_is_better else result.score < result.baseline
            show = (lambda v: f"{v:.1%}") if result.task == "classification" else fmt

            c1, c2, *rest = st.columns(2 + len(result.extra))
            c1.metric(f"Model {result.metric_name.lower()}", show(result.score))
            c2.metric(f"Naive guess ({naive})", show(result.baseline))
            for col, (name, value) in zip(rest, result.extra.items()):
                col.metric(name, f"{value:.2f}")

            if better:
                st.success(f"The model predicts **{target}** better than {naive}.")
            else:
                st.warning(
                    f"The model is no better than {naive}: the other columns don't carry enough signal "
                    f"to predict **{target}**. That is a useful finding too."
                )

            st.subheader(f"What drives {target}?")
            top = result.importances.head(10).assign(importance=lambda t: (t.importance * 100).round(1))
            st.plotly_chart(charts.ranked_bars(top.rename(columns={"importance": "importance %"}), "feature", "importance %", mode))
            st.caption(
                f"Share of the model's decisions that relied on each column. "
                f"Trained on {result.n_train:,} rows, tested on {result.n_test:,}."
                + (f" Ignored ID/free-text columns: {', '.join(result.ignored)}." if result.ignored else "")
            )

st.divider()
st.caption("Built with Python, pandas, scikit-learn, Plotly and Streamlit.")
