# ============================================================
# SMART AUTOMATIC ANALYTICS DASHBOARD
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# LOGIN ACCESS CONTROL
# ============================================================

if not st.session_state.get("logged_in", False):
    st.warning("🔐 Please login first to access the Dashboard.")
    st.stop()

st.title("📊 Smart Automatic Analytics Dashboard")


# ============================================================
# GET DATASET
# ============================================================

# Dataset comes only from the main application after login.
df = st.session_state.get("analytics_df")

if df is None or not isinstance(df, pd.DataFrame):
    st.warning("Please select or upload a dataset from the main application first.")
    st.stop()

df = df.copy()


# ============================================================
# BASIC CLEANING
# ============================================================

if df is None or df.empty:

    st.warning("Dataset is empty.")
    st.stop()


# Remove completely empty columns
df = df.dropna(axis=1, how="all")

# Remove duplicate rows
df = df.drop_duplicates()

# Keep original data for table
original_df = df.copy()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_column(column):

    return (
        str(column)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )


def is_technical_column(column, series):

    name = normalize_column(column)

    technical_names = [
        "unnamed",
        "row index",
        "index",
        "serial number",
        "serial no",
        "uuid"
    ]

    if any(x in name for x in technical_names):
        return True

    id_words = [
        "id",
        "identifier",
        "code",
        "phone",
        "mobile"
    ]

    unique_ratio = (
        series.nunique(dropna=True) /
        max(len(series.dropna()), 1)
    )

    for word in id_words:

        if name == word or name.endswith(" " + word):

            if unique_ratio > 0.50:
                return True

    return False


# ============================================================
# COLUMN DETECTION
# ============================================================

def detect_columns(data):

    numeric_columns = []
    categorical_columns = []
    date_columns = []
    ignored_columns = []

    for column in data.columns:

        series = data[column].dropna()

        if len(series) == 0:

            ignored_columns.append(column)
            continue

        # ----------------------------------------
        # Technical columns
        # ----------------------------------------

        if is_technical_column(column, series):

            ignored_columns.append(column)
            continue

        # ----------------------------------------
        # Numeric
        # ----------------------------------------

        if pd.api.types.is_numeric_dtype(series):

            if series.nunique() > 1:

                numeric_columns.append(column)

            continue

        # ----------------------------------------
        # Already datetime
        # ----------------------------------------

        if pd.api.types.is_datetime64_any_dtype(series):

            date_columns.append(column)
            continue

        # ----------------------------------------
        # Try detecting dates
        # ----------------------------------------

        name = normalize_column(column)

        date_words = [
            "date",
            "time",
            "month",
            "year",
            "day"
        ]

        if any(word in name for word in date_words):

            converted = pd.to_datetime(
                series,
                errors="coerce"
            )

            if converted.notna().mean() >= 0.70:

                data[column] = pd.to_datetime(
                    data[column],
                    errors="coerce"
                )

                date_columns.append(column)
                continue

        # ----------------------------------------
        # Categorical
        # ----------------------------------------

        unique_count = series.nunique()

        if (
            series.dtype == "object"
            or pd.api.types.is_categorical_dtype(series)
        ):

            # Avoid extremely high-cardinality text
            if unique_count <= 100:

                categorical_columns.append(column)

            else:

                ignored_columns.append(column)

    return (
        numeric_columns,
        categorical_columns,
        date_columns,
        ignored_columns
    )


(
    numeric_columns,
    categorical_columns,
    date_columns,
    ignored_columns
) = detect_columns(df)


# ============================================================
# SEMANTIC SCORING
# ============================================================

def numeric_score(column, purpose="general"):

    name = normalize_column(column)

    score = 0

    keywords = {

        "financial": [
            "revenue",
            "sales",
            "profit",
            "amount",
            "income",
            "cost",
            "expense",
            "price",
            "salary",
            "value"
        ],

        "volume": [
            "quantity",
            "qty",
            "units",
            "orders",
            "count",
            "patients",
            "customers",
            "employees",
            "students",
            "jobs",
            "transactions"
        ],

        "quality": [
            "rating",
            "score",
            "quality",
            "satisfaction",
            "percentage",
            "percent",
            "rate"
        ],

        "general": [
            "revenue",
            "sales",
            "profit",
            "amount",
            "cost",
            "expense",
            "price",
            "quantity",
            "qty",
            "units",
            "rating",
            "score",
            "salary",
            "experience",
            "age",
            "count",
            "value",
            "amount"
        ]
    }

    selected_words = keywords.get(
        purpose,
        keywords["general"]
    )

    for word in selected_words:

        if word in name:

            score += 10

    # Prefer columns with useful variation
    unique_count = df[column].nunique()

    if unique_count >= 5:
        score += 3

    # Binary fields should not normally be primary metrics
    if unique_count == 2:
        score -= 5

    return score


def category_score(column):

    name = normalize_column(column)

    unique_count = df[column].nunique()

    score = 0

    category_words = [
        "category",
        "type",
        "class",
        "region",
        "city",
        "state",
        "country",
        "department",
        "product",
        "company",
        "brand",
        "segment",
        "status",
        "gender",
        "group",
        "industry",
        "location",
        "role"
    ]

    for word in category_words:

        if word in name:

            score += 8

    # Ideal grouping
    if 2 <= unique_count <= 10:

        score += 10

    elif 11 <= unique_count <= 25:

        score += 7

    elif 26 <= unique_count <= 50:

        score += 3

    elif unique_count > 100:

        score -= 10

    return score


def get_best_numeric(
    purpose="general",
    exclude=None
):

    exclude = exclude or []

    available = [
        column
        for column in numeric_columns
        if column not in exclude
    ]

    if not available:

        return None

    return max(
        available,
        key=lambda x: numeric_score(
            x,
            purpose
        )
    )


def get_best_category(exclude=None):

    exclude = exclude or []

    available = [
        column
        for column in categorical_columns
        if column not in exclude
    ]

    if not available:

        return None

    return max(
        available,
        key=category_score
    )


# ============================================================
# BAR CHART
# ============================================================

def create_bar_chart():

    category = get_best_category()

    metric = get_best_numeric(
        "financial"
    )

    if category is None or metric is None:

        return None

    data = (
        df.groupby(
            category,
            dropna=False
        )[metric]
        .sum()
        .reset_index()
        .sort_values(
            metric,
            ascending=False
        )
        .head(15)
    )

    fig = px.bar(
        data,
        x=category,
        y=metric,
        title=f"{metric} by {category}"
    )

    return fig


# ============================================================
# PIE CHART
# ============================================================

def create_pie_chart():

    category = get_best_category()

    metric = get_best_numeric(
        "volume"
    )

    if category is None or metric is None:

        return None

    # Pie needs low-cardinality category
    if df[category].nunique() > 10:

        return None

    data = (
        df.groupby(
            category,
            dropna=False
        )[metric]
        .sum()
        .reset_index()
        .sort_values(
            metric,
            ascending=False
        )
        .head(8)
    )

    fig = px.pie(
        data,
        names=category,
        values=metric,
        title=f"{metric} Contribution by {category}"
    )

    return fig


# ============================================================
# LINE CHART
# ============================================================

def create_line_chart():

    metric = get_best_numeric(
        "financial"
    )

    if metric is None:

        return None

    # ----------------------------------------
    # Prefer Date/Time
    # ----------------------------------------

    if date_columns:

        date_column = date_columns[0]

        data = (
            df.groupby(
                date_column,
                dropna=False
            )[metric]
            .sum()
            .reset_index()
            .sort_values(date_column)
        )

        fig = px.line(
            data,
            x=date_column,
            y=metric,
            markers=True,
            title=f"{metric} Trend Over Time"
        )

        return fig

    # ----------------------------------------
    # No date → category based line
    # ----------------------------------------

    category = get_best_category()

    if category is None:

        return None

    data = (
        df.groupby(
            category,
            dropna=False
        )[metric]
        .sum()
        .reset_index()
    )

    fig = px.line(
        data,
        x=category,
        y=metric,
        markers=True,
        title=f"{metric} by {category}"
    )

    return fig


# ============================================================
# SCATTER PLOT
# ============================================================

def create_scatter_chart():

    if len(numeric_columns) < 2:

        return None

    # First metric
    x_column = get_best_numeric(
        "financial"
    )

    if x_column is None:

        return None

    # Second metric must be different
    y_column = get_best_numeric(
        "general",
        exclude=[x_column]
    )

    if y_column is None:

        return None

    fig = px.scatter(
        df,
        x=x_column,
        y=y_column,
        title=f"{y_column} vs {x_column}",
        opacity=0.70
    )

    return fig


# ============================================================
# BOX PLOT
# ============================================================

def create_box_chart():

    metric = get_best_numeric(
        "quality"
    )

    if metric is None:

        return None

    category = get_best_category()

    # Category + numeric
    if (
        category is not None
        and df[category].nunique() <= 20
    ):

        fig = px.box(
            df,
            x=category,
            y=metric,
            title=f"{metric} Distribution by {category}"
        )

        return fig

    # Numeric-only distribution
    fig = px.box(
        df,
        y=metric,
        title=f"{metric} Distribution"
    )

    return fig


# ============================================================
# HISTOGRAM
# ============================================================

def create_histogram():

    metric = get_best_numeric(
        "general"
    )

    if metric is None:

        return None

    fig = px.histogram(
        df,
        x=metric,
        nbins=30,
        title=f"{metric} Distribution"
    )

    return fig


# ============================================================
# CORRELATION HEATMAP
# ============================================================

def create_heatmap():

    if len(numeric_columns) < 2:

        return None

    selected_columns = sorted(
        numeric_columns,
        key=lambda x: numeric_score(
            x,
            "general"
        ),
        reverse=True
    )[:8]

    correlation = df[
        selected_columns
    ].corr()

    fig = px.imshow(
        correlation,
        text_auto=True,
        aspect="auto",
        title="Numeric Correlation Heatmap"
    )

    return fig


# ============================================================
# TOP 10
# ============================================================

def create_top10_chart():

    category = get_best_category()

    metric = get_best_numeric(
        "financial"
    )

    if category is None or metric is None:

        return None

    data = (
        df.groupby(
            category,
            dropna=False
        )[metric]
        .sum()
        .reset_index()
        .sort_values(
            metric,
            ascending=False
        )
        .head(10)
    )

    fig = px.bar(
        data,
        x=metric,
        y=category,
        orientation="h",
        title=f"Top 10 {category} by {metric}"
    )

    return fig


# ============================================================
# DATA TABLE
# ============================================================

def create_data_table():

    return original_df.head(100)


# ============================================================
# CHART LIST
# ============================================================

charts = {

    "📊 Bar Chart": create_bar_chart,

    "📈 Line Chart": create_line_chart,

    "🥧 Pie Chart": create_pie_chart,

    "🔵 Scatter Plot": create_scatter_chart,

    "📦 Box Plot": create_box_chart,

    "📉 Histogram": create_histogram,

    "🔥 Correlation Heatmap": create_heatmap,

    "🏆 Top 10": create_top10_chart
}


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("📊 Visualizations")

selected_chart = st.sidebar.radio(
    "Select Chart",
    [
        "🏠 Full Dashboard",
        *charts.keys(),
        "📋 Data Table"
    ]
)


# ============================================================
# FULL DASHBOARD
# ============================================================

if selected_chart == "🏠 Full Dashboard":

    st.subheader("📊 Automatic Visual Analysis")

    chart_count = 0

    for chart_name, chart_function in charts.items():

        try:

            figure = chart_function()

            if figure is not None:

                st.subheader(chart_name)

                st.plotly_chart(
                    figure,
                    use_container_width=True
                )

                chart_count += 1

        except Exception as e:

            # Don't break complete dashboard
            st.info(
                f"{chart_name} skipped because suitable data was not available."
            )

    if chart_count == 0:

        st.warning(
            "No suitable columns were found for automatic charts."
        )


# ============================================================
# INDIVIDUAL CHART
# ============================================================

elif selected_chart != "📋 Data Table":

    st.subheader(selected_chart)

    chart_function = charts[
        selected_chart
    ]

    try:

        figure = chart_function()

        if figure is not None:

            st.plotly_chart(
                figure,
                use_container_width=True
            )

        else:

            st.warning(
                "This dataset does not contain the required columns "
                "for this chart."
            )

    except Exception:

        st.warning(
            "This chart cannot be generated from the current dataset."
        )


# ============================================================
# DATA TABLE
# ============================================================

else:

    st.subheader("📋 Dataset")

    st.dataframe(
        original_df.head(100),
        use_container_width=True
    )


# ============================================================
# DATASET INFORMATION
# ============================================================

with st.expander("🔍 Dataset Information"):

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Rows",
        f"{len(df):,}"
    )

    col2.metric(
        "Columns",
        len(df.columns)
    )

    col3.metric(
        "Numeric Fields",
        len(numeric_columns)
    )

    col4.metric(
        "Category Fields",
        len(categorical_columns)
    )

    if date_columns:

        st.write(
            "**Date/Time Fields:**",
            ", ".join(
                map(str, date_columns)
            )
        )

    if ignored_columns:

        st.write(
            "**Ignored Technical Fields:**",
            ", ".join(
                map(str, ignored_columns)
            )
        )