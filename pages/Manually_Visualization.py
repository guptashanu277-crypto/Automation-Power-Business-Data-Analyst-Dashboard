import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Manual Visualization",
    page_icon="🎨",
    layout="wide"
)


# =========================================================
# GET DATASET
# =========================================================

df = st.session_state.get("analytics_df")

if df is None or df.empty:
    st.warning(
        "Please load or upload a dataset from the main Analytics page first."
    )
    st.stop()

df = df.copy()


# =========================================================
# COLUMN DETECTION
# =========================================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()


# =========================================================
# SMART DATE / TIME DETECTION
# =========================================================

datetime_columns = []

date_keywords = [
    "date",
    "time",
    "datetime",
    "timestamp",
    "day",
    "month",
    "year"
]

for column in df.columns:

    if column in numeric_columns:
        continue

    column_name = str(column).lower().strip()

    name_looks_like_date = any(
        keyword in column_name
        for keyword in date_keywords
    )

    try:

        converted = pd.to_datetime(
            df[column],
            errors="coerce",
            format="mixed"
        )

        valid_count = converted.notna().sum()
        total_count = len(df[column])

        if total_count > 0:

            valid_ratio = valid_count / total_count

            if valid_ratio >= 0.50:

                datetime_columns.append(column)

            elif name_looks_like_date and valid_ratio >= 0.20:

                datetime_columns.append(column)

    except Exception:
        pass


# =========================================================
# CATEGORICAL COLUMNS
# =========================================================

categorical_columns = df.select_dtypes(
    exclude=np.number
).columns.tolist()

categorical_columns = [
    col
    for col in categorical_columns
    if col not in datetime_columns
]


# =========================================================
# SMART COLUMN FUNCTIONS
# =========================================================

def get_best_numeric(columns=None):

    available = (
        columns
        if columns is not None
        else numeric_columns
    )

    if not available:
        return None

    priority_words = [
        "sales",
        "revenue",
        "profit",
        "amount",
        "value",
        "price",
        "income",
        "quantity",
        "count",
        "rating",
        "score"
    ]

    for word in priority_words:

        for column in available:

            if word in str(column).lower():
                return column

    return available[0]


def get_best_category():

    if not categorical_columns:
        return None

    suitable = [
        col
        for col in categorical_columns
        if 2 <= df[col].nunique(dropna=True) <= 20
    ]

    if suitable:
        return suitable[0]

    return categorical_columns[0]


def get_best_date():

    if datetime_columns:
        return datetime_columns[0]

    return None


# =========================================================
# PAGE HEADER
# =========================================================

st.title("🎨 Manual Visualization")

st.markdown(
    "### Choose any chart for your analysis"
)


# =========================================================
# CHART SELECTION
# =========================================================

chart_type = st.selectbox(
    "Select Visualization",
    [
        "Select a chart",
        "📊 Bar Chart",
        "📈 Line Chart",
        "🥧 Pie Chart",
        "🔵 Scatter Plot",
        "📦 Box Plot",
        "📉 Histogram",
        "🌊 Area Chart",
        "🔥 Heatmap"
    ]
)


# =========================================================
# NO CHART SELECTED
# =========================================================

if chart_type == "Select a chart":

    st.info(
        "Select a chart type above to configure your visualization."
    )

    st.subheader("Available Dataset Columns")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.write("**Numeric Columns**")

        if numeric_columns:

            for col in numeric_columns:
                st.write(f"• {col}")

        else:

            st.write("No numeric columns found.")

    with col2:

        st.write("**Categorical Columns**")

        if categorical_columns:

            for col in categorical_columns:
                st.write(f"• {col}")

        else:

            st.write("No categorical columns found.")

    with col3:

        st.write("**Date / Time Columns**")

        if datetime_columns:

            for col in datetime_columns:
                st.write(f"• {col}")

        else:

            st.write("No date/time columns detected.")


# =========================================================
# BAR CHART
# =========================================================

elif chart_type == "📊 Bar Chart":

    st.subheader("Bar Chart")

    category_default = get_best_category()
    value_default = get_best_numeric()

    if not category_default or not value_default:

        st.warning(
            "Bar Chart requires one categorical column "
            "and one numeric column."
        )

    else:

        st.info(
            f"Suggested setup: Use **{category_default}** "
            f"as Category and **{value_default}** as Value."
        )

        category_col = st.selectbox(
            "Select Category",
            categorical_columns,
            index=categorical_columns.index(category_default),
            key="bar_category"
        )

        value_col = st.selectbox(
            "Select Value",
            numeric_columns,
            index=numeric_columns.index(value_default),
            key="bar_value"
        )

        if st.button(
            "Generate Visualization",
            key="bar_generate"
        ):

            grouped = (
                df.groupby(category_col)[value_col]
                .sum()
                .reset_index()
                .sort_values(
                    value_col,
                    ascending=False
                )
            )

            fig = px.bar(
                grouped,
                x=category_col,
                y=value_col,
                title=f"{value_col} by {category_col}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# =========================================================
# LINE CHART
# =========================================================

elif chart_type == "📈 Line Chart":

    st.subheader("Line Chart")

    value_default = get_best_numeric()
    category_default = get_best_category()
    date_default = get_best_date()

    # -----------------------------------------------------
    # OPTION 1: DATE/TIME + NUMERIC
    # -----------------------------------------------------

    if date_default and value_default:

        st.info(
            f"Suggested setup: **{date_default}** as Date/Time "
            f"and **{value_default}** as Value."
        )

        x_options = datetime_columns + categorical_columns

        x_default = date_default

        x_col = st.selectbox(
            "Select X Axis",
            x_options,
            index=x_options.index(x_default),
            key="line_x_axis"
        )

        value_col = st.selectbox(
            "Select Value",
            numeric_columns,
            index=numeric_columns.index(value_default),
            key="line_value"
        )

        if st.button(
            "Generate Visualization",
            key="line_generate"
        ):

            plot_df = df.copy()

            # Date/Time Line Chart
            if x_col in datetime_columns:

                plot_df[x_col] = pd.to_datetime(
                    plot_df[x_col],
                    errors="coerce",
                    format="mixed"
                )

                plot_df = plot_df.dropna(
                    subset=[x_col, value_col]
                )

                plot_df = plot_df.sort_values(
                    x_col
                )

                fig = px.line(
                    plot_df,
                    x=x_col,
                    y=value_col,
                    markers=True,
                    title=f"{value_col} over {x_col}"
                )

            # Category Line Chart
            else:

                grouped = (
                    plot_df.groupby(x_col)[value_col]
                    .sum()
                    .reset_index()
                )

                fig = px.line(
                    grouped,
                    x=x_col,
                    y=value_col,
                    markers=True,
                    title=f"{value_col} by {x_col}"
                )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


    # -----------------------------------------------------
    # OPTION 2: CATEGORY + NUMERIC
    # -----------------------------------------------------

    elif category_default and value_default:

        st.info(
            f"Your dataset does not require a Date/Time column "
            f"for this Line Chart. Suggested setup: "
            f"**{category_default}** as Category and "
            f"**{value_default}** as Value."
        )

        category_col = st.selectbox(
            "Select Category / X Axis",
            categorical_columns,
            index=categorical_columns.index(category_default),
            key="line_category"
        )

        value_col = st.selectbox(
            "Select Value",
            numeric_columns,
            index=numeric_columns.index(value_default),
            key="line_category_value"
        )

        if st.button(
            "Generate Visualization",
            key="line_category_generate"
        ):

            grouped = (
                df.groupby(category_col)[value_col]
                .sum()
                .reset_index()
            )

            fig = px.line(
                grouped,
                x=category_col,
                y=value_col,
                markers=True,
                title=f"{value_col} by {category_col}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


    # -----------------------------------------------------
    # NOTHING SUITABLE
    # -----------------------------------------------------

    else:

        st.warning(
            "Line Chart requires a numeric column and either "
            "a Date/Time or Categorical column."
        )


# =========================================================
# PIE CHART
# =========================================================

elif chart_type == "🥧 Pie Chart":

    st.subheader("Pie Chart")

    category_default = get_best_category()
    value_default = get_best_numeric()

    if not category_default or not value_default:

        st.warning(
            "Pie Chart requires one categorical column "
            "and one numeric column."
        )

    else:

        st.info(
            f"Suggested setup: **{category_default}** for Category "
            f"and **{value_default}** for Value."
        )

        category_col = st.selectbox(
            "Select Category",
            categorical_columns,
            index=categorical_columns.index(category_default),
            key="pie_category"
        )

        value_col = st.selectbox(
            "Select Value",
            numeric_columns,
            index=numeric_columns.index(value_default),
            key="pie_value"
        )

        if st.button(
            "Generate Visualization",
            key="pie_generate"
        ):

            grouped = (
                df.groupby(category_col)[value_col]
                .sum()
                .reset_index()
            )

            fig = px.pie(
                grouped,
                names=category_col,
                values=value_col,
                title=f"{value_col} Distribution by {category_col}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# =========================================================
# SCATTER PLOT
# =========================================================

elif chart_type == "🔵 Scatter Plot":

    st.subheader("Scatter Plot")

    if len(numeric_columns) < 2:

        st.warning(
            "Scatter Plot requires at least two numeric columns."
        )

    else:

        x_default = numeric_columns[0]
        y_default = numeric_columns[1]

        st.info(
            f"Suggested setup: Compare **{x_default}** "
            f"with **{y_default}**."
        )

        x_col = st.selectbox(
            "Select X Axis",
            numeric_columns,
            index=numeric_columns.index(x_default),
            key="scatter_x"
        )

        y_col = st.selectbox(
            "Select Y Axis",
            numeric_columns,
            index=numeric_columns.index(y_default),
            key="scatter_y"
        )

        color_options = [
            "None"
        ] + categorical_columns

        color_col = st.selectbox(
            "Optional Category",
            color_options,
            key="scatter_color"
        )

        if st.button(
            "Generate Visualization",
            key="scatter_generate"
        ):

            if color_col == "None":

                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} vs {x_col}"
                )

            else:

                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_col,
                    color=color_col,
                    title=f"{y_col} vs {x_col}"
                )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# =========================================================
# BOX PLOT
# =========================================================

elif chart_type == "📦 Box Plot":

    st.subheader("Box Plot")

    value_default = get_best_numeric()

    if not value_default:

        st.warning(
            "Box Plot requires at least one numeric column."
        )

    else:

        st.info(
            f"Suggested setup: **{value_default}** "
            f"as the numeric value."
        )

        value_col = st.selectbox(
            "Select Numeric Column",
            numeric_columns,
            index=numeric_columns.index(value_default),
            key="box_value"
        )

        category_options = [
            "None"
        ] + categorical_columns

        category_col = st.selectbox(
            "Optional Category",
            category_options,
            key="box_category"
        )

        if st.button(
            "Generate Visualization",
            key="box_generate"
        ):

            if category_col == "None":

                fig = px.box(
                    df,
                    y=value_col,
                    title=f"Distribution of {value_col}"
                )

            else:

                fig = px.box(
                    df,
                    x=category_col,
                    y=value_col,
                    title=f"{value_col} Distribution by {category_col}"
                )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# =========================================================
# HISTOGRAM
# =========================================================

elif chart_type == "📉 Histogram":

    st.subheader("Histogram")

    value_default = get_best_numeric()

    if not value_default:

        st.warning(
            "Histogram requires at least one numeric column."
        )

    else:

        st.info(
            f"Suggested setup: Analyze the distribution "
            f"of **{value_default}**."
        )

        value_col = st.selectbox(
            "Select Numeric Column",
            numeric_columns,
            index=numeric_columns.index(value_default),
            key="hist_value"
        )

        if st.button(
            "Generate Visualization",
            key="hist_generate"
        ):

            fig = px.histogram(
                df,
                x=value_col,
                title=f"Distribution of {value_col}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# =========================================================
# AREA CHART
# =========================================================

elif chart_type == "🌊 Area Chart":

    st.subheader("Area Chart")

    value_default = get_best_numeric()
    date_default = get_best_date()
    category_default = get_best_category()

    # Date + Numeric
    if date_default and value_default:

        st.info(
            f"Suggested setup: **{date_default}** as Date/Time "
            f"and **{value_default}** as Value."
        )

        x_options = datetime_columns + categorical_columns

        x_col = st.selectbox(
            "Select X Axis",
            x_options,
            index=x_options.index(date_default),
            key="area_x"
        )

        value_col = st.selectbox(
            "Select Value",
            numeric_columns,
            index=numeric_columns.index(value_default),
            key="area_value"
        )

        if st.button(
            "Generate Visualization",
            key="area_generate"
        ):

            plot_df = df.copy()

            if x_col in datetime_columns:

                plot_df[x_col] = pd.to_datetime(
                    plot_df[x_col],
                    errors="coerce",
                    format="mixed"
                )

                plot_df = plot_df.dropna(
                    subset=[x_col, value_col]
                )

                plot_df = plot_df.sort_values(x_col)

            else:

                plot_df = (
                    plot_df.groupby(x_col)[value_col]
                    .sum()
                    .reset_index()
                )

            fig = px.area(
                plot_df,
                x=x_col,
                y=value_col,
                title=f"{value_col} by {x_col}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # Category + Numeric
    elif category_default and value_default:

        st.info(
            f"Suggested setup: **{category_default}** "
            f"as Category and **{value_default}** as Value."
        )

        category_col = st.selectbox(
            "Select Category / X Axis",
            categorical_columns,
            index=categorical_columns.index(category_default),
            key="area_category"
        )

        value_col = st.selectbox(
            "Select Value",
            numeric_columns,
            index=numeric_columns.index(value_default),
            key="area_category_value"
        )

        if st.button(
            "Generate Visualization",
            key="area_category_generate"
        ):

            grouped = (
                df.groupby(category_col)[value_col]
                .sum()
                .reset_index()
            )

            fig = px.area(
                grouped,
                x=category_col,
                y=value_col,
                title=f"{value_col} by {category_col}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        st.warning(
            "Area Chart requires a numeric column and either "
            "a Date/Time or Categorical column."
        )


# =========================================================
# HEATMAP
# =========================================================

elif chart_type == "🔥 Heatmap":

    st.subheader("Correlation Heatmap")

    if len(numeric_columns) < 2:

        st.warning(
            "Heatmap requires at least two numeric columns."
        )

    else:

        st.info(
            "Select numeric columns to analyze relationships "
            "between variables."
        )

        selected_columns = st.multiselect(
            "Select Numeric Columns",
            numeric_columns,
            default=numeric_columns,
            key="heatmap_columns"
        )

        if st.button(
            "Generate Visualization",
            key="heatmap_generate"
        ):

            if len(selected_columns) < 2:

                st.warning(
                    "Please select at least two numeric columns."
                )

            else:

                corr = df[
                    selected_columns
                ].corr()

                fig = px.imshow(
                    corr,
                    text_auto=True,
                    aspect="auto",
                    title="Correlation Heatmap"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


# =========================================================
# DATASET INFORMATION
# =========================================================

with st.expander("Dataset Column Information"):

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Columns",
            len(df.columns)
        )

    with col2:

        st.metric(
            "Numeric Columns",
            len(numeric_columns)
        )

    with col3:

        st.metric(
            "Date/Time Columns",
            len(datetime_columns)
        )

    st.write("### Detected Date/Time Columns")

    if datetime_columns:
        st.write(datetime_columns)
    else:
        st.write("No Date/Time columns detected.")

    st.write("### Detected Numeric Columns")

    if numeric_columns:
        st.write(numeric_columns)
    else:
        st.write("No numeric columns detected.")

    st.write("### Detected Categorical Columns")

    if categorical_columns:
        st.write(categorical_columns)
    else:
        st.write("No categorical columns detected.")