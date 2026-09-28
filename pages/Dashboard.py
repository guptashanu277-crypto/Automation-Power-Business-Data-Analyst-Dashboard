import streamlit as st
import pandas as pd
import plotly.express as px


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Visual Dashboard",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# GET DATA FROM MAIN APP
# =========================================================

df = st.session_state.get("analytics_df")

if df is None:
    st.warning(
        "Please open the main Analytics page and select a dataset first."
    )
    st.stop()

df = df.copy()


# =========================================================
# AUTOMATIC COLUMN DETECTION
# =========================================================

numeric_cols = df.select_dtypes(
    include="number"
).columns.tolist()

categorical_cols = df.select_dtypes(
    include=["object", "category"]
).columns.tolist()


def get_category_column():

    suitable = [
        col
        for col in categorical_cols
        if 1 < df[col].nunique() <= 20
    ]

    if suitable:
        return suitable[0]

    if categorical_cols:
        return categorical_cols[0]

    return None


def get_value_column():

    if not numeric_cols:
        return None

    priority_words = [
        "sales",
        "revenue",
        "profit",
        "amount",
        "value",
        "price",
        "income",
        "quantity"
    ]

    for col in numeric_cols:

        if any(
            word in col.lower()
            for word in priority_words
        ):
            return col

    return numeric_cols[0]


category_col = get_category_column()
value_col = get_value_column()


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_grouped_data():

    if not category_col or not value_col:
        return None

    data = (
        df.groupby(category_col)[value_col]
        .sum()
        .reset_index()
    )

    return data


def get_highest_lowest():

    data = get_grouped_data()

    if data is None or data.empty:
        return None, None

    highest = data.loc[
        data[value_col].idxmax()
    ]

    lowest = data.loc[
        data[value_col].idxmin()
    ]

    return highest, lowest


def show_category_insight():

    highest, lowest = get_highest_lowest()

    if highest is not None and lowest is not None:

        st.info(
            f"{value_col} comparison across different "
            f"{category_col}. "
            f"Highest: {highest[category_col]} "
            f"({highest[value_col]:,.0f}). "
            f"Lowest: {lowest[category_col]} "
            f"({lowest[value_col]:,.0f})."
        )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📊 Dashboard")

page = st.sidebar.radio(
    "Select Chart",
    [
        "🏠 Full Dashboard",
        "📈 Line Chart",
        "📊 Bar Chart",
        "🥧 Pie Chart",
        "🔵 Scatter Plot",
        "📦 Box Plot",
        "📉 Histogram",
        "🔥 Correlation Heatmap",
        "🏆 Top 10",
        "📋 Data Table"
    ]
)


# =========================================================
# MAIN TITLE
# =========================================================

st.title("📊 AI Smart Visual Dashboard")

st.caption(
    f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns"
)


# =========================================================
# FULL DASHBOARD
# =========================================================

if page == "🏠 Full Dashboard":

    st.subheader("📌 Business Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Rows",
        len(df)
    )

    c2.metric(
        "Total Columns",
        len(df.columns)
    )

    if value_col:

        c3.metric(
            f"Total {value_col}",
            f"{df[value_col].sum():,.0f}"
        )

        c4.metric(
            f"Average {value_col}",
            f"{df[value_col].mean():,.2f}"
        )

    else:

        c3.metric(
            "Numeric Columns",
            len(numeric_cols)
        )

        c4.metric(
            "Categories",
            len(categorical_cols)
        )

    st.divider()


    # =====================================================
    # LINE + BAR
    # =====================================================

    col1, col2 = st.columns(2)


    # -------------------------
    # LINE CHART
    # -------------------------

    with col1:

        st.subheader("📈 Trend")

        if category_col and value_col:

            line_data = (
                df.groupby(category_col)[value_col]
                .sum()
                .reset_index()
            )

            show_category_insight()

            fig = px.line(
                line_data,
                x=category_col,
                y=value_col,
                markers=True,
                title=f"{value_col} by {category_col}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info(
                "Suitable columns were not found."
            )


    # -------------------------
    # BAR CHART
    # -------------------------

    with col2:

        st.subheader("📊 Category Analysis")

        if category_col and value_col:

            bar_data = (
                df.groupby(category_col)[value_col]
                .sum()
                .reset_index()
                .sort_values(
                    value_col,
                    ascending=False
                )
            )

            show_category_insight()

            fig = px.bar(
                bar_data,
                x=category_col,
                y=value_col,
                title=f"{value_col} by {category_col}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info(
                "Suitable columns were not found."
            )


    # =====================================================
    # PIE + HISTOGRAM
    # =====================================================

    col3, col4 = st.columns(2)


    # -------------------------
    # PIE CHART
    # -------------------------

    with col3:

        st.subheader("🥧 Distribution")

        if category_col and value_col:

            pie_data = (
                df.groupby(category_col)[value_col]
                .sum()
                .reset_index()
            )

            show_category_insight()

            fig = px.pie(
                pie_data,
                names=category_col,
                values=value_col,
                title=f"{value_col} Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info(
                "Suitable columns were not found."
            )


    # -------------------------
    # HISTOGRAM
    # -------------------------

    with col4:

        st.subheader("📉 Value Distribution")

        if value_col:

            st.info(
                f"Distribution of {value_col} values."
            )

            fig = px.histogram(
                df,
                x=value_col,
                title=f"{value_col} Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info(
                "No numeric column was found."
            )


# =========================================================
# LINE CHART
# =========================================================

elif page == "📈 Line Chart":

    st.subheader("📈 Line Chart")

    if category_col and value_col:

        data = (
            df.groupby(category_col)[value_col]
            .sum()
            .reset_index()
        )

        show_category_insight()

        fig = px.line(
            data,
            x=category_col,
            y=value_col,
            markers=True,
            title=f"{value_col} by {category_col}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "Suitable columns were not found."
        )


# =========================================================
# BAR CHART
# =========================================================

elif page == "📊 Bar Chart":

    st.subheader("📊 Bar Chart")

    if category_col and value_col:

        data = (
            df.groupby(category_col)[value_col]
            .sum()
            .reset_index()
            .sort_values(
                value_col,
                ascending=False
            )
        )

        show_category_insight()

        fig = px.bar(
            data,
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
            "Suitable columns were not found."
        )


# =========================================================
# PIE CHART
# =========================================================

elif page == "🥧 Pie Chart":

    st.subheader("🥧 Pie Chart")

    if category_col and value_col:

        data = (
            df.groupby(category_col)[value_col]
            .sum()
            .reset_index()
        )

        highest, lowest = get_highest_lowest()

        if highest is not None:

            st.info(
                f"Shows each {category_col}'s contribution "
                f"to total {value_col}. "
                f"Highest contribution: "
                f"{highest[category_col]} "
                f"({highest[value_col]:,.0f})."
            )

        fig = px.pie(
            data,
            names=category_col,
            values=value_col,
            title=f"{value_col} Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "Suitable columns were not found."
        )


# =========================================================
# SCATTER PLOT
# =========================================================

elif page == "🔵 Scatter Plot":

    st.subheader("🔵 Scatter Plot")

    if len(numeric_cols) >= 2:

        x_col = numeric_cols[0]
        y_col = numeric_cols[1]

        correlation = df[x_col].corr(
            df[y_col]
        )

        relationship = "positive" if correlation > 0 else "negative"

        st.info(
            f"Shows the relationship between "
            f"{x_col} and {y_col}. "
            f"Correlation: {correlation:.2f} "
            f"({relationship})."
        )

        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            title=f"{x_col} vs {y_col}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "At least 2 numeric columns are required."
        )


# =========================================================
# BOX PLOT
# =========================================================

elif page == "📦 Box Plot":

    st.subheader("📦 Box Plot")

    if category_col and value_col:

        st.info(
            f"Shows {value_col} variation across "
            f"different {category_col}."
        )

        fig = px.box(
            df,
            x=category_col,
            y=value_col,
            title=f"{value_col} by {category_col}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    elif value_col:

        st.info(
            f"Shows the range and variation of "
            f"{value_col} values."
        )

        fig = px.box(
            df,
            y=value_col,
            title=f"{value_col} Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "No numeric column was found."
        )


# =========================================================
# HISTOGRAM
# =========================================================

elif page == "📉 Histogram":

    st.subheader("📉 Histogram")

    if value_col:

        st.info(
            f"Shows the distribution of {value_col} values."
        )

        fig = px.histogram(
            df,
            x=value_col,
            nbins=20,
            title=f"{value_col} Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "No numeric column was found."
        )


# =========================================================
# CORRELATION HEATMAP
# =========================================================

elif page == "🔥 Correlation Heatmap":

    st.subheader("🔥 Correlation Heatmap")

    if len(numeric_cols) >= 2:

        corr = df[numeric_cols].corr()

        strongest_pair = None
        strongest_value = 0

        for i in range(len(corr.columns)):

            for j in range(i + 1, len(corr.columns)):

                value = abs(
                    corr.iloc[i, j]
                )

                if value > strongest_value:

                    strongest_value = value

                    strongest_pair = (
                        corr.columns[i],
                        corr.columns[j],
                        corr.iloc[i, j]
                    )

        if strongest_pair:

            col_a, col_b, corr_value = strongest_pair

            st.info(
                f"Shows relationships between numerical columns. "
                f"Strongest relationship: {col_a} and {col_b} "
                f"({corr_value:.2f})."
            )

        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title="Correlation Matrix"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "At least 2 numeric columns are required."
        )


# =========================================================
# TOP 10
# =========================================================

elif page == "🏆 Top 10":

    st.subheader("🏆 Top 10")

    if category_col and value_col:

        top10 = (
            df.groupby(category_col)[value_col]
            .sum()
            .reset_index()
            .sort_values(
                value_col,
                ascending=False
            )
            .head(10)
        )

        if not top10.empty:

            top_item = top10.iloc[0]

            st.info(
                f"Top 10 {category_col} based on "
                f"{value_col}."
            )

            st.success(
                f"🏆 Highest: "
                f"{top_item[category_col]} — "
                f"{top_item[value_col]:,.0f}"
            )

            fig = px.bar(
                top10,
                x=value_col,
                y=category_col,
                orientation="h",
                title=f"Top 10 {category_col} by {value_col}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning(
                "No data available."
            )

    else:

        st.warning(
            "Suitable columns were not found."
        )


# =========================================================
# DATA TABLE
# =========================================================

elif page == "📋 Data Table":

    st.subheader("📋 Data Table")

    st.info(
        f"Complete dataset with {len(df)} rows "
        f"and {len(df.columns)} columns."
    )

    st.dataframe(
        df,
        use_container_width=True,
        height=600
    )