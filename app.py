# =========================
# AI Powered Business Data Analytics Dashboard
# Ready-to-run Streamlit App
# =========================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import speech_recognition as sr

st.set_page_config(page_title="AI Data Analytics Tool", layout="wide")
st.title("🚀 AI Powered Business Data Analytics Dashboard")

# -------------------------
# 1️⃣ Dataset Upload
# -------------------------
file = st.file_uploader("Upload Dataset (CSV/XLSX)", type=["csv","xlsx"])

if file is not None:

    # --- Read file ---
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    # -------------------------
    # 2️⃣ Data Quality Check
    # -------------------------
    st.subheader("Data Quality Check")
    st.write("Rows:", len(df))
    st.write("Columns:", len(df.columns))
    st.write("Missing Values:", df.isna().sum().sum())
    st.write("Duplicates:", df.duplicated().sum())
    df_original = df.copy()     


    if st.button("🔍 Check Missing Position"):

        missing_positions = []

        for row_index, row in df_original.iterrows():
            for col in df_original.columns:
                if pd.isna(row[col]):
                    missing_positions.append(f"Row {row_index + 1} → {col} missing")

        if len(missing_positions) > 0:
            st.subheader("Missing Value Positions")
            for item in missing_positions:
                st.write(item)
        else:
            st.success("No Missing Values Found ✅")

    # Cleaning
    df = df.drop_duplicates()
    df = df.fillna(df.mean(numeric_only=True))

    # -------------------------
    # 3️⃣ KPI Detection
    # -------------------------
    # -------------------------
    # 3️⃣ KPI Detection (Fixed)
    # -------------------------
    num_cols = df.select_dtypes(include=np.number).columns

    filtered_cols = []

    for col in num_cols:
        col_lower = col.lower()

        # ❌ Remove Year
        if "year" in col_lower:
            continue

        # ❌ Remove ID type columns
        if any(word in col_lower for word in ["id", "code", "number"]):
            continue

        # ✅ Keep useful columns
        filtered_cols.append(col)

    # Select KPI
    if len(filtered_cols) > 0:
        kpi = filtered_cols[0]

        st.subheader("Detected KPI")
        st.write(kpi)

        # -------------------------
        # 4️⃣ Top Factor Detection
        # -------------------------
        st.subheader("Top Factor Detection")
        corr = df.corr(numeric_only=True)
        st.write(corr[kpi].sort_values(ascending=False))

        # Smart Numeric Column Selection
        # -------------------------
        num_cols = df.select_dtypes(include=np.number).columns

        useful_cols = []

        for col in num_cols:
            
            col_lower = col.lower()
            
            # ❌ Ignore ID type columns based on name
            if any(word in col_lower for word in ["id", "code", "number", "phone"]):
                continue
            
            # ❌ Ignore Year / time column
            if "year" in col_lower:
                continue
            
            # ✅ Keep remaining numeric columns (metrics)
            useful_cols.append(col)


        # -------------------------
        # Auto Insights (Cleaned)
        # -------------------------
        st.subheader("Auto Insights")

        for col in useful_cols:
            st.write(col, "Average:", round(df[col].mean(), 2))
            st.write(col, "Max:", df[col].max())
            st.write(col, "Min:", df[col].min())
            st.write(col, "Std Dev:", round(df[col].std(), 2))
        # -------------------------
        # 6️⃣ Trend Detection
        # -------------------------
        st.subheader("Trend Detection")
        if df[kpi].iloc[-1] > df[kpi].iloc[0]:
            st.success("Trend: Increasing")
        elif df[kpi].iloc[-1] < df[kpi].iloc[0]:
            st.error("Trend: Decreasing")
        else:
            st.info("Trend: Stable")

        # -------------------------
        # 7️⃣ Trend Chart
        # -------------------------
        st.subheader("Trend Chart")
        fig, ax = plt.subplots()
        df[kpi].plot(ax=ax, color='blue', marker='o', title=f"{kpi} Trend")
        st.pyplot(fig)


        # -------------------------
        # 9️⃣ Predictive Analytics
        # -------------------------
        st.subheader("Predictive Analytics")
        X = df[["Marketing_Spend", "Cost"]]
        y = df["Sales"]

            # Model training
        model = LinearRegression()
        model.fit(X, y)

            # Last known values
        last_marketing = df["Marketing_Spend"].iloc[-1]
        last_cost = df["Cost"].iloc[-1]

            # Prediction (same type of input as training)
        prediction = model.predict([[last_marketing, last_cost]])
        st.write("Next Predicted Value:", int(prediction[0]))
                   
            
        # -------------------------
       # 🔟 What-If & Scenario Analysis
        # -------------------------
        st.subheader("What-If & Scenario Analysis")

        if kpi.lower() != "year":

            # Current value
            current_value = df[kpi].iloc[-1]
            st.write(f"Current {kpi}:", int(current_value))

            # 🔹 What-If (User Controlled)
            st.write("### What-If Analysis")
            increase_kpi = st.slider("Adjust KPI %", -50, 50, 10)

            what_if_value = current_value * (1 + increase_kpi / 100)
            st.write(f"After {increase_kpi}% change:", int(what_if_value))

            # 🔹 Scenario Comparison (Fixed Cases)
            st.write("### Scenario Comparison")

            scenario1 = current_value * 1.05
            scenario2 = current_value * 1.10

            st.write("5% Increase Scenario:", int(scenario1))
            st.write("10% Increase Scenario:", int(scenario2))

        else:
            st.warning("Analysis not applicable on Year column")
        # -------------------------
        # 12️⃣ Business Recommendation
        # -------------------------
        st.subheader("Business Recommendation")
        if df[kpi].mean() < df[kpi].max()*0.5:
            st.warning("Recommendation: Improve performance related to "+kpi)
        else:
            st.success("Performance looks healthy")

        # -------------------------
        # 13️⃣ AI Chat with Data
        # -------------------------
        st.subheader("AI Chat with Data")
        question = st.text_input("Ask a question (e.g., Which year had the lowest Profit?)")

        if question:
            q = question.lower()
            # Average
            if "average" in q:
                st.write("Average of", kpi, ":", round(df[kpi].mean(),2))
            # Maximum / Highest
            elif "max" in q or "highest" in q:
                st.write("Highest", kpi, ":", df[kpi].max())
            # Minimum / Lowest
            elif "min" in q or "lowest" in q:
                if "year" in q and "Year" in df.columns:
                    idx = df[kpi].idxmin()
                    year_val = df.loc[idx, "Year"]
                    st.write(f"Year with lowest {kpi}:", year_val)
                elif "region" in q and "Region" in df.columns:
                    idx = df[kpi].idxmin()
                    region_val = df.loc[idx, "Region"]
                    st.write(f"Region with lowest {kpi}:", region_val)
                else:
                    st.write("Lowest", kpi, ":", df[kpi].min())
            else:
                st.write("Question not understood. Try asking about average, max, min, year, or region.")

        # -------------------------
        # 14️⃣ Voice Query
        # -------------------------
        st.subheader("Voice Query (Speak your question)")
        if st.button("Ask by Voice", key="voice_btn"):
            r = sr.Recognizer()
            with sr.Microphone() as source:
                st.write("Speak now...")
                audio = r.listen(source)

            try:
                text = r.recognize_google(audio)
                st.write("You said:", text)

                q = text.lower()

                # 🔥 Full Insights (trend / summary / analysis)
                if any(word in q for word in ["trend", "insight", "summary", "analysis"]):

                    st.write("📊 --- Business Insights Report ---")

                    # Average
                    st.write("Average Sales:", round(df["Sales"].mean(),2))

                    # Max & Min
                    st.write("Highest Sales:", df["Sales"].max())
                    st.write("Lowest Sales:", df["Sales"].min())

                    # Trend
                    if df["Sales"].iloc[-1] > df["Sales"].iloc[0]:
                        st.write("Trend: Increasing 📈")
                    else:
                        st.write("Trend: Decreasing 📉")

                    # Top Factor
                    corr = df.corr(numeric_only=True)
                    top_factor = corr["Sales"].sort_values(ascending=False).index[1]
                    st.write("Top Factor affecting Sales:", top_factor)

                    # Anomaly
                    mean = df["Sales"].mean()
                    std = df["Sales"].std()
                    anomalies = df[(df["Sales"] > mean + 2*std) | (df["Sales"] < mean - 2*std)]
                    st.write("Anomalies Found:", len(anomalies))

                    # Prediction
                    X = df[["Marketing_Spend", "Cost"]]
                    y = df["Sales"]

                    model = LinearRegression()
                    model.fit(X, y)

                    last_marketing = df["Marketing_Spend"].iloc[-1]
                    last_cost = df["Cost"].iloc[-1]

                    prediction = model.predict([[last_marketing, last_cost]])
                    st.write("Next Predicted Sales:", round(prediction[0],2))

                    # Recommendation
                    if df["Sales"].mean() < df["Sales"].max()*0.5:
                        st.write("Recommendation: Improve sales performance")
                    else:
                        st.write("Recommendation: Sales performance is good")

                # Average
                elif any(word in q for word in ["average", "mean", "avg"]):
                    st.write("Average Sales:", round(df["Sales"].mean(),2))

                # Max
                elif any(word in q for word in ["max", "highest"]):
                    st.write("Highest Sales:", df["Sales"].max())

                # Min
                elif any(word in q for word in ["min", "lowest"]):
                    st.write("Lowest Sales:", df["Sales"].min())

                else:
                    st.write("Try saying: trend, insights, average, max, min")

            except:
                st.write("Voice not recognized")

        # -------------------------
        # 15️⃣ Top & Bottom 5 Records
        # -------------------------
        st.subheader("Top & Bottom 5 Records by KPI")
        st.write("Top 5:", df.nlargest(5, kpi))
        st.write("Bottom 5:", df.nsmallest(5, kpi))

        # -------------------------
        # 16️⃣ Category Summary
        # -------------------------
        cat_cols = df.select_dtypes(exclude=np.number).columns
        if len(cat_cols) > 0:
            st.subheader("Category Summary")
            summary = df.groupby(cat_cols[0])[num_cols].mean().reset_index()
            st.write(summary)


        # -------------------------
        # 18️⃣ KPI Distribution Chart
        # -------------------------
        st.subheader("KPI Distribution")
        fig2, ax2 = plt.subplots()
        df[kpi].hist(ax=ax2, bins=10, color='green')
        st.pyplot(fig2)
