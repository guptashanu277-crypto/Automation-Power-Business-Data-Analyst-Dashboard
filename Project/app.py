import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random, time

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import os
from google import genai   # latest package

# 👇 Must be first Streamlit command
st.set_page_config(page_title="AI Analytics System", layout="wide")

# -------------------------
# Demo Dataset Function
# -------------------------
def get_demo_data():
    data = {
        "Year": [2023, 2023, 2023, 2024, 2024],
        "Month": ["Jan","Feb","Mar","Jan","Feb"],
        "Sales": [12000,15000,18000,20000,22000],
        "Profit": [3000,4000,5000,6000,7000],
        "Category": ["Electronics","Clothing","Electronics","Furniture","Clothing"]
    }
    return pd.DataFrame(data)

# -------------------------
# OTP Login System
# -------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.otp = None
    st.session_state.otp_time = None
    st.session_state.role = "user"

if not st.session_state.logged_in:
    st.title("🔐 Secure Login")
    mobile = st.text_input("Enter your Mobile Number:")

    if st.button("Send OTP"):
        st.session_state.otp = str(random.randint(1000, 9999))
        st.session_state.otp_time = time.time()
        st.info(f"Demo OTP (for testing): {st.session_state.otp}")

    otp_input = st.text_input("Enter OTP:")

    if st.button("Verify OTP"):
        if st.session_state.otp is None:
            st.error("Please request OTP first.")
        else:
            elapsed = time.time() - st.session_state.otp_time
            if elapsed > 30:
                st.error("OTP expired ❌ Please request a new one.")
            elif otp_input == st.session_state.otp:
                st.session_state.logged_in = True
                st.session_state.role = "user"
                st.success("Login successful ✅")
            else:
                st.error("Invalid OTP ❌")

else:
    # -------------------------
    # Fake Payment for Admin Unlock
    # -------------------------
    if st.session_state.role == "user":
        st.subheader("💳 Upgrade to Admin Access")
        payment_code = st.text_input("Enter Payment Code:")
        if st.button("Submit Payment Code"):
            if payment_code == "0156":
                st.session_state.role = "admin"
                st.success("Payment successful ✅ Admin Access unlocked!")
            else:
                st.error("Invalid Payment Code ❌")

    # -------------------------
    # Dataset Selection
    # -------------------------
    st.title("🚀 AI Smart Business Analytics System")

    dataset_choice = st.radio("Select Dataset:", ["Demo Dataset", "Upload Your Own"])
    if dataset_choice == "Upload Your Own":
        file = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx"])
        if file:
            df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
        else:
            st.warning("Please upload a dataset to continue.")
            st.stop()
    else:
        df = get_demo_data()

    # -------------------------
    # AI Assistant Section
    # -------------------------
    st.subheader("🤖 AI Business Assistant")
    question = st.text_input("Ask Anything About Your Business Data...")
    ask_ai = st.button("Ask AI")

    if ask_ai and question:
        with st.spinner("AI Analyzing..."):
            client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
            summary = f"""
Columns: {df.columns.tolist()}

First Rows:
{df.head(10).to_string()}

Statistics:
{df.describe(include='all').to_string()}
"""
            prompt = f"""
You are a Smart AI Business Analyst.
Project Name: AI Smart Business Analytics System
Founder: Yashvant Gupta
Rules:
- Default answers must be in English or Hinglish only.
- Do NOT use pure Hindi unless the user question itself is written in Hindi.
- Never use 'Namaskar' or similar greetings. Use 'Welcome' instead.
- Focus only on business insights, strategies, and analysis.
Dataset:
{summary}
Question:
{question}
"""
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=prompt
                )
                st.subheader("💡 AI Answer")
                st.success(response.text)
            except Exception:
                st.warning("AI quota exceeded. Showing analytics only.")

    # -------------------------
    # Data Quality
    # -------------------------
    st.subheader("🧹 Data Quality")
    st.write("Rows:", len(df))
    st.write("Missing:", df.isna().sum().sum())
    st.write("Duplicates:", df.duplicated().sum())

    if st.button("🔍 Show Missing Value Locations"):
        missing_cols = df.isna().sum()
        if missing_cols.sum() > 0:
            st.write("Missing values per column:")
            st.write(missing_cols[missing_cols > 0])
            st.write("Row indices with missing values:")
            for col in df.columns:
                rows = df[df[col].isna()].index.tolist()
                if rows:
                    st.write(f"{col}: Missing at rows {rows}")
        else:
            st.success("No missing values found.")

    df = df.drop_duplicates()
    df = df.fillna(df.mean(numeric_only=True))

    # -------------------------
    # Useful Columns
    # -------------------------
    def get_useful_columns(df):
        cols = df.select_dtypes(include=np.number).columns
        useful = []
        for col in cols:
            c = col.lower()
            if "year" in c: continue
            if any(x in c for x in ["id","code","number","phone"]): continue
            useful.append(col)
        return useful

    useful_cols = get_useful_columns(df)
    if len(useful_cols) == 0:
        st.error("No valid numeric columns")
        st.stop()

    kpi = st.selectbox("Select KPI", useful_cols)

    # Correlation
    st.subheader("📌 Correlation")
    corr = df[useful_cols].corr()
    st.write(corr[kpi].sort_values(ascending=False))

    # Insights
    st.subheader("📊 Auto Insights")
    for col in useful_cols:
        st.write(f"🔹 {col}")
        st.write("Mean:", round(df[col].mean(), 2))
        st.write("Max:", df[col].max())
        st.write("Min:", df[col].min())
        st.write("Std:", round(df[col].std(), 2))
        st.write("---")

    # Trend
    st.subheader("📈 Trend")
    growth = ((df[kpi].iloc[-1] - df[kpi].iloc[0]) / df[kpi].iloc[0]) * 100
    if growth > 0:
        st.success("Increasing 📈")
    else:
        st.error("Decreasing 📉")
    st.line_chart(df[kpi])

    # Prediction
    st.subheader("🤖 Prediction")
    features = st.multiselect("Select Features", useful_cols)
    target = st.selectbox("Select Target", useful_cols)

    if len(features) > 0 and target and target not in features:
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor()
        }
        best_model = None
        best_score = -1
        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            score = r2_score(y_test, pred)
            st.write(name, round(score, 2))
            if score > best_score:
                best_score = score
                best_model = model
        st.success(f"Best Model: {type(best_model).__name__}")
        prediction = best_model.predict(X.iloc[[-1]])
        st.write("Prediction:", int(prediction[0]))

    # What-If
    st.subheader("🔮 What-If")
    current = df[kpi].iloc[-1]
    change = st.slider("Change %", -50, 50, 10)
    st.write("Result:", int(current * (1 + change/100)))

    # Top/Bottom
    st.subheader("🏆 Top/Bottom")
    st.write(df.nlargest(5, kpi))
    st.write(df.nsmallest(5, kpi))

    # Download
    st.download_button(
        label="Download",
        data=df.to_csv(index=False),
        file_name="data.csv",
        mime="text/csv"
    )
