import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import random
import time
import smtplib

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from database.connection import supabase

import os
from google import genai

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
# DATABASE LOGIN SYSTEM
# EMAIL OTP LOGIN SYSTEM
# -------------------------

def send_email_otp(receiver_email, otp):

    sender_email = st.secrets["GMAIL_EMAIL"]
    app_password = st.secrets["GMAIL_APP_PASSWORD"]

    subject = "AI Analytics System - OTP"

    body = f"""
Hello,
I am Yash Gupta !
Your OTP for AI Analytics System is:

{otp}

This OTP is valid for 5 minutes.

Do not share this OTP with anyone.

Regards,
AI Analytics System
"""

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    message.attach(
        MIMEText(body, "plain")
    )

    with smtplib.SMTP("smtp.gmail.com", 587) as server:

        server.starttls()

        server.login(
            sender_email,
            app_password
        )

        server.sendmail(
            sender_email,
            receiver_email,
            message.as_string()
        )


# -------------------------
# SESSION STATE
# -------------------------

if "logged_in" not in st.session_state:

    st.session_state.logged_in = False
    st.session_state.otp = None
    st.session_state.otp_time = None
    st.session_state.user = None
    st.session_state.login_email = None


# -------------------------
# LOGIN
# -------------------------

if not st.session_state.logged_in:

    st.title("🔐 Student Login")

    email = st.text_input(
        "Enter your email:",
        placeholder="example@gmail.com"
    )

    if st.button("Send OTP"):

        if not email:

            st.error("Please enter your email.")

        else:

            try:

                email = email.strip().lower()

                # Generate 6-digit OTP
                otp = str(
                    random.randint(100000, 999999)
                )

                # Save OTP
                st.session_state.otp = otp
                st.session_state.otp_time = time.time()
                st.session_state.login_email = email

                # Send REAL email OTP
                send_email_otp(
                    email,
                    otp
                )

                st.success(
                    "OTP sent successfully 📧"
                )

            except Exception as e:

                st.error(
                    f"Failed to send OTP: {e}"
                )


    # OTP input
    otp_input = st.text_input(
        "Enter OTP:",
        max_chars=6,
        type="password"
    )


    if st.button("Verify OTP"):

        if st.session_state.otp is None:

            st.error(
                "Please request OTP first."
            )

        else:

            elapsed = (
                time.time()
                - st.session_state.otp_time
            )

            # OTP expiry = 5 minutes
            if elapsed > 300:

                st.error(
                    "OTP expired ❌ Please request a new OTP."
                )

                st.session_state.otp = None

            elif otp_input == st.session_state.otp:

                st.session_state.logged_in = True

                # Create temporary user
                st.session_state.user = {
                    "Name": "User",
                    "Email": st.session_state.login_email,
                    "Mobile": "",
                    "Role": "student"
                }

                st.success(
                    "Login successful ✅"
                )

                st.rerun()

            else:

                st.error(
                    "Invalid OTP ❌"
                )


# -------------------------
# LOGGED-IN USER
# -------------------------
else:

    user = st.session_state.user

    st.sidebar.success(
        f"Welcome, {user.get('Name', 'User')} 👋"
    )

    st.sidebar.write(
        f"Email: {user.get('Email', '')}"
    )

    st.sidebar.write(
        f"Role: {user.get('Role', 'student')}"
    )

    if st.sidebar.button("Logout", key="logout_button"):

        st.session_state.logged_in = False
        st.session_state.otp = None
        st.session_state.otp_time = None
        st.session_state.user = None
        st.session_state.login_email = None

        st.rerun()

    # -------------------------
    # YOUR EXISTING ANALYTICS CODE
    # -------------------------

    st.title("🚀 AI Smart Business Analytics System")
    # -------------------------
    # Dataset Selection
    # -------------------------

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
        
    st.session_state["analytics_df"] = df.copy()
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
