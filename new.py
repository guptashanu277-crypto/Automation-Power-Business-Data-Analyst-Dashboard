import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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
import io
import textwrap
from datetime import datetime
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

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:

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

                supabase.table("Login_History").insert({
               "email": st.session_state.login_email
                }).execute()

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
    # ADAPTIVE DATA INTELLIGENCE + SAFE DATA PREPARATION
    # -------------------------

    def quality_metrics(data):
        """Transparent score based on completeness and duplicate-free records."""
        if data.empty or len(data.columns) == 0:
            return {"completeness": 0.0, "duplicate_free": 0.0,
                    "score": 0.0, "missing_cells": 0, "duplicate_rows": 0,
                    "total_cells": 0, "total_rows": len(data)}

        total_cells = max(data.shape[0] * data.shape[1], 1)
        missing_cells = int(data.isna().sum().sum())
        duplicate_rows = int(data.duplicated().sum())
        completeness = 100 * (1 - missing_cells / total_cells)
        duplicate_free = 100 * (1 - duplicate_rows / max(len(data), 1))
        # Do not round an imperfect component up to a perfect 100.00%.
        if missing_cells > 0:
            completeness = min(completeness, 99.99)
        if duplicate_rows > 0:
            duplicate_free = min(duplicate_free, 99.99)

        # 70% completeness + 30% duplicate-free records.
        score = 0.70 * completeness + 0.30 * duplicate_free
        # Never display 100 while missing cells or duplicate rows still exist.
        if missing_cells > 0 or duplicate_rows > 0:
            score = min(score, 99.99)

        return {"completeness": round(completeness, 2),
                "duplicate_free": round(duplicate_free, 2),
                "score": round(max(0, min(100, score)), 2),
                "missing_cells": missing_cells,
                "duplicate_rows": duplicate_rows,
                "total_cells": total_cells,
                "total_rows": len(data)}

    def quality_score(data):
        return quality_metrics(data)["score"]

    def imputable_numeric_columns(data):
        # IDs/codes/phone/postal fields are identifiers, not measurements to average.
        return [c for c in data.select_dtypes(include=np.number).columns
                if not any(token in str(c).lower()
                           for token in ["id", "code", "phone", "zip", "postal", "account", "number"])]

    def detect_date_column(data):
        for col in data.columns:
            name = str(col).lower()
            if any(word in name for word in ["date", "time", "timestamp", "month"]):
                parsed = pd.to_datetime(data[col], errors="coerce")
                if parsed.notna().sum() >= max(3, int(len(data) * 0.5)):
                    return col
        return None

    def find_measure_columns(data):
        numeric = data.select_dtypes(include=np.number).columns.tolist()
        return [c for c in numeric if not any(x in str(c).lower()
                for x in ["id", "code", "phone", "zip", "postal", "year"])]

    def find_dimension_columns(data):
        return [c for c in data.columns if
                (data[c].dtype == "object" or str(data[c].dtype) == "category" or
                 data[c].dtype == "bool") and data[c].nunique(dropna=True) > 1]

    def choose_column(columns, keywords):
        for col in columns:
            if any(word in str(col).lower() for word in keywords):
                return col
        return None

    def make_pdf_report(report_text):
        buffer = io.BytesIO()
        with PdfPages(buffer) as pdf:
            lines = report_text.splitlines()
            page_lines = []
            for line in lines:
                wrapped = textwrap.wrap(line, width=92) or [""]
                page_lines.extend(wrapped)
            chunks = [page_lines[i:i + 43] for i in range(0, len(page_lines), 43)] or [[]]
            for page_no, chunk in enumerate(chunks, start=1):
                fig = plt.figure(figsize=(8.27, 11.69))
                fig.patch.set_facecolor("white")
                fig.text(0.08, 0.95, "AI SMART BUSINESS ANALYTICS SYSTEM",
                         fontsize=12, weight="bold", va="top")
                fig.text(0.08, 0.91, f"Dataset Intelligence Report | Page {page_no}",
                         fontsize=9, color="dimgray", va="top")
                y = 0.86
                for line in chunk:
                    fig.text(0.08, y, line, fontsize=8.5, va="top", family="monospace")
                    y -= 0.018
                fig.text(0.08, 0.035, f"Generated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}",
                         fontsize=8, color="gray")
                plt.axis("off")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
        buffer.seek(0)
        return buffer.getvalue()

    # Keep the uploaded/demo dataset unchanged until the user accepts cleaning.
    dataset_fingerprint = str((df.shape, list(df.columns), df.head(3).astype(str).to_dict()))
    if st.session_state.get("_dataset_fingerprint") != dataset_fingerprint:
        st.session_state["_dataset_fingerprint"] = dataset_fingerprint
        st.session_state["cleaned_df"] = df.copy()
        st.session_state["cleaning_log"] = ["Original dataset loaded. No changes applied."]
        st.session_state.pop("cleaning_preview_df", None)
        st.session_state.pop("cleaning_preview_log", None)
    raw_df = df.copy()
    active_df = st.session_state["cleaned_df"].copy()

    st.subheader("📌 Executive Dataset Intelligence")
    q_metrics = quality_metrics(active_df)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{len(active_df):,}")
    c2.metric("Columns", f"{len(active_df.columns):,}")
    c3.metric("Missing Cells", f"{q_metrics['missing_cells']:,}")
    c4.metric("Data Completeness", f"{q_metrics['completeness']:.2f}%")
    c5.metric("Overall Quality Score", f"{q_metrics['score']:.2f}/100")

    numeric_cols = find_measure_columns(active_df)
    dimension_cols = find_dimension_columns(active_df)
    date_col = detect_date_column(active_df)

    st.markdown("#### Dataset profile")
    profile = pd.DataFrame({
        "Column": active_df.columns.astype(str),
        "Data Type": [str(active_df[c].dtype) for c in active_df.columns],
        "Missing": [int(active_df[c].isna().sum()) for c in active_df.columns],
        "Unique Values": [int(active_df[c].nunique(dropna=True)) for c in active_df.columns]
    })
    st.dataframe(profile, use_container_width=True, hide_index=True)

    st.markdown("#### Data quality inspection")
    missing_by_col = active_df.isna().sum()
    duplicate_count = int(active_df.duplicated().sum())
    current_quality = quality_metrics(active_df)
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Missing cells", int(missing_by_col.sum()))
    q2.metric("Duplicate rows", duplicate_count)
    q3.metric("Data completeness", f"{current_quality['completeness']:.2f}%")
    q4.metric("Overall quality", f"{current_quality['score']:.2f}/100")
    st.caption("Score formula: 70% Data Completeness + 30% Duplicate-free Records. Constant columns are not penalized because a column containing one repeated value can still be valid business data. Any remaining missing cell or duplicate row prevents a 100/100 score.")
    if missing_by_col.sum():
        st.write("Missing values by column:")
        st.dataframe(missing_by_col[missing_by_col > 0].rename("Missing Values"),
                     use_container_width=True)
        with st.expander("Show exact row locations of missing values"):
            for col in active_df.columns:
                missing_rows = active_df.index[active_df[col].isna()].tolist()
                if missing_rows:
                    st.write(f"**{col}** — row indexes: {missing_rows[:500]}")
                    if len(missing_rows) > 500:
                        st.caption(f"Showing first 500 of {len(missing_rows)} rows.")
    else:
        st.success("No missing values found.")

    # -------------------------
    # SAFE CLEANING ENGINE
    # -------------------------
    st.subheader("🧹 Data Preparation & Cleaning Engine")
    st.caption("Choose the exact column and method. Preview the impact before applying changes; the original upload stays unchanged.")

    numeric_missing_cols = [c for c in imputable_numeric_columns(active_df)
                            if active_df[c].isna().any()]
    categorical_missing_cols = [c for c in active_df.columns
                                if (active_df[c].dtype == "object" or
                                    str(active_df[c].dtype) == "category" or
                                    active_df[c].dtype == "bool") and active_df[c].isna().any()]

    with st.expander("Mean vs Median — what do these options mean?"):
        st.write("**Mean:** fills blank numeric cells with that column's average. Use when values are reasonably balanced and do not contain extreme outliers.")
        st.write("**Median:** fills blank numeric cells with the middle value of that column. It is usually safer for skewed data or columns with extreme values, such as salary or transaction amount.")
        st.write("**Most frequent value (Mode):** fills a blank category with the value that appears most often, such as the most common city or product category.")
        st.write("**Unknown:** explicitly labels the missing category as Unknown instead of guessing its value.")
        st.warning("Mean, Median and Mode are estimates. They do not recover the original unknown value. Never use Mean/Median for IDs, phone numbers, postal codes or product codes.")

    if not numeric_missing_cols and not categorical_missing_cols and int(active_df.duplicated().sum()) == 0:
        st.success("No missing values or exact duplicate rows detected. No cleaning is currently required.")

    numeric_col = None
    numeric_method = "Do not fill"
    if numeric_missing_cols:
        st.markdown("**1. Fill missing numeric values**")
        numeric_col = st.selectbox("Select the numeric column to clean", numeric_missing_cols,
                                   key="clean_numeric_column")
        st.caption(f"{int(active_df[numeric_col].isna().sum())} missing value(s) in **{numeric_col}**.")
        numeric_method = st.selectbox("Choose how to fill this column",
                                      ["Do not fill", "Median", "Mean", "Zero"],
                                      key="clean_numeric_method")
    else:
        st.caption("No eligible numeric columns currently contain missing values.")

    category_col = None
    category_method = "Do not fill"
    if categorical_missing_cols:
        st.markdown("**2. Fill missing category/text values**")
        category_col = st.selectbox("Select the category/text column to clean",
                                    categorical_missing_cols, key="clean_category_column")
        st.caption(f"{int(active_df[category_col].isna().sum())} missing value(s) in **{category_col}**.")
        category_method = st.selectbox("Choose how to fill this column",
                                       ["Do not fill", "Most frequent value (Mode)", "Unknown"],
                                       key="clean_category_method")
    else:
        st.caption("No category/text columns currently contain missing values.")

    remove_duplicates = st.checkbox("Remove exact duplicate rows", value=False,
                                    help="Removes rows only when every column value matches another row exactly.")
    st.caption("Outliers are not deleted automatically. Review them separately before deciding whether they are errors.")

    if st.button("🔎 Preview cleaning", key="preview_cleaning_button"):
        candidate = active_df.copy()
        changes = []

        if remove_duplicates:
            before = len(candidate)
            candidate = candidate.drop_duplicates()
            removed = before - len(candidate)
            changes.append(f"Removed {removed} exact duplicate row(s).")

        if numeric_col and numeric_method != "Do not fill":
            count = int(candidate[numeric_col].isna().sum())
            if count:
                if numeric_method == "Median":
                    fill_value = candidate[numeric_col].median()
                elif numeric_method == "Mean":
                    fill_value = candidate[numeric_col].mean()
                else:
                    fill_value = 0
                if pd.notna(fill_value):
                    candidate[numeric_col] = candidate[numeric_col].fillna(fill_value)
                    changes.append(f"Filled {count} missing value(s) in '{numeric_col}' with {numeric_method} ({fill_value}). These are estimated values.")
                else:
                    changes.append(f"Could not fill '{numeric_col}': no valid numeric values are available to calculate {numeric_method}.")

        if category_col and category_method != "Do not fill":
            count = int(candidate[category_col].isna().sum())
            if count:
                if category_method == "Unknown":
                    fill_value = "Unknown"
                else:
                    mode_values = candidate[category_col].mode(dropna=True)
                    fill_value = mode_values.iloc[0] if not mode_values.empty else None
                if fill_value is not None:
                    candidate[category_col] = candidate[category_col].fillna(fill_value)
                    changes.append(f"Filled {count} missing value(s) in '{category_col}' with {category_method} ('{fill_value}').")
                else:
                    changes.append(f"Could not fill '{category_col}': no existing category is available to calculate Mode.")

        st.session_state["cleaning_preview_df"] = candidate
        st.session_state["cleaning_preview_log"] = changes or ["No changes were made. Choose a cleaning action first."]

    if "cleaning_preview_df" in st.session_state:
        preview_df = st.session_state["cleaning_preview_df"]
        st.markdown("#### Cleaning impact — before vs preview")
        before_q = quality_metrics(active_df)
        after_q = quality_metrics(preview_df)
        before_col, after_col = st.columns(2)
        before_col.metric("Before quality score", f"{before_q['score']:.2f}/100")
        before_col.metric("Before missing cells", f"{before_q['missing_cells']:,}")
        before_col.metric("Before duplicate rows", f"{before_q['duplicate_rows']:,}")
        after_col.metric("Preview quality score", f"{after_q['score']:.2f}/100",
                         f"{after_q['score'] - before_q['score']:+.2f} points")
        after_col.metric("Preview missing cells", f"{after_q['missing_cells']:,}",
                         f"{after_q['missing_cells'] - before_q['missing_cells']:+,} cells")
        after_col.metric("Preview duplicate rows", f"{after_q['duplicate_rows']:,}",
                         f"{after_q['duplicate_rows'] - before_q['duplicate_rows']:+,} rows")
        st.write("**Actions that will be applied:**")
        for item in st.session_state.get("cleaning_preview_log", []):
            st.write("•", item)
        st.dataframe(preview_df.head(20), use_container_width=True)

        if after_q["score"] > before_q["score"]:
            st.success("The preview improves the measured data quality score. Review the changes before applying.")
        elif after_q["score"] == before_q["score"]:
            st.info("The measured score has not changed. The selected action may not address the remaining quality issues.")
        else:
            st.warning("The preview lowers the measured score. Do not apply unless you understand why.")

        approve_col, cancel_col, reset_col = st.columns(3)
        if approve_col.button("✅ Apply approved cleaning", type="primary"):
            st.session_state["cleaned_df"] = preview_df.copy()
            st.session_state["analytics_df"] = preview_df.copy()
            st.session_state["cleaning_log"] = st.session_state.get("cleaning_preview_log", [])
            st.session_state.pop("cleaning_preview_df", None)
            st.session_state.pop("cleaning_preview_log", None)
            st.success("Cleaning applied. The dashboard now uses the approved dataset.")
            st.rerun()
        if cancel_col.button("✖ Cancel preview"):
            st.session_state.pop("cleaning_preview_df", None)
            st.session_state.pop("cleaning_preview_log", None)
            st.rerun()
        if reset_col.button("↩️ Reset to original dataset"):
            st.session_state["cleaned_df"] = raw_df.copy()
            st.session_state["analytics_df"] = raw_df.copy()
            st.session_state["cleaning_log"] = ["Dataset reset to original uploaded/demo data."]
            st.session_state.pop("cleaning_preview_df", None)
            st.session_state.pop("cleaning_preview_log", None)
            st.rerun()

    active_df = st.session_state["cleaned_df"].copy()
    st.session_state["analytics_df"] = active_df.copy()

    if st.session_state.get("cleaning_log"):
        with st.expander("Cleaning audit log"):
            for log_item in st.session_state["cleaning_log"]:
                st.write("•", log_item)

    # -------------------------
    # ADAPTIVE BUSINESS INSIGHTS
    # -------------------------
    st.subheader("📊 Automatic Dataset Insights")
    if active_df.empty:
        st.warning("The dataset has no rows to analyze.")
        st.stop()

    if numeric_cols:
        for col in numeric_cols:
            series = active_df[col].dropna()
            if not series.empty:
                st.write(f"**{col}** — Total: {series.sum():,.2f} | Average: {series.mean():,.2f} | "
                         f"Maximum: {series.max():,.2f} | Minimum: {series.min():,.2f}")
    else:
        st.info("No numeric measure columns were detected. Numeric summaries are not available.")

    # Most/least frequent values for categorical dimensions.
    if dimension_cols:
        st.markdown("#### Most and least common categories")
        selected_dimension = st.selectbox("Choose a category/dimension", dimension_cols)
        counts = active_df[selected_dimension].value_counts(dropna=False)
        if not counts.empty:
            common_col, rare_col = st.columns(2)
            common_col.metric("Most common", str(counts.index[0])[:45], f"{int(counts.iloc[0])} rows")
            rare_col.metric("Least common", str(counts.index[-1])[:45], f"{int(counts.iloc[-1])} rows")
            st.dataframe(counts.rename("Record Count").head(10), use_container_width=True)

    # Profitability: only show when an actual profit-like measure and product-like dimension exist.
    profit_col = choose_column(numeric_cols, ["profit", "net_income", "net profit", "earnings", "margin"])
    product_col = choose_column(active_df.columns, ["product", "item", "sku"])
    if profit_col and product_col:
        st.markdown("#### Most and least profitable products")
        product_profit = active_df.groupby(product_col, dropna=False)[profit_col].sum().sort_values(ascending=False)
        if not product_profit.empty:
            p1, p2 = st.columns(2)
            p1.metric("Most profitable", str(product_profit.index[0])[:45], f"{product_profit.iloc[0]:,.2f} {profit_col}")
            p2.metric("Least profitable", str(product_profit.index[-1])[:45], f"{product_profit.iloc[-1]:,.2f} {profit_col}")
            st.dataframe(product_profit.rename(f"Total {profit_col}").head(10), use_container_width=True)
    else:
        st.info("Product profitability is not shown because a product/item column and a profit-like numeric column were not both detected.")

    # Trend is date-based when a usable date field exists; never assume row order is time order.
    st.markdown("#### 📈 Trend detection")
    if date_col and numeric_cols:
        trend_measure = st.selectbox("Select measure for trend analysis", numeric_cols)
        trend_data = active_df[[date_col, trend_measure]].copy()
        trend_data[date_col] = pd.to_datetime(trend_data[date_col], errors="coerce")
        trend_data = trend_data.dropna().groupby(date_col)[trend_measure].sum().sort_index()
        if len(trend_data) >= 2:
            first_value, last_value = float(trend_data.iloc[0]), float(trend_data.iloc[-1])
            delta = last_value - first_value
            pct = (delta / abs(first_value) * 100) if first_value != 0 else None
            st.line_chart(trend_data)
            if delta > 0:
                st.success(f"{trend_measure} increased by {delta:,.2f} from first to last observed date.")
            elif delta < 0:
                st.warning(f"{trend_measure} decreased by {abs(delta):,.2f} from first to last observed date.")
            else:
                st.info(f"{trend_measure} was unchanged between the first and last observed dates.")
            if pct is not None:
                st.write(f"Percentage change: {pct:,.2f}%")
            st.caption("This is a descriptive change, not proof that any factor caused the trend.")
        else:
            st.info("Not enough valid date observations to calculate a trend.")
    elif numeric_cols:
        st.info("No reliable date/time column was detected. A date-based trend is not displayed.")
    else:
        st.info("Trend analysis needs at least one numeric measure.")

    # Outlier review using IQR; flags only, does not remove data.
    if numeric_cols:
        st.markdown("#### 🚩 Potential outlier review")
        outlier_col = st.selectbox("Choose numeric column to inspect for outliers", numeric_cols)
        values = active_df[outlier_col].dropna()
        if len(values) >= 4:
            q_low, q_high = values.quantile([0.25, 0.75])
            iqr = q_high - q_low
            lower_bound, upper_bound = q_low - 1.5 * iqr, q_high + 1.5 * iqr
            outliers = active_df[(active_df[outlier_col] < lower_bound) |
                                 (active_df[outlier_col] > upper_bound)]
            st.write(f"Potential outlier rows: {len(outliers)} (IQR rule)")
            if not outliers.empty:
                st.dataframe(outliers.head(100), use_container_width=True)
            st.caption("Outliers may be valid rare events, refunds, or data-entry errors. Review before changing them.")
        else:
            st.info("At least four valid numeric observations are needed for IQR review.")


    # -------------------------
    # PREDICTIVE ANALYSIS (numeric columns only)
    # -------------------------
    st.subheader("🤖 Predictive Analysis")
    if len(numeric_cols) >= 2 and len(active_df) >= 10:
        pred_features = st.multiselect(
            "Select numeric input features",
            [c for c in numeric_cols],
            default=[numeric_cols[0]],
            key="prediction_features"
        )
        pred_target = st.selectbox("Select prediction target", numeric_cols, key="prediction_target")
        if st.button("Train and compare prediction models"):
            valid_features = [c for c in pred_features if c != pred_target]
            model_data = active_df[valid_features + [pred_target]].replace([np.inf, -np.inf], np.nan).dropna()
            if not valid_features:
                st.warning("Select at least one feature different from the target.")
            elif len(model_data) < 10 or model_data[pred_target].nunique() < 2:
                st.warning("Not enough valid rows or target variation to train a reliable regression model.")
            else:
                X = model_data[valid_features]
                y = model_data[pred_target]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                model_options = {
                    "Linear Regression": LinearRegression(),
                    "Decision Tree": DecisionTreeRegressor(random_state=42),
                    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
                }
                scores = []
                for model_name, model in model_options.items():
                    try:
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                        score = r2_score(y_test, predictions)
                        scores.append((model_name, score, model))
                        st.write(f"{model_name}: R² = {score:.3f}")
                    except Exception as model_error:
                        st.warning(f"{model_name} could not be trained: {model_error}")
                if scores:
                    best_name, best_score, best_model = max(scores, key=lambda item: item[1])
                    st.success(f"Best test-set score: {best_name} (R² = {best_score:.3f})")
                    st.caption("R² is a test-set metric, not a guarantee of future accuracy. Small datasets can produce unstable scores.")
    else:
        st.info("Prediction requires at least two numeric measure columns and 10 dataset rows.")

    # -------------------------
    # WHAT-IF + TOP/BOTTOM
    # -------------------------
    if numeric_cols:
        st.subheader("🔮 What-if analysis")
        whatif_col = st.selectbox("Choose a measure", numeric_cols, key="whatif_measure")
        latest_value = active_df[whatif_col].dropna()
        if not latest_value.empty:
            change_percent = st.slider("Simulated change (%)", -50, 50, 10, key="whatif_change")
            simulated_value = float(latest_value.iloc[-1]) * (1 + change_percent / 100)
            st.metric("Simulated value based on last available row",
                      f"{simulated_value:,.2f}",
                      f"{change_percent:+d}%")
            st.caption("This is a simple scenario calculation, not a forecast.")
        st.subheader("🏆 Highest and lowest records")
        rank_col = st.selectbox("Rank records by", numeric_cols, key="ranking_measure")
        top_tab, bottom_tab = st.tabs(["Highest 5", "Lowest 5"])
        with top_tab:
            st.dataframe(active_df.nlargest(5, rank_col), use_container_width=True)
        with bottom_tab:
            st.dataframe(active_df.nsmallest(5, rank_col), use_container_width=True)

    # -------------------------
    # DOWNLOADS + REPORT
    # -------------------------
    st.subheader("📥 Downloads & Report")
    csv_data = active_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download current dataset (CSV)", csv_data,
                       "analytics_dataset.csv", "text/csv")
    excel_buffer = io.BytesIO()
    try:
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            active_df.to_excel(writer, index=False, sheet_name="Dataset")
            pd.DataFrame({"Cleaning Action": st.session_state.get("cleaning_log", [])}).to_excel(
                writer, index=False, sheet_name="Cleaning Audit")
        st.download_button("Download dataset + audit log (Excel)", excel_buffer.getvalue(),
                           "analytics_dataset_with_audit.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception:
        st.info("Excel export is unavailable in this environment; CSV export is still available.")

    report_lines = [
        "AI SMART BUSINESS ANALYTICS SYSTEM",
        "EXECUTIVE DATASET INTELLIGENCE REPORT",
        "",
        f"Dataset rows: {len(active_df)}",
        f"Dataset columns: {len(active_df.columns)}",
        f"Missing cells: {int(active_df.isna().sum().sum())}",
        f"Duplicate rows: {int(active_df.duplicated().sum())}",
        f"Data quality score: {quality_score(active_df)}/100",
        "",
        "COLUMN PROFILE:"
    ]
    for col in active_df.columns:
        report_lines.append(
            f"{col} | type={active_df[col].dtype} | missing={int(active_df[col].isna().sum())} "
            f"| unique={int(active_df[col].nunique(dropna=True))}"
        )
    report_lines.extend(["", "NUMERIC SUMMARIES:"])
    for col in numeric_cols:
        s = active_df[col].dropna()
        if not s.empty:
            report_lines.append(f"{col}: total={s.sum():.2f}, mean={s.mean():.2f}, min={s.min()}, max={s.max()}")
    report_lines.extend(["", "CLEANING AUDIT:"])
    report_lines.extend(st.session_state.get("cleaning_log", ["No cleaning actions recorded."]))
    report_lines.extend(["", "LIMITATION: Findings are descriptive and depend on the columns and data quality provided."])
    pdf_bytes = make_pdf_report("\n".join(report_lines))
    st.download_button("📄 Download Dataset Intelligence PDF Report", pdf_bytes,
                       "AI_Analytics_Report.pdf", "application/pdf")
