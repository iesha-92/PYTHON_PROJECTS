# app.py — HR Attrition Risk Suite (Streamlit)
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ---------- Paths ----------
# If app.py is in Predicting_Employee_Attrition/App/, project_dir is Predicting_Employee_Attrition
PROJECT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR  = PROJECT_DIR / "Models"

MODEL_PATH   = MODELS_DIR / "logreg_model.pkl"
SCALER_PATH  = MODELS_DIR / "scaler.pkl"
COLUMNS_PATH = MODELS_DIR / "X_columns.pkl"

st.set_page_config(page_title="HR Attrition Risk Suite — Aisha Mohammed", layout="wide")

# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists() or not COLUMNS_PATH.exists():
        raise FileNotFoundError(
            f"Could not load artifacts from {MODELS_DIR}. "
            f"Missing at least one of: {MODEL_PATH.name}, {SCALER_PATH.name}, {COLUMNS_PATH.name}"
        )
    model   = joblib.load(MODEL_PATH)
    scaler  = joblib.load(SCALER_PATH)
    columns = joblib.load(COLUMNS_PATH)
    return model, scaler, columns

model, scaler, FEATURE_COLS = load_artifacts()

# number columns we expect as numeric (adjust if yours differ)
NUMERIC_COLS = ["Age", "Years of Service", "Salary"]

# ---------- Preprocess ----------
def preprocess_df(df_raw: pd.DataFrame) -> np.ndarray:
    df = df_raw.copy()

    # Coerce numeric columns
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # One-hot encode categoricals
    df = pd.get_dummies(df)

    # Add any missing expected columns, in case the upload didn’t contain all levels
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0

    # Make sure the order matches training
    df = df[FEATURE_COLS]

    # Scale
    X = scaler.transform(df.values)
    return X

def predict_row(payload: dict, threshold: float = 0.50):
    raw = pd.DataFrame([payload])
    X   = preprocess_df(raw)
    prob = float(model.predict_proba(X)[:, 1])  # probability of "Exit"
    label = "Likely to Exit" if prob >= threshold else "Likely to Stay"
    return prob, label

# ---------- UI ----------
with st.sidebar:
    st.header("Navigation")
    section = st.radio(
        "Go to",
        ["Overview", "Single Prediction", "Batch Scoring", "Department Insights"],
        key="nav",
    )
    st.caption("Built by Aisha Mohammed")

if section == "Overview":
    st.title("HR Attrition Risk Suite")
    st.write(
        "Predict exit probability, score CSVs, and visualize risk by department. "
        "Model: **Logistic Regression** • Frameworks: **scikit-learn**, **Streamlit**, **Plotly**."
    )
    st.success("Artifacts loaded successfully.")
    st.markdown(
        """
        **Tabs**  
        – Single Prediction (manual entry)  
        – Batch Scoring (CSV upload & download)  
        – Department Insights (average risk by department)
        """
    )

elif section == "Single Prediction":
    st.title("Single Prediction")
    c1, c2, c3 = st.columns(3)

    with c1:
        dept = st.selectbox("Department", ["HR","Billing","Sales & Marketing","Data Analytics","Operations","IT","Finance","Admin"])
        job_title = st.text_input("Job Title", "Data Analyst")
        gender = st.selectbox("Gender", ["Female","Male"])
    with c2:
        age = st.number_input("Age", 18, 75, 30)
        years = st.number_input("Years of Service", 0, 50, 2)
        salary = st.number_input("Salary", 0, 300000, 50000, step=1000)
    with c3:
        threshold = st.slider("Decision threshold", 0.10, 0.90, 0.50, 0.05)
        st.caption("Lower threshold → more sensitive; higher → more precise.")

    if st.button("Predict"):
        payload = {
            "Department": dept, "Job Title": job_title, "Gender": gender,
            "Age": age, "Years of Service": years, "Salary": salary
        }
        prob, label = predict_row(payload, threshold)
        k1, k2 = st.columns(2)
        k1.metric("Exit Probability", f"{prob:.2%}")
        k2.metric("Prediction", label)

elif section == "Batch Scoring":
    st.title("Batch Scoring")
    st.caption('Upload a CSV with columns **Department, Job Title, Gender, Age, Years of Service, Salary**.')

    g = st.file_uploader("Upload CSV", type=["csv"], key="batch")
    threshold = st.slider("Decision threshold", 0.10, 0.90, 0.50, 0.05, key="thresh_batch")

    if g is not None:
        df_raw = pd.read_csv(g)
        st.write("Preview", df_raw.head())

        X = preprocess_df(df_raw)
        probs = model.predict_proba(X)[:, 1]
        out = pd.DataFrame({"Exit_Probability": probs.round(4)})
        out["Prediction"] = np.where(out["Exit_Probability"] >= threshold, "Likely to Exit", "Likely to Stay")

        st.success("Scoring complete")
        st.dataframe(out.head(20), use_container_width=True)
        st.download_button("Download scored CSV", out.to_csv(index=False).encode(), "scored_employees.csv", "text/csv")

elif section == "Department Insights":
    st.title("Department Insights")
    st.caption("Upload the scored CSV from Batch Scoring to see department-level risk.")

    g = st.file_uploader("Upload scored CSV", type=["csv"], key="insights")
    if g is not None:
        df = pd.read_csv(g)
        if "Department" in df.columns and "Exit_Probability" in df.columns:
            agg = df.groupby("Department", as_index=False)["Exit_Probability"].mean().sort_values("Exit_Probability", ascending=False)
            fig = px.bar(agg, x="Exit_Probability", y="Department", orientation="h",
                         title="Average Exit Probability by Department", text=agg["Exit_Probability"].round(2))
            fig.update_traces(texttemplate="%{text}", textposition="outside")
            fig.update_layout(xaxis_title="Avg Probability", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("File must include `Department` and `Exit_Probability`.")
