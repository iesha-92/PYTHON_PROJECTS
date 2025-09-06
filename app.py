# --- Imports
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path
import os
if os.path.exists(MODEL_PATH):
    #load model
else:
    st.error("Model file not found")

# ---------- Robust paths + diagnostics ----------
APP_DIR = Path(__file__).resolve().parent          # /.../App
ROOT    = APP_DIR.parent                           # repo root
MODELS_DIR   = ROOT / "Models"
DATASETS_DIR = ROOT / "Datasets"

# If your filenames differ, edit these names only:
MODEL_PATH   = MODELS_DIR / "logreg_model.pkl"
SCALER_PATH  = MODELS_DIR / "scaler.pkl"
COLUMNS_PATH = MODELS_DIR / "X_columns.pkl"
SAMPLE_CSV   = DATASETS_DIR / "Employee Records.csv"  # change if your CSV has a different name

# Minimal diagnostics shown in sidebar (very helpful on Streamlit Cloud)
with st.sidebar:
    st.caption("Debug Info")
    st.write("App dir:", str(APP_DIR))
    st.write("Repo root:", str(ROOT))
    st.write("Models dir exists:", MODELS_DIR.exists())
    st.write("Datasets dir exists:", DATASETS_DIR.exists())
    try:
        if MODELS_DIR.exists():
            st.write("Models files:", [p.name for p in MODELS_DIR.iterdir()])
        if DATASETS_DIR.exists():
            st.write("Datasets files:", [p.name for p in DATASETS_DIR.iterdir()])
    except Exception:
        pass

def _require(p: Path, label: str):
    if not p.exists():
        st.error(f"❌ Missing {label}: `{p}`. Check exact name and folder case.")
        st.stop()

@st.cache_resource
def load_artifacts():
    _require(MODEL_PATH, "model.pkl")
    _require(SCALER_PATH, "scaler.pkl")
    _require(COLUMNS_PATH, "X_columns.pkl")
    model   = joblib.load(MODEL_PATH)
    scaler  = joblib.load(SCALER_PATH)
    columns = joblib.load(COLUMNS_PATH)
    return model, scaler, columns

# --- Load model artifacts once
model, scaler, FEATURE_COLS = load_artifacts()

# --- Page config
st.set_page_config(
    page_title="HR Attrition Risk Suite — Aisha Mohammed",
    page_icon="🛡️",
    layout="wide"
)

# --- Minimal CSS polish
st.markdown("""
<style>
    .metric-card {padding:1rem;border-radius:12px;border:1px solid #262730;background:#0e1117;}
    .good {color:#22c55e;font-weight:600;}
    .warn {color:#f59e0b;font-weight:600;}
    .bad  {color:#ef4444;font-weight:600;}
    .section-title {font-size:1.3rem;font-weight:700;margin:0.5rem 0 0.25rem 0;}
    .subtle {color:#9aa0a6;font-size:0.9rem;}
</style>
""", unsafe_allow_html=True)

# --- Helpers
def preprocess(df_raw: pd.DataFrame, feature_cols) -> np.ndarray:
    """Align columns + one-hot encode to match training schema."""
    df = pd.get_dummies(df_raw, drop_first=False)
    # add any missing columns
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0
    # drop extra columns not in training
    df = df[feature_cols]
    # numeric matrix for scaler (trained on full feature matrix)
    X = df.values
    return X

def predict_single(payload: dict, model, scaler, feature_cols, threshold: float = 0.5):
    raw = pd.DataFrame([payload])
    X = preprocess(raw, feature_cols)
    Xs = scaler.transform(X)
    prob = float(model.predict_proba(Xs)[:, 1][0])
    label = "Likely to Exit" if prob >= threshold else "Likely to Stay"
    return prob, label

# --- Sidebar navigation
st.sidebar.title("HR Attrition")
st.sidebar.caption("Predict and analyze employee attrition risk with advanced analytics and insights.")
section = st.sidebar.radio("NAVIGATION", ["Overview", "Single Prediction", "Batch Scoring", "Department Insights"])
st.sidebar.markdown("—")
st.sidebar.caption("Built by **Aisha Mohammed**")

# ====== OVERVIEW =============================================================
if section == "Overview":
    st.title("HR Attrition Risk Suite")
    st.subheader("Leverage analytics to predict, analyze, and mitigate employee attrition risks across your organization.")

    # If you have a company-wide snapshot (optional), use CSV; else synthesize a small demo dataframe
    if SAMPLE_CSV.exists():
        df_org = pd.read_csv(SAMPLE_CSV)
    else:
        np.random.seed(7)
        depts = ["Sales","Engineering","Marketing","HR","Finance","Operations"]
        df_org = pd.DataFrame({
            "Department": np.random.choice(depts, 392, p=[.25,.28,.1,.1,.12,.15]),
            "Gender": np.random.choice(["Male","Female"], 392),
            "Age": np.random.randint(21, 60, 392),
            "Years of Service": np.random.randint(0, 12, 392),
            "Salary": np.random.randint(25000, 180000, 392)
        })

    # Score the org quickly (rough estimate)
    X  = preprocess(df_org, FEATURE_COLS)       # FIX: don't call get_dummies() here; preprocess handles it
    Xs = scaler.transform(X)
    probs = model.predict_proba(Xs)[:, 1]
    df_org["Exit_Probability"] = probs
    df_org["High_Risk"] = (probs >= 0.5).astype(int)

    total_emp = len(df_org)
    avg_prob  = df_org["Exit_Probability"].mean()
    high_risk = int(df_org["High_Risk"].sum())

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><div class="section-title">Total Employees</div>'
                    f'<div style="font-size:2rem;font-weight:800;">{total_emp:,}</div>'
                    '<div class="subtle">active across departments</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="section-title">Avg Exit Probability</div>'
                    f'<div style="font-size:2rem;font-weight:800;">{avg_prob*100:.1f}%</div>'
                    '<div class="subtle">organization average</div></div>', unsafe_allow_html=True)
    with col3:
        color = "bad" if total_emp and (high_risk/total_emp) >= 0.25 else "warn"
        st.markdown(f'<div class="metric-card"><div class="section-title">High-Risk Employees</div>'
                    f'<div style="font-size:2rem;font-weight:800;" class="{color}">{high_risk}</div>'
                    '<div class="subtle">≥ 50% exit probability</div></div>', unsafe_allow_html=True)

    st.markdown("### Key Sections")
    c1, c2, c3 = st.columns(3)
    c1.info("**Single Prediction** — analyze one employee’s risk.")
    c2.info("**Batch Scoring** — upload CSV and download scored results.")
    c3.info("**Department Insights** — visualize risk by department.")

# ====== SINGLE PREDICTION ====================================================
elif section == "Single Prediction":
    st.title("Single Employee Prediction")

    c1, c2, c3 = st.columns(3)
    with c1:
        dept = st.selectbox("Department", ["HR","Billing","Sales & Marketing","Data Analytics",
                                           "Operations","IT","Finance","Admin"])
        job_title = st.text_input("Job Title", "Data Analyst")
        gender = st.selectbox("Gender", ["Female","Male"])
    with c2:
        age = st.number_input("Age", 18, 75, 30)
        yos = st.number_input("Years of Service", 0, 40, 3)
        salary = st.number_input("Annual Salary ($)", 10000, 300000, 60000, step=1000)
    with c3:
        threshold = st.slider("Decision threshold", 0.10, 0.90, 0.50, 0.05)
        st.caption("Lower threshold = more sensitive; higher = more strict.")

    if st.button("Predict Attrition Risk", type="primary"):
        payload = {
            "Department": dept, "Job Title": job_title, "Gender": gender,
            "Age": age, "Years of Service": yos, "Salary": salary
        }
        prob, label = predict_single(payload, model, scaler, FEATURE_COLS, threshold)
        st.success("Prediction complete")
        k1, k2 = st.columns(2)
        k1.metric("Exit Probability", f"{prob*100:.1f}%")
        k2.metric("Prediction", label)

# ====== BATCH SCORING ========================================================
elif section == "Batch Scoring":
    st.title("Batch Scoring")
    st.caption("Upload a CSV with columns like Department, Job Title, Gender, Age, Years of Service, Salary.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    threshold = st.slider("Decision threshold", 0.10, 0.90, 0.50, 0.05, key="thresh_batch")

    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.write("Preview", df_raw.head())
        X  = preprocess(df_raw, FEATURE_COLS)    # FIX: preprocess handles dummies
        Xs = scaler.transform(X)
        probs = model.predict_proba(Xs)[:, 1]
        out = df_raw.copy()
        out["Exit_Probability"] = probs.round(4)
        out["Prediction"] = np.where(out["Exit_Probability"] >= threshold, "Likely to Exit", "Likely to Stay")
        st.success("Scoring complete")
        st.dataframe(out.head(20), use_container_width=True)
        st.download_button("Download scored CSV", out.to_csv(index=False).encode(), "scored_employees.csv", "text/csv")

# ====== DEPARTMENT INSIGHTS ==================================================
elif section == "Department Insights":
    st.title("Department Insights")
    st.caption("Analyze risk patterns and trends across departments.")

    if SAMPLE_CSV.exists():
        df_org = pd.read_csv(SAMPLE_CSV)
    else:
        np.random.seed(7)
        depts = ["Sales","Engineering","Marketing","HR","Finance","Operations"]
        df_org = pd.DataFrame({
            "Department": np.random.choice(depts, 392, p=[.25,.28,.1,.1,.12,.15]),
            "Gender": np.random.choice(["Male","Female"], 392),
            "Age": np.random.randint(21, 60, 392),
            "Years of Service": np.random.randint(0, 12, 392),
            "Salary": np.random.randint(25000, 180000, 392)
        })

    # Score
    X  = preprocess(df_org, FEATURE_COLS)        # FIX: preprocess handles dummies
    Xs = scaler.transform(X)
    probs = model.predict_proba(Xs)[:, 1]
    df_org["Exit_Probability"] = probs
    df_org["High_Risk"] = (probs >= 0.5).astype(int)

    # Bar: average exit probability by department
    dep = (df_org.groupby("Department", as_index=False)
                 .agg(Avg_Exit_Prob=("Exit_Probability", "mean"),
                      Employees=("Exit_Probability", "size"),
                      High_Risk=("High_Risk", "sum")))
    dep = dep.sort_values("Avg_Exit_Prob", ascending=False)

    fig = px.bar(dep, x="Department", y="Avg_Exit_Prob",
                 text="Avg_Exit_Prob", title="Average Exit Probability by Department",
                 labels={"Avg_Exit_Prob":"Avg Exit Probability"})
    fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    fig.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    # Quick recommendations (toy rules)
    st.markdown("### Key Insights & Recommendations")
    risky = dep.head(2)["Department"].tolist()
    safe  = dep.tail(1)["Department"].tolist()
    recs = [
        f"Focus retention efforts on **{risky[0]}** (highest risk)." if len(risky) > 0 else "",
        f"Investigate factors contributing to **{risky[1]}** turnover." if len(risky) > 1 else "",
        f"Leverage **{safe[0]}**’s low turnover best practices." if len(safe) > 0 else "",
        "Implement targeted interventions for high-risk employees."
    ]
    for r in recs:
        if r:
            st.warning(r)
