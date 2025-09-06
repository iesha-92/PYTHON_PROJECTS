# app.py
from pathlib import Path
import joblib, numpy as np, pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------- Page setup ----------
st.set_page_config(
    page_title="HR Attrition Risk Suite",
    page_icon="🧭",
    layout="wide"
)

# ---------- Locate artifacts robustly ----------
def find_path(*parts):
    """Join parts relative to this file, falling back to repo root."""
    here = Path(__file__).resolve()
    candidates = [
        here.parent.joinpath(*parts),                  # ./Models/...
        here.parent.parent.joinpath(*parts),           # ./App/Models/... (if app.py inside /App)
        here.parents[2].joinpath(*parts) if len(here.parents) > 2 else None  # fallback
    ]
    for p in candidates:
        if p and p.exists():
            return p
    return candidates[0]  # default first even if missing (error will be shown nicely)

@st.cache_resource(show_spinner=False)
def load_artifacts():
    models_dir = None
    # Try common locations
    for rel in ("Models", "App/Models", "Predicting_Employee_Attrition/Models"):
        p = find_path(rel)
        if p and p.exists():
            models_dir = p
            break
    if models_dir is None:
        raise FileNotFoundError("Could not find the Models/ folder.")

    model   = joblib.load(models_dir / "logreg_model.pkl")
    scaler  = joblib.load(models_dir / "scaler.pkl")
    columns = joblib.load(models_dir / "X_columns.pkl")
    return model, scaler, columns, models_dir

# ---------- Helpers ----------
NUMERIC_COLS = ["Age", "Salary", "Years of Service"]
DEFAULTS = {"Age": 35, "Salary": 65000, "Years of Service": 5}
CATEGORICALS = {
    "Department": ["HR","Billing","Sales & Marketing","Data Analytics","Operations","IT","Finance","Admin"],
    "Job Title": ["Data Scientist","Product Manager","Software Developer","Project Manager","HR Analyst","Customer Support Agent"],
    "Gender": ["Female","Male"],
    "Marital Status": ["Single","Married"]
}

def preprocess_df(df_raw: pd.DataFrame, feature_cols, scaler):
    df = df_raw.copy()
    # keep only known columns (others will be ignored)
    # one-hot encode categoricals
    df = pd.get_dummies(df, drop_first=False)
    # add any missing one-hot columns (from training)
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0
    # enforce column order
    df = df[feature_cols]
    # scale numeric columns that exist in features
    to_scale = [c for c in NUMERIC_COLS if c in df.columns]
    if to_scale:
        df[to_scale] = scaler.transform(df[to_scale].values)
    return df

def predict_one(payload: dict, model, scaler, feature_cols):
    raw = pd.DataFrame([payload])
    X   = preprocess_df(raw, feature_cols, scaler)
    prob = float(model.predict_proba(X)[0,1])
    label = "Likely to Exit" if prob >= 0.5 else "Likely to Stay"
    return prob, label

# ---------- Sidebar ----------
st.sidebar.title("Navigation")
st.sidebar.caption("Built by **Aisha Mohammed**")
st.sidebar.markdown(
    """
**Project**: HR Attrition Risk Suite  
Predict exit probability, score CSVs, and visualize risk by department.

**Model**: Logistic Regression  
**Tech**: scikit-learn, Streamlit, Plotly  
    """
)
with st.sidebar.expander("Need a sample CSV?"):
    sample = pd.DataFrame({
        "Department":["HR","IT"],
        "Job Title":["HR Analyst","Software Developer"],
        "Gender":["Female","Male"],
        "Marital Status":["Single","Married"],
        "Age":[28,32],
        "Years of Service":[2,4],
        "Salary":[42000,78000]
    })
    st.download_button("Download sample.csv", sample.to_csv(index=False), "sample.csv", "text/csv")

# ---------- Load artifacts ----------
try:
    model, scaler, FEATURE_COLS, MODELS_DIR = load_artifacts()
    st.toast("Artifacts loaded successfully.", icon="✅")
except Exception as e:
    st.error(f"Could not load artifacts.\n\n{e}")
    st.stop()

# ---------- Header ----------
st.markdown(
    """
# HR Attrition Risk Suite
*Predict exit probability, score CSVs, and view department-level risk.*
"""
)

# ---------- Tabs ----------
tab_overview, tab_single, tab_batch, tab_insights = st.tabs(
    ["Overview", "Single Prediction", "Batch Scoring", "Department Insights"]
)

with tab_overview:
    c1, c2, c3 = st.columns(3)
    c1.metric("Model", "Logistic Regression")
    c2.metric("Frameworks", "scikit-learn / Streamlit")
    c3.metric("Artifacts Folder", MODELS_DIR.name)

    st.markdown("### What’s inside")
    st.markdown(
        """
- **Single Prediction** — enter details to predict one employee’s attrition risk  
- **Batch Scoring** — upload a CSV and download results with predicted probability  
- **Department Insights** — average risk by department  
        """
    )

with tab_single:
    st.subheader("Single Prediction")
    with st.form("single_form"):
        colA, colB, colC = st.columns(3)
        with colA:
            dept = st.selectbox("Department", CATEGORICALS["Department"])
            job  = st.selectbox("Job Title", CATEGORICALS["Job Title"])
        with colB:
            gender = st.selectbox("Gender", CATEGORICALS["Gender"])
            ms     = st.selectbox("Marital Status", CATEGORICALS["Marital Status"])
        with colC:
            age    = st.number_input("Age", 18, 75, DEFAULTS["Age"])
            yos    = st.number_input("Years of Service", 0, 40, DEFAULTS["Years of Service"])
            salary = st.number_input("Salary", 0, 300000, DEFAULTS["Salary"])
        thresh = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05)
        submitted = st.form_submit_button("Predict")

    if submitted:
        payload = {
            "Department": dept, "Job Title": job,
            "Gender": gender, "Marital Status": ms,
            "Age": age, "Years of Service": yos, "Salary": salary
        }
        prob, label = predict_one(payload, model, scaler, FEATURE_COLS)

        k1, k2 = st.columns(2)
        k1.metric("Exit Probability", f"{prob:.2%}")
        k2.metric("Prediction", label)

        # nice gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob*100,
            number={'suffix': "%"},
            title={'text': "Exit Probability"},
            gauge={'axis': {'range': [0,100]},
                   'bar': {'thickness': 0.3},
                   'steps': [
                       {'range':[0, 40], 'color':"#1f77b4"},
                       {'range':[40,60], 'color':"#ffbf00"},
                       {'range':[60,100], 'color':"#d62728"},
                   ]}
        ))
        st.plotly_chart(fig, use_container_width=True)

with tab_batch:
    st.subheader("Batch Scoring (CSV)")
    st.caption("CSV must include: Department, Job Title, Gender, Marital Status, Age, Years of Service, Salary")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up:
        df_raw = pd.read_csv(up)
        st.markdown("**Preview**")
        st.dataframe(df_raw.head(), use_container_width=True)
        thresh_b = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05, key="th_b")
        if st.button("Score file"):
            X = preprocess_df(df_raw, FEATURE_COLS, scaler)
            probs = model.predict_proba(X)[:,1]
            out = df_raw.copy()
            out["Exit_Probability"] = probs.round(4)
            out["Prediction"] = np.where(out["Exit_Probability"] >= thresh_b, "Likely to Exit", "Likely to Stay")
            st.success("Scoring complete")
            st.dataframe(out.head(), use_container_width=True)
            st.download_button("Download scored CSV", out.to_csv(index=False).encode(), "scored_employees.csv", "text/csv")

with tab_insights:
    st.subheader("Department Insights")
    st.caption("Upload the scored CSV generated in the Batch tab to see average risk by department.")
    g = st.file_uploader("Upload scored CSV", type=["csv"], key="insights")
    if g is not None:
        df = pd.read_csv(g)
        need = {"Department","Exit_Probability"}
        if need.issubset(df.columns):
            agg = df.groupby("Department", as_index=False)["Exit_Probability"].mean().sort_values("Exit_Probability", ascending=False)
            fig = px.bar(agg, x="Exit_Probability", y="Department",
                         orientation="h", title="Average Exit Probability by Department",
                         text="Exit_Probability")
            fig.update_layout(xaxis_title="Avg Probability", yaxis_title="", uniformtext_minsize=10, uniformtext_mode='hide')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("File must include 'Department' and 'Exit_Probability'.")
