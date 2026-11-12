import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

st.set_page_config(page_title="Diabetes Risk Predictor", layout="wide")

BASE_FEATURES = [ "Glucose", "BloodPressure", "SkinThickness",
                 "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
EXTRA_FEATURES = ["Glucose_Insulin_Ratio", "Glucose_BMI",
                  "BMI_Age"]
FEATURES = BASE_FEATURES + EXTRA_FEATURES

ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Glucose_Insulin_Ratio"] = df["Glucose"] / df["Insulin"]
    df["Glucose_BMI"] = df["Glucose"] * df["BMI"] / 100
    df["BMI_Age"] = df["BMI"] * df["Age"] / 100
    return df


@st.cache_resource
def train_model():
    df = pd.read_csv(Path(__file__).parent / "diabetes.csv")
    df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, np.nan)
    df = add_features(df)
    X, y = df[FEATURES], df["Outcome"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    model = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss")
    model.fit(X_tr, y_tr)

    proba = model.predict_proba(X_te)[:, 1]
    metrics = {"accuracy": accuracy_score(y_te, proba > 0.5),
               "auc": roc_auc_score(y_te, proba)}
    explainer = shap.TreeExplainer(model)
    return model, explainer, X_te, metrics


model, explainer, X_test, metrics = train_model()

st.title("🩺 Diabetes Risk Predictor")
st.caption("Educational demo only. Not a medical diagnosis.")

st.sidebar.header("Patient data")

age = st.sidebar.number_input("Age (years)", 18, 90, 35)

st.sidebar.subheader("Body measurements")
use_calc = st.sidebar.checkbox("Calculate BMI from height & weight")
if use_calc:
    height = st.sidebar.number_input("Height (cm)", 120, 220, 165)
    weight = st.sidebar.number_input("Weight (kg)", 30, 200, 70)
    bmi = round(weight / (height / 100) ** 2, 1)
    st.sidebar.info(f"BMI = {bmi}")
else:
    bmi = st.sidebar.slider("BMI", 15.0, 70.0, 28.0, 0.1)
skin = st.sidebar.slider("Skin thickness (mm)", 5, 100, 25)

st.sidebar.subheader("Blood tests")
glucose = st.sidebar.slider("Glucose (mg/dL)", 40, 250, 120)
insulin = st.sidebar.slider("Insulin (mu U/ml)", 10, 850, 80)
bp = st.sidebar.slider("Blood pressure (mm Hg)", 30, 130, 70)

st.sidebar.subheader("Family history")
dpf = st.sidebar.slider("Pedigree function", 0.05, 2.5, 0.4, 0.01)

inputs = {
    "Glucose": glucose,
    "BloodPressure": bp,
    "SkinThickness": skin,
    "Insulin": insulin,
    "BMI": bmi,
    "DiabetesPedigreeFunction": dpf,
    "Age": age,
}
patient = add_features(pd.DataFrame([inputs]))[FEATURES]

risk = float(model.predict_proba(patient)[0, 1])
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Predicted risk")
    st.metric("Diabetes probability", f"{risk:.0%}")
    st.progress(risk)
    if risk >= 0.7:
        st.error("High risk. Consider consulting a doctor.")
    elif risk >= 0.4:
        st.warning("Moderate risk. Monitor and consider a checkup.")
    else:
        st.success("Lower risk.")
    st.caption(f"Model test accuracy: {metrics['accuracy']:.1%} | "
               f"ROC-AUC: {metrics['auc']:.2f}")

    st.subheader("Your inputs")
    st.dataframe(patient.T.rename(columns={0: "Value"}).round(2))

with col2:
    st.subheader("Why this prediction?")
    st.write("Red bars push risk **up**, blue bars push it **down**.")
    exp = explainer(patient)
    fig = plt.figure()
    shap.plots.waterfall(exp[0], max_display=12, show=False)
    st.pyplot(fig)
    plt.close(fig)

st.divider()
st.subheader("Global feature importance")
shap_values = explainer(X_test)
fig2 = plt.figure()
shap.plots.bar(shap_values, max_display=12, show=False)
st.pyplot(fig2)
plt.close(fig2)
