# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# -------------------------------
# 1. Page Config
# -------------------------------
st.set_page_config(page_title="Healthcare DSS Prototype", layout="wide")
st.title("Decision Support System for Chronic Disease Risk Prediction")

# -------------------------------
# 2. Dataset Loading
# -------------------------------
@st.cache_data
def load_datasets():
    diabetes = pd.read_csv("diabetes.csv")
    heart = pd.read_csv("heart.csv")
    return diabetes, heart

diabetes, heart = load_datasets()

# -------------------------------
# 3. Convert Heart Disease to Binary
# -------------------------------
# 0 = no heart disease, 1 = any heart disease
heart["num"] = heart["num"].apply(lambda x: 0 if x == 0 else 1)

# -------------------------------
# 4. Preprocessing Function
# -------------------------------
def preprocess_data(df, target_col, drop_cols=None):
    df = df.copy()
    
    if drop_cols:
        df = df.drop(columns=drop_cols)
    
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    
    df = df.fillna(df.median(numeric_only=True))
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    
    return X, X_scaled, y, scaler

# -------------------------------
# 5. Train Models Function
# -------------------------------
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        results[name] = {"model": model, "accuracy": acc}
    
    return results, (X_test, y_test)

# -------------------------------
# 6. Preprocess datasets and train models
# -------------------------------
X_dia, X_dia_scaled, y_dia, scaler_dia = preprocess_data(diabetes, target_col="Outcome")
dia_models, dia_test = train_models(X_dia_scaled, y_dia)

X_heart, X_heart_scaled, y_heart, scaler_heart = preprocess_data(
    heart, target_col="num", drop_cols=["id", "dataset"]
)
heart_models, heart_test = train_models(X_heart_scaled, y_heart)

# -------------------------------
# 7. Risk Stratification Function
# -------------------------------
def stratify_risk(prob):
    if prob < 0.33:
        return "Low Risk", "✅ Maintain healthy lifestyle, regular checkups"
    elif prob < 0.66:
        return "Medium Risk", "⚠ Improve diet & exercise, monitor regularly"
    else:
        return "High Risk", "🚨 Consult doctor immediately, further tests required"

# -------------------------------
# 8. User Interface
# -------------------------------
st.sidebar.header("Select Disease to Predict")
choice = st.sidebar.radio("Choose Model:", ["Diabetes", "Heart Disease"])

def display_result(risk, recommendation):
    color_map = {
        "Low Risk": "green",
        "Medium Risk": "orange",
        "High Risk": "red"
    }
    st.markdown(
        f"<h3>Predicted Risk Level: <span style='color:{color_map[risk]};'>{risk}</span></h3>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<p style='font-size:22px;'><b>Recommendation:</b> <span style='color:pink; font-size:20px;'>{recommendation}</span></p>",
        unsafe_allow_html=True
    )

if choice == "Diabetes":
    st.header("🩺 Diabetes Risk Prediction (PIMA Dataset)")
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose Level", 0, 300, 120)
    bp = st.number_input("Blood Pressure", 0, 200, 70)
    skin = st.number_input("Skin Thickness", 0, 100, 20)
    insulin = st.number_input("Insulin Level", 0, 900, 80)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.number_input("Age", 1, 120, 30)
    
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_data_scaled = scaler_dia.transform(input_data)
    
    model = dia_models["Random Forest"]["model"]
    prob = model.predict_proba(input_data_scaled)[0][1]
    risk, recommendation = stratify_risk(prob)
    
    if st.button("Predict Diabetes Risk"):
        display_result(risk, recommendation)

elif choice == "Heart Disease":
    st.header("❤ Heart Disease Risk Prediction (UCI Dataset)")
    age = st.number_input("Age", 20, 100, 45)
    sex = st.selectbox("Sex (1=Male, 0=Female)", [0, 1])
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1=Yes, 0=No)", [0, 1])
    restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
    thalch = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [0, 1])
    oldpeak = st.number_input("Oldpeak (ST Depression)", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope (0-2)", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thal (0=Normal,1=Fixed Defect,2=Reversible Defect)", [0, 1, 2])
    
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalch, exang, oldpeak, slope, ca, thal]])
    input_data_scaled = scaler_heart.transform(input_data)
    
    model = heart_models["Random Forest"]["model"]
    prob = model.predict_proba(input_data_scaled)[0][1]
    risk, recommendation = stratify_risk(prob)
    
    if st.button("Predict Heart Disease Risk"):
        display_result(risk, recommendation)

# -------------------------------
# 9. Model Evaluation Section
# -------------------------------
st.sidebar.subheader("Evaluation")
if st.sidebar.checkbox("Show Evaluation Metrics"):
    if choice == "Diabetes":
        X_test, y_test = dia_test
        model = dia_models["Random Forest"]["model"]
    else:
        X_test, y_test = heart_test
        model = heart_models["Random Forest"]["model"]
    
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    roc_auc = auc(fpr, tpr)
    
    st.write("### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)
    
    st.write("### ROC Curve")
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0,1], [0,1], "r--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)
