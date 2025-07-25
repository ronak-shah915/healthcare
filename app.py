import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

def check_login():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.subheader("🔐 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username == "admin" and password == "1234":
                st.session_state.authenticated = True
                st.success("✅ Logged in successfully!")
            else:
                st.error("❌ Invalid credentials")
        return False
    return True

if not check_login():
    st.stop()
    
scaler = joblib.load('scaler.pkl')
nb = joblib.load('naive_bayes.pkl')
rf = joblib.load('rf_model.pkl')
lr = joblib.load('lr_model.pkl')
dl = tf.keras.models.load_model('diabetes_dl_model.h5')

st.title("Diabetes Risk Predictor")
st.sidebar.header("Patient Data")

def get_input():
    return {
        'Pregnancies': st.sidebar.number_input("Pregnancies", 0, 20, 0),
        'Glucose': st.sidebar.number_input("Glucose", 0, 300, 120),
        'BloodPressure': st.sidebar.number_input("BloodPressure", 0, 200, 70),
        'SkinThickness': st.sidebar.number_input("SkinThickness", 0, 100, 20),
        'Insulin': st.sidebar.number_input("Insulin", 0, 900, 79),
        'BMI': st.sidebar.number_input("BMI", 0.0, 70.0, 25.5),
        'DiabetesPedigreeFunction': st.sidebar.number_input("Pedigree Function", 0.0, 3.0, 0.42),
        'Age': st.sidebar.number_input("Age", 1, 120, 29)
    }

patient = get_input()
df_input = pd.DataFrame([patient])

median_values = {
    'Glucose': 117.0,
    'BloodPressure': 72.0,
    'SkinThickness': 23.0,
    'Insulin': 30.5,
    'BMI': 32.0
}

for col in median_values:
    if df_input.at[0, col] == 0:
        df_input.at[0, col] = median_values[col]

X = scaler.transform(df_input)

def get_risk_level(prob):
    if prob < 0.33:
        return "Low Risk"
    elif prob < 0.66:
        return "Medium Risk"
    else:
        return "High Risk"

if st.button("Predict"):
    st.subheader("Model Predictions")
    preds = {
        'Naive Bayes': nb.predict_proba(X)[0][1],
        'Random Forest': rf.predict_proba(X)[0][1],
        'Logistic Regression': lr.predict_proba(X)[0][1],
        'Deep Learning': dl.predict(X)[0][0]
    }

    for name, prob in preds.items():
        risk = get_risk_level(prob)
        st.write(f"{name}: **{prob:.2f}** probability of diabetes — *{risk}*")

    names = list(preds.keys())
    probabilities = list(preds.values())
    fig, ax = plt.subplots()
    bars = ax.bar(names, probabilities, color=['#4caf50', '#2196f3', '#ff9800', '#9c27b0'])
    ax.set_ylim([0, 1])
    ax.set_ylabel('Probability')
    ax.set_title('Diabetes Risk Probability by Model')

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    st.pyplot(fig)
