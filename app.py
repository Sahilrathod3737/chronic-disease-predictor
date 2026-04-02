import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Healthcare AI Analytics", layout="wide")

# 2. Model Load
try:
    model = pickle.load(open('heart_model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: 'heart_model.pkl' nahi mili! Pehle train_model.py run karein.")

# --- UI Header ---
st.title("🩺 Predictive Modeling of Chronic Disease Trends")
st.markdown("Early disease prediction using Machine Learning and Healthcare Data.")

# --- MODULE 1: Single Patient Prediction ---
st.header("👤 Single Patient Prediction")
st.write("Patient ki clinical details enter karein:")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 1, 100, 30)
    sex = st.selectbox("Sex (1=Male, 0=Female)", [1, 0])
with col2:
    trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
    chol = st.slider("Cholesterol", 100, 500, 200)
with col3:
    cp = st.slider("Chest Pain Type (0-3)", 0, 3, 1)
    thalach = st.slider("Max Heart Rate", 60, 220, 150)

if st.button("Analyze Single Patient"):
    # Input data
    single_data = pd.DataFrame(
        {'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps], 'chol': [chol], 'thalach': [thalach]})

    # Prediction logic
    input_cols = model.feature_names_in_
    single_data_final = single_data.reindex(columns=input_cols, fill_value=0)

    # Prediction logic
    prediction = model.predict(single_data_final)

    # Safety Check for Probability
    try:
        # Agar model dono results (0 aur 1) jaanta hai
        probability = model.predict_proba(single_data_final)[0][1]
    except IndexError:
        # Agar model sirf ek hi result jaanta hai (Test data ki wajah se)
        probability = 1.0 if prediction[0] == 1 else 0.0

    # Result Display
    st.divider()
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        if prediction[0] == 1:
            st.error("Result: High Risk Detected")
        else:
            st.success("Result: Low Risk / Normal")
    with res_col2:
        st.metric(label="Risk Percentage", value=f"{round(probability * 100, 2)}%")

st.divider()

# --- MODULE 2: Bulk Prediction (Updated) ---
st.header("📂 Bulk Prediction Module")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
   # st.write("Data Preview:", data.head())
   # Purani line: st.write("Data Preview:", data.head())
# Nayi line (Poora data dikhane ke liye):
st.write("Data Preview:", data)
    # Button click check
    if st.button("Predict All Patients"):
        # Technical Logic: Model features align karna
        input_cols = model.feature_names_in_
        data_final = data.reindex(columns=input_cols, fill_value=0)

        # Predictions
        preds = model.predict(data_final)

        # Probability handling
        try:
            probs = model.predict_proba(data_final)[:, 1]
        except:
            probs = [1.0 if p == 1 else 0.0 for p in preds]

        # Results columns add karna
        data['Prediction'] = ["High Risk" if p == 1 else "Low Risk" for p in preds]
        data['Confidence (%)'] = (np.array(probs) * 100).round(2)

        st.success("✅ Batch Prediction Successful!")
        st.dataframe(data)  # Isse puri table dikhegi

        # Download button
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results", csv, "results.csv", "text/csv")
