import streamlit as st
import joblib
import numpy as np

# Page configuration
st.set_page_config(page_title="Heart Health AI", page_icon="❤️", layout="wide")

# Custom CSS for a Professional Look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #e63946;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover { background-color: #d62828; color: white; }
    .report-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    footer {visibility: hidden;}
    .custom-footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #1d3557;
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)

# 1. Model loading
try:
    model = joblib.load("rf_model.pkl")
except:
    st.error("Model file not found. Please run Model.py first.")

# Sidebar - Branding & Info
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/822/822118.png", width=80)
    st.title("Project Info")
    st.markdown("""
    **Intern:** Shaiba Ali  
    **ID:** #CC69858  
    **Level:** Golden Level  
    **Domain:** Data Science
    """)
    st.write("---")
    st.help("Fill all health metrics accurately for better prediction.")

# Main UI
st.title("🏥 Heart Disease Risk Assessment")
st.markdown("### Clinical Parameters Input")

# Input Section with Columns
with st.container():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", 1, 100, 45)
        sex = st.selectbox("Gender", options=[(1, "Male"), (0, "Female")], format_func=lambda x: x[1])[0]
        cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
        trestbps = st.number_input("Resting BP (mm Hg)", 80, 200, 120)

    with col2:
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 240)
        fbs = st.selectbox("Fasting Blood Sugar > 120", options=[(1, "Yes"), (0, "No")], format_func=lambda x: x[1])[0]
        restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate", 60, 220, 150)

    with col3:
        exang = st.selectbox("Exercise Induced Angina", options=[(1, "Yes"), (0, "No")], format_func=lambda x: x[1])[0]
        oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
        slope = st.selectbox("ST Slope (0-2)", [0, 1, 2])
        ca = st.selectbox("Major Vessels (0-4)", [0, 1, 2, 3, 4])

# Center the Predict Button
st.write("")
if st.button("GENERATE DIAGNOSIS REPORT"):
    # Pre-processing input (Adjust based on your 13 columns)
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, 0]]) # Adding 13th feature placeholder
    
    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)[0][1]

    st.write("---")
    st.markdown("### 📋 Diagnostic Result")
    
    if prediction[0] == 1:
        st.error(f"**HIGH RISK DETECTED** (Risk Probability: {prob*100:.1f}%)")
        st.warning("⚠️ Recommendation: Immediate consultation with a Cardiologist is advised.")
    else:
        st.success(f"**LOW RISK DETECTED** (Confidence: {(1-prob)*100:.1f}%)")
        st.info("✅ Recommendation: Maintain current healthy lifestyle and regular checkups.")

# Permanent Professional Footer
st.markdown(f"""
    <div class="custom-footer">
        Developed by: <b>SHAIBA ALI</b> | Email: sa1610166@gmail.com | 
        © 2026 CodeClause Internship
    </div>
    """, unsafe_allow_html=True)