import streamlit as st
import requests

# Link to our live FastAPI server and strict API key
API_URL = "http://127.0.0.1:8000/predict"
HEADERS = {"X-API-Key": "advance-soft-logics-beastmode"}

st.set_page_config(page_title="AI/ML Diagnostics", layout="centered")
st.title("🩺 Breast Cancer Diagnostic AI")
st.write("This dashboard connects securely to our FastAPI inference engine.")

# A perfect 30-feature payload mimicking a real medical record
sample_data = {
    "mean_radius": 17.99, "mean_texture": 10.38, "mean_perimeter": 122.8, "mean_area": 1001.0, 
    "mean_smoothness": 0.1184, "mean_compactness": 0.2776, "mean_concavity": 0.3001, "mean_concave_points": 0.1471, 
    "mean_symmetry": 0.2419, "mean_fractal_dimension": 0.07871, 
    "radius_error": 1.095, "texture_error": 0.9053, "perimeter_error": 8.589, "area_error": 153.4, 
    "smoothness_error": 0.006399, "compactness_error": 0.04904, "concavity_error": 0.05373, "concave_points_error": 0.01587, 
    "symmetry_error": 0.03003, "fractal_dimension_error": 0.006193, 
    "worst_radius": 25.38, "worst_texture": 17.33, "worst_perimeter": 184.6, "worst_area": 2019.0, 
    "worst_smoothness": 0.1622, "worst_compactness": 0.6656, "worst_concavity": 0.7119, "worst_concave_points": 0.2654, 
    "worst_symmetry": 0.4601, "worst_fractal_dimension": 0.1189
}

st.subheader("Patient Clinical Data (30 Features)")
st.json(sample_data)

if st.button("Run AI Diagnostics", type="primary"):
    with st.spinner("Transmitting data to FastAPI engine..."):
        try:
            response = requests.post(API_URL, json=sample_data, headers=HEADERS)
            if response.status_code == 200:
                result = response.json()
                prediction = result["prediction"]
                confidence = result["confidence_score"] * 100
                
                if prediction == "Malignant":
                    st.error(f"⚠️ **Diagnosis:** {prediction} (Confidence: {confidence:.2f}%)")
                else:
                    st.success(f"✅ **Diagnosis:** {prediction} (Confidence: {confidence:.2f}%)")
                    
                st.info(f"Engine: {result['model_version']}")
            else:
                st.error(f"API Error ({response.status_code}): {response.text}")
        except Exception as e:
            st.error(f"Connection Failed: Ensure the FastAPI server is running. Error: {e}")