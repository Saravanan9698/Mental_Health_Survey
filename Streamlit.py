import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Load the trained model (Replace with actual model path if available)
try:
    model = joblib.load("D:\Projects\Mini_Projects\Mental_Health_Survey\Model\neural_network.keras")  # Ensure you have a trained model
except:
    model = None

# Load and display the image
image_path = r"D:\Projects\Mini_Projects\Mental_Health_Survey\Image\stock-photo-human-brain-stimulation-or-activity-with-neuron-close-up-d-rendering-illustration-neurology-1907619667.jpg"
image = Image.open(image_path)

# Streamlit UI
st.set_page_config(page_title="Mental Health Prediction", layout="centered")
st.title("🧠 Mental Health Survey & Prediction")

# Display image
st.image(image, caption="Brain Activity Illustration", use_column_width=True)

st.markdown("### Please fill out the survey")

# Collect user inputs
age = st.slider("Age", 18, 100, 25)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
sleep_hours = st.slider("Average Sleep Hours", 3.0, 10.0, 6.5)
exercise = st.selectbox("Exercise Frequency", ["Never", "Rarely", "Sometimes", "Often", "Daily"])
work_hours = st.slider("Work Hours per Week", 10, 80, 40)

# Convert categorical values to numerical
gender_map = {"Male": 0, "Female": 1, "Other": 2}
exercise_map = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3, "Daily": 4}

data = pd.DataFrame({
    "age": [age],
    "gender": [gender_map[gender]],
    "stress_level": [stress_level],
    "sleep_hours": [sleep_hours],
    "exercise": [exercise_map[exercise]],
    "work_hours": [work_hours]
})

# Prediction
if st.button("Predict Mental Health Status"):
    if model:
        prediction = model.predict(data)[0]
        st.success(f"Predicted Mental Health Status: {'Stable' if prediction == 0 else 'At Risk'}")
    else:
        st.error("Model not found. Train and provide a valid model.")
