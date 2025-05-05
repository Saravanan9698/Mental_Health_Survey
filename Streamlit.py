# Optimized Streamlit App for Mental Health Depression Prediction

import base64
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# Set Streamlit page config
st.set_page_config(page_title="Mental Health Depression Prediction", layout="wide")

def img_to_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return None

# Background image setup
image_path = "D:/Projects/Mini_Projects/Mental_Health_Survey/Image/Neural_Networks.jpg"
img_base64 = img_to_base64(image_path)
if img_base64:
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)),
            url('data:image/jpeg;base64,{img_base64}');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
    """, unsafe_allow_html=True)

class MentalHealthPreprocessor:
    def __init__(self):
        self.encoders = {}
        self.columns_to_drop = []
        self.numerical_features = []
        self.categorical_features = []

    def fit(self, data):
        self.numerical_features = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
        self.categorical_features = data.select_dtypes(include=["object"]).columns.tolist()

        corr_matrix = data[self.numerical_features].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.columns_to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.6)]
        data = data.drop(columns=self.columns_to_drop, errors="ignore")
        self.numerical_features = data.select_dtypes(include=["int64", "float64"]).columns.tolist()

        for col in self.categorical_features:
            if col in data.columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                self.encoders[col] = le
        return data

    def transform(self, data):
        data = data.drop(columns=self.columns_to_drop, errors="ignore")
        for col in self.categorical_features:
            if col in data.columns and col in self.encoders:
                try:
                    data[col] = self.encoders[col].transform(data[col].astype(str))
                except ValueError:
                    data[col] = 0
        for col in self.numerical_features:
            if col not in data.columns:
                data[col] = 0
        for col in self.categorical_features:
            if col not in data.columns and col in self.encoders:
                data[col] = 0
        return data[self.numerical_features + [col for col in self.categorical_features if col in self.encoders]]

@st.cache_resource
def load_pickle(path, default=None):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return default

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("D:/Projects/Mini_Projects/Mental_Health_Survey/Model/neural_network.keras")

@st.cache_resource
def load_cleaning():
    return load_pickle("D:/Projects/Mini_Projects/Mental_Health_Survey/Model/cleaning.pkl", pd.DataFrame())

def predict_depression(input_data, model, preprocessor):
    try:
        processed = preprocessor.transform(pd.DataFrame([input_data]))
        processed = np.pad(processed, ((0, 0), (0, max(0, 13 - processed.shape[1]))), mode='constant')
        processed = processed[:, :13]
        output = model.predict(processed)
        label = "Depression Detected" if output[0] > 0.5 else "No Depression"
        return label, output[0][0]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return "Error", 0.0

# Sidebar navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Page", ["Home", "Upload Data", "Manual Entry", "Visualizations", "Bias Evaluation"])

if "predicted_data" not in st.session_state:
    st.session_state.predicted_data = pd.DataFrame()

if page == "Home":
    st.title("Mental Health Depression Prediction Application")
    st.subheader("Understanding Depression")
    st.markdown("""
        Depression is a serious mental health disorder that affects mood, thought, and behavior.
        This application allows prediction of depression based on user input or uploaded survey data.
        If you or someone you know is struggling, seek help—you're not alone.
    """)

elif page == "Upload Data":
    st.title("Upload Survey CSV")
    file = st.file_uploader("Upload CSV", type="csv")
    if file:
        df = pd.read_csv(file)
        st.write("### Raw Data", df.head())

        cleaned = load_cleaning()
        preprocessor = MentalHealthPreprocessor()
        preprocessor.fit(cleaned.copy())
        model = load_model()

        processed = preprocessor.transform(df.copy())
        processed = np.pad(processed, ((0, 0), (0, max(0, 13 - processed.shape[1]))), mode='constant')
        processed = processed[:, :13]

        if st.button("Predict"):
            predictions = model.predict(processed)
            result = df.copy()
            result["Prediction"] = (predictions > 0.5).astype(int)
            result["Confidence"] = predictions.round(2)

            st.session_state.predicted_data = result
            st.write("### Results", result.head())

            csv = result.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results", data=csv, file_name="depression_predictions.csv")

elif page == "Manual Entry":
    st.title("Manual Survey Input")
    input_data = {
        "Gender": st.selectbox("Gender", ["Male", "Female"]),
        "Age": st.slider("Age", 18, 80),
        "City": st.selectbox("City", ["Mumbai", "Delhi", "Chennai"]),
        "Sleep Duration": st.selectbox("Sleep Duration", ["6-8 hours", "Less than 5 hours", "More than 8 hours"]),
        "Dietary Habits": st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"]),
        "Have you ever had suicidal thoughts ?": st.selectbox("Suicidal Thoughts?", ["Yes", "No"]),
        "Work/Study Hours": st.slider("Work/Study Hours", 0, 24),
        "Financial Stress": st.slider("Financial Stress", 0, 5),
        "Family History of Mental Illness": st.selectbox("Family History?", ["Yes", "No"]),
    }

    role = st.radio("Are you a...", ["Professional", "Student"])
    if role == "Professional":
        input_data.update({
            "Profession": st.selectbox("Profession", ["Software Engineer", "Doctor", "Teacher"]),
            "Work Pressure": st.slider("Work Pressure", 0, 10),
            "Job Satisfaction": st.slider("Job Satisfaction", 0, 10),
        })
    else:
        input_data.update({
            "Academic Pressure": st.slider("Academic Pressure", 0, 10),
            "CGPA": st.number_input("CGPA", 0.0, 10.0),
            "Study Satisfaction": st.slider("Study Satisfaction", 0, 10),
            "Degree": st.selectbox("Degree", ["BSc", "BA", "B.Tech"]),
        })

    if st.button("Predict"):
        model = load_model()
        cleaned = load_cleaning()
        preprocessor = MentalHealthPreprocessor()
        preprocessor.fit(cleaned.copy())
        pred, conf = predict_depression(input_data, model, preprocessor)
        st.success(f"Prediction: {pred}")
        st.info(f"Confidence Score: {conf:.2f}")

elif page == "Visualizations":
    st.title("Data Insights & Visualizations")
    if not st.session_state.predicted_data.empty:
        df = st.session_state.predicted_data.copy()
        st.write("### Predicted Data", df.head())
        numeric_data = df.select_dtypes(include=[np.number])

        if not numeric_data.empty:
            fig, ax = plt.subplots(figsize=(14, 6))
            sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

            fig2, ax2 = plt.subplots()
            sns.countplot(x="Prediction", data=df, ax=ax2)
            st.pyplot(fig2)
    else:
        st.warning("Please upload data and generate predictions to view visualizations.")

elif page == "Bias Evaluation":
    st.title("Bias Evaluation")
    df = st.session_state.predicted_data.copy()
    if df.empty:
        st.warning("Please upload data and generate predictions first.")
        st.stop()

    group_factors = ["Gender", "Age", "City", "Sleep Duration", "Dietary Habits", "Work/Study Hours"]
    for factor in group_factors:
        if factor in df.columns:
            st.write(f"### {factor} vs Prediction")
            fig, ax = plt.subplots(figsize=(14, 6))
            sns.countplot(x=factor, hue="Prediction", data=df, ax=ax)
            st.pyplot(fig)

            if df[factor].dtype != 'object':
                bins = pd.cut(df[factor], bins=5)
                grouped = df.groupby(bins)["Prediction"].mean().reset_index()
            else:
                grouped = df.groupby(factor)["Prediction"].mean().reset_index()
            grouped.columns = [factor, "Depression Prediction Rate"]
            fig2, ax2 = plt.subplots(figsize=(14, 6))
            sns.barplot(x=factor, y="Depression Prediction Rate", data=grouped, ax=ax2)
            plt.ylim(0, 1)
            st.pyplot(fig2)
            st.dataframe(grouped)
