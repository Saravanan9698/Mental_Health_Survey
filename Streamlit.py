import base64
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Set Streamlit page config
st.set_page_config(page_title="Mental Health Depression Prediction", layout="wide")

def img_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Apply background image
image_path = r"D:\\Projects\\Mini_Projects\\Mental_Health_Survey\\Image\\Neural_Networks.jpg"
try:
    img_base64 = img_to_base64(image_path)
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), url('data:image/jpeg;base64,{img_base64}');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            padding: 0;
        }}
        .title-container {{
            text-align: center;
            color: white;
            font-size: 4em;
            margin-top: 350px;
        }}
        </style>
    """, unsafe_allow_html=True)
except FileNotFoundError:
    st.error("Image not found at the specified path.")

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

    def ensure_all_features(self, data):
        for col in self.numerical_features:
            data[col] = data.get(col, 0)
        for col in self.categorical_features:
            data[col] = data.get(col, 0)
        return data[self.numerical_features + [col for col in self.categorical_features if col in self.encoders]]

    def transform(self, data):
        data = data.drop(columns=self.columns_to_drop, errors="ignore")
        for col in self.categorical_features:
            if col in data.columns and col in self.encoders:
                try:
                    data[col] = self.encoders[col].transform(data[col].astype(str))
                except ValueError:
                    data[col] = 0
        return self.ensure_all_features(data)

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
    try:
        return tf.keras.models.load_model(r"D:\\Projects\\Mini_Projects\\Mental_Health_Survey\\Model\\neural_network.keras")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_resource
def load_cleaning():
    return load_pickle(r"D:\\Projects\\Mini_Projects\\Mental_Health_Survey\\Model\\cleaning.pkl", pd.DataFrame())

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

# Sidebar and page routing
st.title("Mental Health Depression Prediction Application")
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Page", ["Home", "Upload Data", "Manual Entry", "Visualizations", "Bias_Evaluation"])

if "predicted_data" not in st.session_state:
    st.session_state.predicted_data = pd.DataFrame()

if page == "Home":
    st.subheader("Understanding Depression")
    st.markdown("""
        Depression is a serious mental health disorder that affects mood, thought, and behavior.
        This application allows prediction of depression based on user input or uploaded survey data.
        If you or someone you know is struggling, seek help—you're not alone.
    """)

elif page == "Upload Data":
    st.header("Upload Survey CSV")
    file = st.file_uploader("Upload CSV", type="csv")
    if file:
        df = pd.read_csv(file)
        st.write("### Raw Data", df.head())

        cleaned = load_cleaning()
        preprocessor = MentalHealthPreprocessor()
        model = load_model()

        preprocessor.fit(cleaned.copy())
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
    st.header("Manual Survey Input")

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
        preprocessor = MentalHealthPreprocessor()
        cleaned = load_cleaning()
        preprocessor.fit(cleaned.copy())
        pred, conf = predict_depression(input_data, model, preprocessor)
        st.success(f"Prediction: {pred}")
        st.info(f"Confidence Score: {conf:.2f}")

elif page == "Visualizations":
    st.header("Data Insights & Visualizations")
    preprocessor = MentalHealthPreprocessor()
    model = load_model()
    cleaned_data = load_cleaning()

    preprocessor.fit(cleaned_data.copy())
    preprocessed_data = preprocessor.transform(cleaned_data.copy())

    if preprocessed_data.shape[1] > 13:
        preprocessed_data = preprocessed_data.iloc[:, :13]

    if not st.session_state["predicted_data"].empty:
        st.write("#### Predicted Data Overview")
        st.dataframe(st.session_state["predicted_data"].head())

        numeric_data = st.session_state["predicted_data"].select_dtypes(include=[np.number])

        if not numeric_data.empty:
            st.write("#### Correlation Heatmap (Predicted Data)")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        st.write("#### Depression Distribution in Predictions")
        fig2, ax2 = plt.subplots()
        sns.countplot(x="Prediction", data=st.session_state["predicted_data"], ax=ax2)
        st.pyplot(fig2)
    else:
        st.warning("No predicted data available. Please run predictions in the Upload Data section first.")

elif page == "Bias_Evaluation":
    st.header("Bias Evaluation for Prediction Distribution")
    if "predicted_data" in st.session_state and not st.session_state["predicted_data"].empty:
        data = st.session_state["predicted_data"].copy()
    else:
        st.warning("No predicted data available. Please run predictions in the Upload Data section first.")
        st.stop()

    if "Gender" in data.columns and "Prediction" in data.columns:
        st.write("#### Gender Distribution in Predictions")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x="Gender", hue="Prediction", data=data, ax=ax)
        st.pyplot(fig)

        gender_stats = data.groupby("Gender")["Prediction"].mean().reset_index()
        gender_stats.columns = ["Gender", "Depression Prediction Rate"]
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.barplot(x="Gender", y="Depression Prediction Rate", data=gender_stats, ax=ax2)
        plt.ylim(0, 1)
        st.pyplot(fig2)
        st.dataframe(gender_stats)

    if "Age" in data.columns and "Prediction" in data.columns:
        st.write("#### Age Distribution in Predictions")
        data["Age Group"] = pd.cut(data["Age"], bins=[17, 25, 35, 45, 55, 85], labels=["18-25", "26-35", "36-45", "46-55", "56+"])
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.countplot(x="Age Group", hue="Prediction", data=data, ax=ax3)
        st.pyplot(fig3)

        age_stats = data.groupby("Age Group")["Prediction"].mean().reset_index()
        age_stats.columns = ["Age Group", "Depression Prediction Rate"]
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        sns.barplot(x="Age Group", y="Depression Prediction Rate", data=age_stats, ax=ax4)
        plt.ylim(0, 1)
        st.pyplot(fig4)
        st.dataframe(age_stats)

    for factor in ["City", "Sleep Duration", "Dietary Habits", "Work/Study Hours"]:
        if factor in data.columns and "Prediction" in data.columns:
            st.write(f"#### {factor} Distribution in Predictions")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(x=factor, hue="Prediction", data=data, ax=ax)
            st.pyplot(fig)
