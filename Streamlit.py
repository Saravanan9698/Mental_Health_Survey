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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

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
    st.header("Data Insights & Visualizations")
    if st.session_state.predicted_data.empty:
        st.warning("No predicted data available. Please run predictions in the Upload Data section.")
        st.stop()

    st.write("#### Predicted Data Overview")
    st.dataframe(st.session_state.predicted_data.head())

    numeric_data = st.session_state.predicted_data.select_dtypes(include=[np.number])
    if not numeric_data.empty:
        st.write("#### Correlation Heatmap")
        st.write("Numeric columns:", numeric_data.columns.tolist())
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", ax=ax, annot_kws={"size": 10}, fmt=".2f")
        plt.title("Correlation Heatmap of Numeric Features", fontsize=8)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout(pad=3.0)
        st.pyplot(fig)

    st.write("#### Depression Prediction Distribution")
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    sns.countplot(x="Prediction", data=st.session_state.predicted_data, ax=ax2)
    ax2.set_title("Distribution of Depression Predictions", fontsize=8)
    ax2.set_xlabel("Prediction", fontsize=8)
    ax2.set_ylabel("Count", fontsize=8)
    plt.xticks(fontsize=8)
    plt.tight_layout(pad=3.0)
    st.pyplot(fig2)

    # Age vs. Depression Prediction Boxplot
    if "Age" in st.session_state.predicted_data.columns and "Prediction" in st.session_state.predicted_data.columns:
        st.write("#### Age Distribution by Depression Prediction")
        data = st.session_state.predicted_data.copy()
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        sns.boxplot(x="Prediction", y="Age", data=data, ax=ax3)
        ax3.set_title("Age Distribution by Depression Prediction", fontsize=8)
        ax3.set_xlabel("Prediction (0: No Depression, 1: Depression Detected)", fontsize=8)
        ax3.set_ylabel("Age", fontsize=8)
        ax3.set_xticklabels(["No Depression", "Depression Detected"], fontsize=8)
        plt.tight_layout(pad=3.0)
        st.pyplot(fig3)

    # Financial Stress vs. Prediction Violin Plot
    if "Financial Stress" in st.session_state.predicted_data.columns and "Prediction" in st.session_state.predicted_data.columns:
        st.write("#### Financial Stress Distribution by Depression Prediction")
        data = st.session_state.predicted_data.copy()
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        sns.violinplot(x="Prediction", y="Financial Stress", data=data, ax=ax4)
        ax4.set_title("Financial Stress Distribution by Depression Prediction", fontsize=8)
        ax4.set_xlabel("Prediction (0: No Depression, 1: Depression Detected)", fontsize=8)
        ax4.set_ylabel("Financial Stress", fontsize=8)
        ax4.set_xticklabels(["No Depression", "Depression Detected"], fontsize=8)
        plt.tight_layout(pad=3.0)
        st.pyplot(fig4)

elif page == "Bias Evaluation":
    st.header("Bias Evaluation for Prediction Distribution")
    if st.session_state.predicted_data.empty:
        st.warning("No predicted data available. Please run predictions in the Upload Data section.")
        st.stop()

    data = st.session_state.predicted_data.copy()
    data["City"] = data["City"].replace({np.nan: "Unknown", "Mumbay": "Mumbai"})

    # Model performance metrics
    if "Prediction" in data.columns and "Confidence" in data.columns:
        st.write("#### Model Performance Metrics")
        y_true = data["Prediction"]
        y_pred = (data["Confidence"] > 0.5).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        st.write(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")

        cm = confusion_matrix(y_true, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        ax_cm.set_title("Confusion Matrix", fontsize=8)
        ax_cm.set_xlabel("Predicted", fontsize=8)
        ax_cm.set_ylabel("True", fontsize=8)
        plt.tight_layout(pad=3.0)
        st.pyplot(fig_cm)

    if "Age" in data.columns and "Prediction" in data.columns:
        st.write("#### Age Distribution in Predictions")
        data["Age Group"] = pd.cut(data["Age"], bins=[17, 25, 35, 45, 55, 85], labels=["18-25", "26-35", "36-45", "46-55", "56+"])
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.countplot(x="Age Group", hue="Prediction", data=data, ax=ax)
        ax.set_title("Depression Predictions by Age Group", fontsize=8)
        ax.set_xlabel("Age Group", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.tight_layout(pad=3.0)
        st.pyplot(fig)

        age_stats = data.groupby("Age Group")["Prediction"].mean().reset_index()
        age_stats.columns = ["Age Group", "Depression Prediction Rate"]
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        sns.barplot(x="Age Group", y="Depression Prediction Rate", data=age_stats, ax=ax2)
        ax2.set_title("Depression Prediction Rate by Age Group", fontsize=8)
        ax2.set_xlabel("Age Group", fontsize=8)
        ax2.set_ylabel("Prediction Rate", fontsize=8)
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.tight_layout(pad=3.0)
        st.pyplot(fig2)
        st.dataframe(age_stats)

    if "Gender" in data.columns and "Prediction" in data.columns:
        st.write("#### Gender Distribution in Predictions")
        st.write("Columns in data:", data.columns.tolist())
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        sns.countplot(x="Gender", hue="Prediction", data=data, ax=ax3)
        ax3.set_title("Depression Predictions by Gender", fontsize=8)
        ax3.set_xlabel("Gender", fontsize=8)
        ax3.set_ylabel("Count", fontsize=8)
        ax3.set_xticklabels(data["Gender"].unique(), rotation=45, ha='right', fontsize=8)

        plt.tight_layout(pad=3.0)
        st.pyplot(fig3)

        gender_stats = data.groupby("Gender")["Prediction"].mean().reset_index()
        gender_stats.columns = ["Gender", "Depression Prediction Rate"]
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        sns.barplot(x="Gender", y="Depression Prediction Rate", data=gender_stats, ax=ax4)
        ax4.set_title("Depression Prediction Rate by Gender", fontsize=8)
        ax4.set_xlabel("Gender", fontsize=8)
        ax4.set_ylabel("Prediction Rate", fontsize=8)
        ax3.set_xticklabels(data["Gender"].unique(), rotation=45, ha='right', fontsize=14)

        plt.ylim(0, 1)
        plt.tight_layout(pad=3.0)
        st.pyplot(fig4)
        st.dataframe(gender_stats)

    for factor in ["City", "Sleep Duration", "Dietary Habits", "Work/Study Hours"]:
        if factor in data.columns and "Prediction" in data.columns:
            st.write(f"#### {factor} Distribution in Predictions")
            num_unique = len(data[factor].unique())
            fig, ax = plt.subplots(figsize=(max(12, num_unique * 2), 8))
            sns.countplot(x=factor, hue="Prediction", data=data, ax=ax)
            ax.set_title(f"Depression Predictions by {factor}", fontsize=8)
            ax.set_xlabel(factor, fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
            ax3.set_xticklabels(data["Gender"].unique(), rotation=45, ha='right', fontsize=14)

            plt.subplots_adjust(bottom=0.2)
            plt.tight_layout(pad=3.0)
            st.pyplot(fig)

            factor_stats = data.groupby(factor)["Prediction"].mean().reset_index()
            factor_stats.columns = [factor, "Depression Prediction Rate"]
            fig_factor, ax_factor = plt.subplots(figsize=(12, 8))
            sns.barplot(x=factor, y="Depression Prediction Rate", data=factor_stats, ax=ax_factor)
            ax_factor.set_title(f"Depression Prediction Rate by {factor}", fontsize=8)
            ax_factor.set_xlabel(factor, fontsize=8)
            ax_factor.set_ylabel("Prediction Rate", fontsize=8)
            ax3.set_xticklabels(data["Gender"].unique(), rotation=45, ha='right', fontsize=8)

            plt.ylim(0, 1)
            plt.tight_layout(pad=3.0)
            st.pyplot(fig_factor)
            st.dataframe(factor_stats)

elif page == "Help":
    st.header("Help & Guidelines")
    st.markdown("""
        ## Mental Health Depression Prediction Application: User Guide
        
        🧠 About This App

        This application predicts depression based on survey data, using a neural network model. It supports batch predictions via CSV uploads and individual predictions via manual input, with visualizations and bias evaluations to interpret results.
        The app is designed to be user-friendly, with clear navigation and informative outputs. Below are the key features and instructions for using the app effectively.
        This Mental Health Prediction App uses AI to analyze survey responses and predict the likelihood of depression.
        It aims to promote early awareness and support mental well-being using technology.
        
        📋 How to Use
        
        Upload Data- You can upload a CSV file containing survey responses, or manually enter data in the form.
        
        Run Prediction- Click the "Predict" button to get results, including depression risk level and confidence score.
        
        View Insights- Check the visualizations for correlation, feature importance, and distribution of predictions.
        
        Evaluation- Metrics include Accuracy, Precision, Recall, F1-score, and more.
        
        🔐 Data Privacy
        
        Your privacy is our priority. The app does not store or share any personal data. All processing is done locally on your device.
        Your data is not stored or shared. The app processes data locally and does not retain any personal information.
        All processing is done **locally** – no data is uploaded or stored externally.
        This is a research prototype and not a substitute for professional mental health diagnosis.
        The app is designed for educational purposes and should not be used as a diagnostic tool.
        The app is not a substitute for professional medical advice, diagnosis, or treatment.
        
        
        💡 Tips for Best Results
        
        - Ensure your CSV file has the correct format and includes all necessary columns.
        - Use the manual entry form for quick predictions without uploading files.
        - Check the visualizations to understand the model's predictions and biases.
        - For best results, ensure your data is clean and well-structured.
        - Use the manual entry form for quick predictions without uploading files.
        - Ensure the input data matches the required format
        - Use realistic survey data – don't enter fake or incomplete info
        - Always interpret predictions with caution and context
        
        
        📌 Note: This app is designed to raise awareness, not replace clinical judgment. Please consult professionals for medical advice.

        
        ### 1. Overview
        The app is designed to:
        - Predict depression using features like age, gender, city, sleep duration, and more.
        - Handle both professional and student roles, with student-specific features (e.g., Academic Pressure, CGPA) if available in the training data.
        - Provide insights through visualizations and bias evaluations.
        - Ensure robust predictions with preprocessing (StandardScaler, SMOTE) and feature engineering (e.g., Age_City interaction).

        ### 2. Prerequisites
        Before running the app, ensure the following:
        - **System Requirements**:
          - Python 3.8 or higher.
          - A Windows/Linux/macOS system with at least 4GB RAM.
        - **Dependencies**:
          Install required packages using the requirements.txt file:
          
          """)
        
    
    # =========================== Help & Footer =============================
with st.sidebar:
    st.markdown("---")
    st.markdown("""
    ### How to use:
    1. Upload your mental health survey data in CSV format, or use manual form input
    2. Click 'Predict' to get the likelihood of depression
    3. View prediction confidence score and decision insights
    4. Use the Insights tab to explore visualizations (e.g., correlation heatmap, distributions)
    5. Go to Bias Evaluation to assess model fairness and accuracy if ground truth is provided
    
    ### About the model:
    This application uses a trained Deep Neural Network (DNN) built with TensorFlow/Keras, optimized using:
    - RandomOverSampler for class balance
    - RandomForestClassifier for top feature selection
    - Batch Normalization, Dropout, Early Stopping for robust learning

    Evaluation metrics include:
    - Accuracy, Precision, Recall, F1-score
    - Confusion matrix and Bias dashboard
    - Optional BLEU/ROUGE/Embedding metrics (coming soon)

    This project is part of a health-tech AI initiative designed to support early screening for depression risk.\n
    Your privacy is respected – no data is stored.\n
    App is currently in beta. Continuous improvements underway!\n
    Stay healthy. Stay aware. Mental health matters ❤️
    
    """)


