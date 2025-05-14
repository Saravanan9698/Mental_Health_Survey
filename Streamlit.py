# Optimized Streamlit App for Mental Health Depression Prediction


# Import necessary libraries
import os
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
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2


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
image_path = r"D:/Projects/Mini_Projects/Mental_Health_Survey/Image/Neural_Networks.jpg"
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
def load_pickle(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading pickle file: {str(e)}")
        return pd.DataFrame()

@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model(r"D:\Projects\Mini_Projects\Mental_Health_Survey\Model\neural_network.keras")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_resource
def load_cleaning():
    return load_pickle(r"D:\Projects\Mini_Projects\Mental_Health_Survey\Model\cleaning.pkl")

def balance_data(data, target_col, stratify_col=None):
    if target_col not in data.columns:
        return data
    X = data.drop(columns=[target_col])
    y = data[target_col]
    smote = SMOTE(random_state=42)
    if stratify_col and stratify_col in data.columns:
        group_counts = data.groupby([stratify_col, target_col]).size().unstack(fill_value=0)
        st.write("Data distribution by", stratify_col, "and", target_col, ":\n", group_counts)
        X_res, y_res = smote.fit_resample(X, y)
    else:
        X_res, y_res = smote.fit_resample(X, y)
    return pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=target_col)], axis=1)

def preprocess_input_data(input_data, preprocessor, role):
    input_data = input_data.copy()
    # Add Working Professional or Student
    input_data["Working Professional or Student"] = role
    # Compute engineered features
    city_map = {"Mumbai": 1, "Delhi": 2, "Chennai": 3, "Unknown": 0}
    if "Age" in input_data and "City" in input_data:
        input_data["Age_City"] = input_data["Age"] * city_map.get(input_data["City"], 0)
        input_data["City_Financial_Stress"] = input_data.get("Financial Stress", 0) * city_map.get(input_data["City"], 0)
    else:
        input_data["Age_City"] = 0
        input_data["City_Financial_Stress"] = 0
    # Handle student-specific features
    student_features = ["Academic Pressure", "Degree"]
    if role == "Professional":
        for feature in student_features:
            input_data[feature] = 0
    expected_features = preprocessor.numerical_features + [col for col in preprocessor.categorical_features if col in preprocessor.encoders]
    for feature in expected_features:
        if feature not in input_data:
            input_data[feature] = 0
    return input_data

def predict_depression(input_data, model, preprocessor, role):
    try:
        input_data = preprocess_input_data(input_data, preprocessor, role)
        processed = preprocessor.transform(pd.DataFrame([input_data]))
        processed = np.pad(processed, ((0, 0), (0, max(0, 13 - processed.shape[1]))), mode='constant')
        processed = processed[:, :13]
        output = model.predict(processed, verbose=0)
        label = "Depression Detected" if output[0] > 0.5 else "No Depression"
        return label, output[0][0]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}.")
        return "Error", 0.0

def retrain_model(data, model_path, cleaning_path):
    try:
        preprocessor = MentalHealthPreprocessor()
        processed = preprocessor.fit(data.copy())
        model = Sequential([
            Dense(64, activation='relu', input_shape=(13,), kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(processed.drop(columns=["Prediction"]), processed["Prediction"], 
                 epochs=100, batch_size=32, validation_split=0.2, verbose=0)
        model.save(model_path)
        with open(cleaning_path, "wb") as f:
            pickle.dump(data, f)
        st.success("Model retrained and saved successfully.")
    except Exception as e:
        st.error(f"Retraining failed: {str(e)}")

def format_labels(labels, max_length=None):
    num_labels = len(labels)
    if max_length is None:
        max_length = 15 if num_labels <= 5 else 10 if num_labels <= 10 else 7
    formatted = [str(label).strip()[:max_length] + ("…" if len(str(label).strip()) > max_length else "") for label in labels]
    return formatted
    
    ### Create the sidebar navigation and footer 
    
    ### Analysis and visualization with Separate Pages with the Help of Navigator and Footer

# Sidebar navigation and footer

st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Page", ["Home", "Upload Data", "Manual Entry", "Visualizations", "Bias Evaluation", "Help"])
st.sidebar.markdown("---")
st.sidebar.markdown("""
    ### How to Use
    1. Upload survey data (CSV) or use manual input.
    2. Click 'Predict' for depression likelihood.
    3. View confidence scores and insights.
    4. Explore visualizations in the Insights tab.
    5. Assess fairness in Bias Evaluation.

    ### About the Model
    Deep Neural Network (TensorFlow/Keras) with:
    - SMOTE for class balance
    - L2 regularization, Batch Normalization, Dropout
    - Metrics: Accuracy, Precision, Recall, F1, ROC-AUC

    🔒 *Privacy*: Local processing, no data stored.  
    🛠 *Status*: Beta, actively improved.  
    ❤ *Mental Health Matters*: Seek professional help if needed.
""")

# Initialize session state

if "predicted_data" not in st.session_state:
    st.session_state.predicted_data = pd.DataFrame()
    
    ### Home Page

if page == "Home":
    st.title("🧠 Mental Health Depression Prediction")
    st.markdown("""
        An AI-powered tool to predict depression risk from survey data, promoting early awareness and mental well-being.
        Analyze factors like age, financial stress, and sleep habits with a robust neural network.
    """)
    st.info("*Disclaimer*: This app is for research and education only, not a substitute for professional diagnosis.")

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Guidelines", "Use Cases", "Standards"])
    
    with tab1:
        st.header("📋 Overview")
        st.markdown("""
            This application leverages a deep neural network to predict depression risk based on survey data, 
            supporting both batch CSV uploads and manual input. Key features include:
            - *Prediction*: Batch or individual depression risk assessment.
            - *Visualizations*: Insights into correlations, age, and financial stress distributions.
            - *Bias Evaluation*: Fairness analysis across demographics.
            - *Robustness*: SMOTE, feature engineering, and regularization for reliable predictions.
            
            *Goal*: Empower researchers, educators, and communities with data-driven mental health insights.
        """)

    with tab2:
        st.header("🔧 Guidelines for New Users")
        st.markdown("""
            ### Accessing the Project
            - *Clone Repository* (if hosted, e.g., GitHub):
              bash
              git clone https://github.com/your-repo/mental-health-survey.git
              cd mental-health-survey
              
            - *Download*: Get the project zip or copy files to:
              
              D:\\Projects\\Mini_Projects\\Mental_Health_Survey\\
              
            - *Verify Files*:
              - Model\\cleaning.pkl: Training data.
              - Model\\neural_network.keras: Trained model.
              - Image\\Neural_Networks.jpg: Background image.

            ### Setting Up
            - *Install Python*: Use Python 3.8+.
            - *Create Virtual Environment*:
              bash
              cd D:\\Projects\\Mini_Projects\\Mental_Health_Survey
              python -m venv venv
              venv\\Scripts\\activate
              
            - *Install Dependencies*:
              bash
              pip install -r requirements.txt
              
              requirements.txt:
              
              numpy==1.26.4
              pandas==2.2.2
              seaborn==0.13.2
              streamlit==1.39.0
              tensorflow==2.17.0
              matplotlib==3.9.2
              scikit-learn==1.5.2
              imblearn==0.12.3
              
            - *Run the App*:
              bash
              streamlit run Streamlit.py
              
              Access at http://localhost:8501.

            ### Preparing Data
            - *CSV Format*: Match cleaning.pkl columns, e.g.:
              - Core: Gender, Age, City, Sleep Duration, Dietary Habits, Have you ever had suicidal thoughts ?, Work/Study Hours, Financial Stress, Family History of Mental Illness, Working Professional or Student.
              - Professional: Profession, Work Pressure, Job Satisfaction.
              - Student: Academic Pressure, CGPA, Study Satisfaction, Degree.
            - *Sample CSV*:
              csv
              Gender,Age,City,Sleep Duration,Dietary Habits,Have you ever had suicidal thoughts ?,Work/Study Hours,Financial Stress,Family History of Mental Illness,Working Professional or Student,Profession,Work Pressure,Job Satisfaction
              Male,30,Mumbai,6-8 hours,Healthy,No,8,2,No,Professional,Software Engineer,5,7
              Female,20,Delhi,Less than 5 hours,Unhealthy,Yes,10,4,Yes,Student,,,,
              
            - *Validation*:
              - Numerical: Non-negative (e.g., Age ≥ 0, Financial Stress 0–5).
              - Categorical: Match encoded values (e.g., Gender: Male/Female, City: Mumbai/Delhi/Chennai).
              - Use Unknown for new categories.
            - *Check Columns*:
              python
              cleaned = pd.read_pickle("Model/cleaning.pkl")
              print(cleaned.columns.tolist())
              
            - *Handle Missing*: The app fills missing features with defaults.

            ### Using the App
            - *Upload Data*: Batch predictions via CSV.
            - *Manual Entry*: Single predictions with professional/student roles.
            - *Visualizations*: Analyze correlations, distributions.
            - *Bias Evaluation*: Check fairness and performance.
            - *Download*: Save results as depression_predictions.csv.

            ### Contributing
            - *Report Issues*: Submit bugs via GitHub Issues.
            - *Add Features*: Propose visualizations, metrics, or preprocessing.
            - *Pull Requests*: Fork, modify, submit PRs with clear descriptions.
            - *Guidelines*:
              - Follow PEP 8.
              - Test changes locally.
              - Document in “Help” page.

            ### Best Practices
            - Include Working Professional or Student in inputs.
            - Use realistic data for accurate predictions.
            - Monitor debug logs for data issues.
            - Retrain model with new data (see “Help”).
        """)

    with tab3:
        st.header("🚀 Use Cases")
        st.markdown("""
            - *Research*: Study mental health trends across age, gender, or cities.
            - *Healthcare Support*: Identify at-risk individuals for clinical follow-up.
            - *Education*: Teach AI and mental health applications in classrooms.
            - *Community Health*: Support NGOs with targeted outreach programs.
            - *Personal Awareness*: Explore how lifestyle impacts mental health risk.
        """)

    with tab4:
        st.header("🏅 Industry Standards")
        st.markdown("""
            - *🔒 Data Privacy*:
              - Local processing; no data storage or sharing.
              - Aligns with GDPR, HIPAA principles.
            - *🤝 Ethics*:
              - Transparent predictions with confidence scores.
              - Non-diagnostic tool; professional consultation advised.
            - *🛠 AI Standards*:
              - SMOTE, L2 regularization, dropout for robustness.
              - Metrics: ROC-AUC, precision, recall, F1-score.
              - Feature engineering (e.g., Age_City) for accuracy.
            - *📈 Quality*:
              - Cross-validation for unseen data performance.
              - Regular retraining for updated datasets.
        """)
        st.warning("*Reminder*: Seek professional help for mental health concerns.")
        
        
        # Add a footer
        
        ### Uplaod Data Page

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
            st.success("Predictions completed successfully!")
            
            ### Manual Entry Page

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
        pred, conf = predict_depression(input_data, model, preprocessor, role)
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
        st.warning("No predicted data available. Run predictions in Upload Data.")
        st.stop()

    st.write("#### Predicted Data Overview")
    st.dataframe(st.session_state.predicted_data.head())

    numeric_data = st.session_state.predicted_data.select_dtypes(include=[np.number])
    if not numeric_data.empty:
        st.write("#### Correlation Heatmap")
        st.write("Numeric columns:", numeric_data.columns.tolist())
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", ax=ax, annot_kws={"size": 10}, fmt=".2f")
        plt.title("Correlation Heatmap of Numeric Features", fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout(pad=3.0)
        st.pyplot(fig)

        st.write("#### Depression Prediction Distribution")
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        sns.countplot(x="Prediction", data=st.session_state.predicted_data, ax=ax2)
        ax2.set_title("Distribution of Depression Predictions", fontsize=12)
        ax2.set_xlabel("Prediction", fontsize=12)
        ax2.set_ylabel("Count", fontsize=12)
        plt.xticks(fontsize=12)
        plt.tight_layout(pad=3.0)
        st.pyplot(fig2)

    if "Age" in st.session_state.predicted_data.columns and "Prediction" in st.session_state.predicted_data.columns:
        st.write("#### Age Distribution by Depression Prediction")
        data = st.session_state.predicted_data.copy()
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        sns.boxplot(x="Prediction", y="Age", data=data, ax=ax3)
        ax3.set_title("Age Distribution by Depression Prediction", fontsize=12)
        ax3.set_xlabel("Prediction (0: No Depression, 1: Depression Detected)", fontsize=12)
        ax3.set_ylabel("Age", fontsize=12)
        ax3.set_xticklabels(["No Depression", "Depression Detected"], fontsize=12)
        plt.tight_layout(pad=3.0)
        st.pyplot(fig3)

    if "Financial Stress" in st.session_state.predicted_data.columns and "Prediction" in st.session_state.predicted_data.columns:
        st.write("#### Financial Stress Distribution by Depression Prediction")
        data = st.session_state.predicted_data.copy()
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        sns.violinplot(x="Prediction", y="Financial Stress", data=data, ax=ax4)
        ax4.set_title("Financial Stress Distribution by Depression Prediction", fontsize=12)
        ax4.set_xlabel("Prediction (0: No Depression, 1: Depression Detected)", fontsize=12)
        ax4.set_ylabel("Financial Stress", fontsize=12)
        ax4.set_xticklabels(["No Depression", "Depression Detected"], fontsize=12)
        plt.tight_layout(pad=3.0)
        st.pyplot(fig4)

elif page == "Bias Evaluation":
    st.header("Bias Evaluation for Prediction Distribution")
    if st.session_state.predicted_data.empty:
        st.warning("No predicted data available. Run predictions in Upload Data.")
        st.stop()

    data = st.session_state.predicted_data.copy()
    data["City"] = data["City"].replace({np.nan: "Unknown", "Mumbay": "Mumbai"})

    if "Prediction" in data.columns and "Confidence" in data.columns:
        st.write("#### Model Performance Metrics")
        y_true = data["Prediction"]
        y_pred = (data["Confidence"] > 0.5).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_true, data["Confidence"])
        report = classification_report(y_pred, y_true, output_dict=True, zero_division=0)
        
        st.write(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}, ROC-AUC: {roc_auc:.2f}")
        st.write("### Classification Report")
        st.dataframe(pd.DataFrame(report).T)

        cm = confusion_matrix(y_true, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        ax_cm.set_title("Confusion Matrix", fontsize=12)
        ax_cm.set_xlabel("Predicted", fontsize=12)
        ax_cm.set_ylabel("True", fontsize=12)
        plt.tight_layout(pad=3.0)
        st.pyplot(fig_cm)

        st.write("#### ROC Curve")
        fpr, tpr, _ = roc_curve(y_true, data["Confidence"])
        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
        ax_roc.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
        ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax_roc.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=12)
        ax_roc.set_xlabel("False Positive Rate", fontsize=12)
        ax_roc.set_ylabel("True Positive Rate", fontsize=12)
        ax_roc.legend(fontsize=12)
        plt.tight_layout(pad=3.0)
        st.pyplot(fig_roc)

        st.write("#### Prediction Probability Distribution")
        fig_prob, ax_prob = plt.subplots(figsize=(12, 8))
        sns.histplot(data=data, x="Confidence", hue="Prediction", bins=20, ax=ax_prob)
        ax_prob.set_title("Distribution of Prediction Probabilities", fontsize=12)
        ax_prob.set_xlabel("Confidence Score", fontsize=12)
        ax_prob.set_ylabel("Count", fontsize=12)
        plt.tight_layout(pad=3.0)
        st.pyplot(fig_prob)

        # Cross-validation estimate
        preprocessor = MentalHealthPreprocessor()
        processed_data = preprocessor.fit(data.copy())
        rf = RandomForestClassifier(random_state=42)
        cv_scores = cross_val_score(rf, processed_data.drop(columns=["Prediction", "Confidence"]), y_true, cv=5, scoring='roc_auc')
        st.write(f"Cross-Validation ROC-AUC (5-fold): Mean = {cv_scores.mean():.2f}, Std = {cv_scores.std():.2f}")

        # Feature importance
        rf.fit(processed_data.drop(columns=["Prediction", "Confidence"]), y_true)
        importance = pd.DataFrame({
            "Feature": processed_data.drop(columns=["Prediction", "Confidence"]).columns,
            "Importance": rf.feature_importances_
        }).sort_values("Importance", ascending=False)
        st.write("#### Feature Importance")
        st.dataframe(importance)

        # Fairness analysis
        if "Gender" in data.columns:
            st.write("#### Fairness Analysis: Gender")
            parity = data.groupby("Gender")["Prediction"].mean().reset_index()
            parity.columns = ["Gender", "Prediction Rate"]
            recall_by_gender = data.groupby("Gender").apply(
                lambda x: recall_score(x["Prediction"], (x["Confidence"] > 0.5).astype(int), zero_division=0)
            ).reset_index(name="Recall")
            st.write("*Demographic Parity* (Prediction Rate by Gender):")
            st.dataframe(parity)
            st.write("*Equal Opportunity* (Recall by Gender):")
            st.dataframe(recall_by_gender)

    if "Age" in data.columns and "Prediction" in data.columns:
        st.write("#### Age Distribution in Predictions")
        data["Age Group"] = pd.cut(data["Age"], bins=[17, 25, 35, 45, 55, 85], labels=["18-25", "26-35", "36-45", "46-55", "56+"])
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.countplot(x="Age Group", hue="Prediction", data=data, ax=ax)
        ax.set_title("Depression Predictions by Age Group", fontsize=12)
        ax.set_xlabel("Age Group", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.tight_layout(pad=3.0)
        st.pyplot(fig)

        age_stats = data.groupby("Age Group")["Prediction"].mean().reset_index()
        age_stats.columns = ["Age Group", "Depression Prediction Rate"]
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        sns.barplot(x="Age Group", y="Depression Prediction Rate", data=age_stats, ax=ax2)
        ax2.set_title("Depression Prediction Rate by Age Group", fontsize=12)
        ax2.set_xlabel("Age Group", fontsize=12)
        ax2.set_ylabel("Prediction Rate", fontsize=12)
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.tight_layout(pad=3.0)
        st.pyplot(fig2)
        st.dataframe(age_stats)

    if "Gender" in data.columns and "Prediction" in data.columns:
        st.write("#### Gender Distribution in Predictions")
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        sns.countplot(x="Gender", hue="Prediction", data=data, ax=ax3)
        ax3.set_title("Depression Predictions by Gender", fontsize=12)
        ax3.set_xlabel("Gender", fontsize=12)
        ax3.set_ylabel("Count", fontsize=12)
        ax3.set_xticklabels(format_labels(data["Gender"].unique()), rotation=45, ha='right', fontsize=12)
        plt.tight_layout(pad=3.0)
        st.pyplot(fig3)

        gender_stats = data.groupby("Gender")["Prediction"].mean().reset_index()
        gender_stats.columns = ["Gender", "Depression Prediction Rate"]
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        sns.barplot(x="Gender", y="Depression Prediction Rate", data=gender_stats, ax=ax4)
        ax4.set_title("Depression Prediction Rate by Gender", fontsize=12)
        ax4.set_xlabel("Gender", fontsize=12)
        ax4.set_ylabel("Prediction Rate", fontsize=12)
        ax4.set_xticklabels(format_labels(gender_stats["Gender"]), rotation=45, ha='right', fontsize=12)
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
            ax.set_title(f"Depression Predictions by {factor}", fontsize=12)
            ax.set_xlabel(factor, fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.set_xticklabels(format_labels(data[factor].unique()), rotation=45, ha='right', fontsize=12)
            plt.subplots_adjust(bottom=0.2)
            plt.tight_layout(pad=3.0)
            st.pyplot(fig)

            factor_stats = data.groupby(factor)["Prediction"].mean().reset_index()
            factor_stats.columns = [factor, "Depression Prediction Rate"]
            fig_factor, ax_factor = plt.subplots(figsize=(12, 8))
            sns.barplot(x=factor, y="Depression Prediction Rate", data=factor_stats, ax=ax_factor)
            ax_factor.set_title(f"Depression Prediction Rate by {factor}", fontsize=12)
            ax_factor.set_xlabel(factor, fontsize=12)
            ax_factor.set_ylabel("Prediction Rate", fontsize=12)
            ax_factor.set_xticklabels(format_labels(factor_stats[factor]), rotation=45, ha='right', fontsize=12)
            plt.ylim(0, 1)
            plt.tight_layout(pad=3.0)
            st.pyplot(fig_factor)
            st.dataframe(factor_stats)

elif page == "Help":
    st.header("🧠 Help & Guidelines")
    st.markdown("This AI-powered app predicts depression risk using survey data, offering batch predictions, manual input, visualizations, and fairness analysis.")
    
    with st.expander("📋 Overview"):
        st.markdown("""
            - **Prediction**: Neural network assesses depression risk.
            - **Features**: Age, gender, city, financial stress, sleep duration, role (Professional/Student), etc.
            - **Roles**: Supports Working Professional and Student inputs (e.g., Academic Pressure or Work Pressure).
            - **Insights**: Visualizations and bias evaluations for interpretability.
            - **Robustness**: Uses SMOTE, L2 regularization, and feature engineering (e.g., Age_City).
        """)

    with st.expander("🔧 Prerequisites"):
        st.markdown("""
            - **System**: Python 3.8+, 4GB RAM, Windows/Linux/macOS.
            - **Dependencies**:
              ```bash
              pip install -r requirements.txt
              ```
              **requirements.txt**:
              ```
              numpy==1.26.4
              pandas==2.2.2
              seaborn==0.13.2
              streamlit==1.39.0
              tensorflow==2.17.0
              matplotlib==3.9.2
              scikit-learn==1.5.2
              imblearn==0.12.3
              ```
            - **Files**:
              - `Model\\cleaning.pkl`
              - `Model\\neural_network.keras`
              - `Image\\Neural_Networks.jpg`
            - **Structure**:
              ```
              D:\\Projects\\Mini_Projects\\Mental_Health_Survey\\
              ├── venv\\
              │   └── Virtual environment
              ├── Scripts\\
              │   ├── Data_Cleaning.ipynb
              │   ├── Data_Preprocessing.ipynb
              │   ├── EDA.ipynb
              │   ├── Model_Building.ipynb
              ├── Image\\
              │   └── Neural_Networks.jpg
              ├── Model\\
              │   ├── cleaning.pkl
              │   ├── Preprocessor.pkl
              │   └── neural_network.keras
              ├── Research_Data\\
              │   ├── cleaned_data.ipynb
              │   ├── preprocessed_data.csv
              │   ├── test.csv
              │   ├── train.csv
              ├── requirements.txt
              ├── Streamlit.py
              ```
            - **Note**: Ensure Streamlit 1.39.0 compatibility with `streamlit --version`.
        """)
        if st.button("Check Dependencies"):
            st.write("Run `pip list` in your terminal to verify installed packages.")

    with st.expander("🚀 Running the App"):
        st.markdown("""
            ```bash
            cd D:\\Projects\\Mini_Projects\\Mental_Health_Survey
            venv\\Scripts\\activate
            streamlit run Streamlit.py
            ```
            Access: [http://localhost:8501](http://localhost:8501).
        """)

    with st.expander("📊 Navigation"):
        st.markdown("""
            Visit *Home* for detailed guidelines and use cases.
            - **Home**: App overview and instructions.
            - **Upload Data**: Batch predictions via CSV.
            - **Manual Entry**: Single predictions with Role (Professional/Student).
            - **Visualizations**: Insights on correlations, age, and financial stress.
            - **Bias Evaluation**: Fairness and performance metrics.
            - **Help**: This guide.
        """)

    with st.expander("🔍 Troubleshooting"):
        st.markdown("""
            - **Missing Features**:
              - Check features in `cleaning.pkl`:
                ```python
                import pandas as pd
                cleaned = pd.read_pickle("Model/cleaning.pkl")
                print(cleaned.columns.tolist())
                ```
              - Ensure CSV includes Role (Professional/Student) and matches `cleaning.pkl`.
              - For manual entry, select correct role and include all required fields.
            - **File Not Found**:
              - Verify paths for `cleaning.pkl`, `neural_network.keras`, and `Neural_Networks.jpg`.
            - **Prediction Errors**:
              - Check debug logs (e.g., "Processed rows").
              - Ensure input data includes Role and matches expected features (13 total).
              - Retrain model (see below).
            - **Visualization Issues**:
              - In `Streamlit.py` or visualization script, adjust figure size:
                ```python
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(max(12, num_unique * 3), 8))
                ```
        """)

    with st.expander("🛠 Retraining the Model"):
        st.markdown("""
            ```python
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
            from tensorflow.keras.regularizers import l2
            model = Sequential([
                Dense(64, activation='relu', input_shape=(13,), kernel_regularizer=l2(0.01)),
                BatchNormalization(),
                Dropout(0.3),
                Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
                BatchNormalization(),
                Dropout(0.3),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            # Train: model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
            model.save("Model/neural_network.keras")
            ```
            - Ensure `X_train` has 13 features, including Role.
            - Update `cleaning.pkl` with new data.
        """)

    with st.expander("💡 Tips"):
        st.markdown("""
            - Include all expected features in CSV, including Role (see *Home* > Guidelines).
            - Use realistic data for accurate predictions.
            - Monitor fairness metrics for biases.
            - Retrain model periodically with fresh data.
        """)

    with st.expander("📞 Support"):
        st.markdown("""
            - **Streamlit**: [https://docs.streamlit.io](https://docs.streamlit.io)
            - **TensorFlow**: [https://www.tensorflow.org/guide](https://www.tensorflow.org/guide)
            - **Issues**: Submit via GitHub (if applicable).
            - **❤ Mental Health Matters**: Consult professionals for medical advice.
        """)