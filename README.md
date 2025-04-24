# 🧠 Mental Health Prediction

An AI-powered web application that predicts the likelihood of depression based on mental health survey responses. The app provides real-time predictions, data preprocessing, and insightful visualizations to assist healthcare professionals and individuals in understanding mental health risks.

---

## 🚀 Features

- **📁 Upload Data**  
  Upload survey data in CSV format for batch analysis and depression prediction.

- **📝 Manual Entry**  
  Input individual responses manually to receive real-time depression risk predictions.

- **🧹 Preprocessing & Cleaning**  
  Automatically cleans and preprocesses data using predefined pipelines.

- **🤖 AI-Powered Predictions**  
  Utilizes a Neural Network built with TensorFlow/Keras for accurate depression risk prediction.

- **📊 Data Visualizations**  
  Generates interactive plots like correlation heatmaps and depression distribution charts.

---

## 🧱 Project Structure

📦 Mental Health Prediction
┃ 
┣ 📂 Model 
┃ ┣ 📜 neural_network.keras **->** Trained TensorFlow model 
┃ ┣ 📜 preprocessor.pkl **->** Preprocessing pipeline 
┃ ┣ 📜 cleaning.pkl **->** Data cleaning logic 
┃
┃ ┣ 📂 Data 
┃ ┣ 📜 train.csv **->** Raw training dataset 
┃ ┣ 📜 test.csv **->** Raw test dataset
┃ ┣ 📜 sample_submission.csv **->** Sample survey dataset 
┃
┃ ┣ 📂 Dataset 
┃ ┣ 📜 cleaned_data.csv **->** Cleaned version of raw dataset
┃ ┣ 📜 preprocessor_data.csv **->** Preprocessed version of dataset
┃ ┣ 📂 Scripts
┃ ┣ 📜 data_understanding.ipynb **->** Dataset characteristics analysis 
┃ ┣ 📜 data_processing.ipynb **->** Data preparation for modeling
┃ ┣ 📜 EDA.ipynb **->** Exploratory data analysis and visualizations
┃ ┣ 📜 model_building.ipynb **->** Model training and evaluation
┣ 📜 Streamlit.py **->** Streamlit web application 
┣ 📜 requirements.txt **->** Python dependencies 
┣ 📜 README.md **->** Project documentation


---

## 📈 Visualizations & Insights

- **🧬 Correlation Heatmaps**  
  Understand the relationships between features and depression indicators.

- **📉 Depression Distribution**  
  Visual representation of how depression cases are spread across the dataset.

- **⚡ Real-time Predictions**  
  Instant results when users manually input their responses.

---

## 🛠️ Technologies Used

- [Streamlit](https://streamlit.io/) – For building the interactive web application.
- [TensorFlow/Keras](https://www.tensorflow.org/) – Neural Network framework for predictions.
- [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) – Data manipulation and transformation.
- [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) – Data visualization libraries.

---

## ▶️ How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/mental-health-prediction.git
    cd mental-health-prediction
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run Streamlit.py
    ```

---

## 🙌 Acknowledgments

This project aims to support mental health awareness by leveraging AI tools for early detection and intervention.

---


