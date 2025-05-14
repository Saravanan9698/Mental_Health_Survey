# Mental Health Prediction

#### An AI-powered web application that predicts the likelihood of depression based on survey responses.  

## Features  

#####  *Upload Data* –  Upload survey data in CSV format for analysis and prediction.  
#####  *Manual Entry* –  Manually input survey responses and get depression risk predictions.  
#####  *Preprocessing & Cleaning* – Automatically cleans and preprocesses data.  
#####  *AI-Powered Predictions* – Uses a Neural Network model to predict depression risk.  
#####  *Data Visualizations* – Generates insights with correlation heatmaps and depression distribution charts.  

## Project Structure  
📦 Mental Health Prediction  
┃   
┣ 📂 **Model**   
┃  ┣ 📜 `cleaning.pkl` -> *Data cleaning logic* 
┃  ┣ 📜 `preprocessor.pkl` -> *Preprocessing pipeline*  
┃  ┣ 📜 `neural_network.keras` -> *Trained TensorFlow Model* 
┃  
┣ 📂 **Mental-Health-Survey-Dataset**   
┃  ┣ 📜 `sample_submission.csv` -> *Sample Survey dataset*
┃  ┣ 📜 `test.csv` -> *Test dataset*
┃  ┣ 📜 `train.csv` -> *Train dataset*
┃   
┣ 📂 **Research_Data**   
┃  ┣ 📜 `cleaned_data.csv` -> *Cleaned Dataset*  
┃  ┣ 📜 `preprocessed_data.csv` -> *Preprocessed Dataset*    
┃  
┣ 📂 **Scripts**   
┃  ┣ 📜 `Data_Cleaning.ipynb` -> *Helps in Finding and Understanding Dataset's Characteristic*  
┃  ┣ 📜 `Data_Preprocessing.ipynb` -> *Prepares Data for Model Building and Prediction*   
┃  ┣ 📜 `EDA.ipynb` -> *Visualizes data for the understanding*  
┃  ┣ 📜 `Model_Building.ipynb` -> *Builds Neural Networks Model and Prepares for the Prediction*  
┃  
┣ 📜 Streamlit.py -> *Streamlit Application*  
┣ 📜 requirements.txt -> *Python dependencies*    
┣ 📜 README.md -> *Documentation*


## Visualizations & Insights  
#####  *Correlation Heatmaps* – Find relationships between features.  
#####  *Depression Distribution* – See how depression cases are distributed.  
#####  *Real-time Predictions* – Get instant results for manual data entry.


## Technologies Used  
#####  *Streamlit* – Web application framework  
#####  *TensorFlow/Keras* – Neural Network for predictions  
#####  *Pandas & NumPy* – Data processing and transformations  
#####  *Matplotlib & Seaborn* – Visualizations  
