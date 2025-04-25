#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import RandomOverSampler

from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')


# In[6]:


file_path = r"D:\Projects\Mini_Projects\Mental_Health_Survey\Research_Data\preprocessed_data.csv"


# In[7]:


data = pd.read_csv(file_path)


# In[8]:


data.head()


# In[9]:


data.shape


# In[10]:


data['Depression'].value_counts()


# In[11]:


val = data.drop('Depression', axis = 1)
tar = data['Depression']


# In[12]:


val = data.drop('Depression', axis = 1)
tar = data['Depression']


# In[13]:


# Imbalance

sampler = RandomOverSampler(random_state = 65)


# In[14]:


os_val, os_tar = sampler.fit_resample(val, tar)


# In[15]:


# feature selection

fs = RandomForestClassifier(n_estimators= 200, max_depth= 3, random_state= 65)

fs.fit(os_val, os_tar)


# In[16]:


s_col = pd.DataFrame({
    "col": os_val.columns,
    "score": fs.feature_importances_
}).sort_values('score', ascending = False). head(10)['col'].to_list()


# In[17]:


s_col


# In[18]:


# Train Test Split

tr_data, ts_data, tr_lab, ts_lab = train_test_split(os_val, os_tar, test_size = 0.2, random_state = 65)


# In[19]:


tr_data.shape, ts_data.shape, tr_lab.shape, ts_lab.shape


# In[20]:


model = Sequential([
    Dense(100, activation='relu', input_shape=(tr_data.shape[1],)),
    BatchNormalization(), 
    Dropout(0.3),  

    Dense(65, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(35, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(15, activation='relu'),
    Dropout(0.2),

    Dense(5, activation='relu'),

    Dense(1, activation='sigmoid')
])


# In[21]:


optimizer = Adam(learning_rate=0.001)  
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


# In[22]:


early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)


# In[23]:


history = model.fit(tr_data, tr_lab, validation_data=(ts_data, ts_lab),
                    epochs=50, batch_size=64, callbacks=[early_stopping], verbose=1)


# In[24]:


ts_pred = (model.predict(ts_data) > 0.5).astype(int)


# In[25]:


ts_probs = model.predict(ts_data).flatten()


# In[26]:


fpr, tpr, _ = roc_curve(ts_lab, ts_probs)


# In[27]:


auc_score = roc_auc_score(ts_lab, ts_probs)


# In[28]:


accuracy = accuracy_score(ts_lab, ts_pred)
precision = precision_score(ts_lab, ts_pred)
recall = recall_score(ts_lab, ts_pred)
f1 = f1_score(ts_lab, ts_pred)


# In[29]:


print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1-Score: {f1:.4f}")
print("\n =============== Classification Report: ===============")
print(f"\n {classification_report(ts_lab, ts_pred)}")


# In[30]:


plt.figure(figsize=(5, 4))
cm = confusion_matrix(ts_lab, ts_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
plt.title('Neural Network Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


# In[31]:


plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f"ROC AUC = {auc_score:.2f}", color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='red')
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve")
plt.legend()
plt.show()


# In[32]:


model.save('D:\\Projects\\Mini_Projects\\Mental_Health_Survey\\Model\\neural_network.keras')
print('Model Successfully Saved to \Models')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




