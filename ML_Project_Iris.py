
### Step 1: Data Preparation

# 1.1. Dataset Selection
# We will use the Iris dataset, which is a simple and commonly used dataset in machine learning.

from sklearn.datasets import load_iris
import pandas as pd

# Load the dataset
iris = load_iris()

# Create a DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Display the first few rows of the DataFrame
print(df.head())

# 1.2. Data Preprocessing
# We will check for missing values, normalize the features, and split the data into training and testing sets.

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Check for missing values
print(df.isnull().sum())

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(df.iloc[:, :-1])  # Normalizing features
y = df['species']  # Target variable

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

### Step 2: Model Selection and Training

# 2.1. Model Selection
# We will start with a Decision Tree Classifier.

from sklearn.tree import DecisionTreeClassifier

# Initialize the model
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# 2.2. Model Evaluation
# Evaluate the model's performance using accuracy as a metric.

from sklearn.metrics import accuracy_score, confusion_matrix

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

### Step 3: Prediction on New Data

# Let's use our trained model to predict the species of a new iris sample.

import numpy as np

# New sample: [sepal length, sepal width, petal length, petal width]
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])
new_sample_scaled = scaler.transform(new_sample)

# Predict the species
predicted_species = model.predict(new_sample_scaled)
print(f"Predicted species: {iris.target_names[predicted_species][0]}")

### Optional Features

# Visualization

# 1. Confusion Matrix Visualization:
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 2. Feature Importance:
# Plotting feature importance
importances = model.feature_importances_
features = iris.feature_names

plt.figure(figsize=(8, 6))
plt.barh(features, importances, color='teal')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance in Decision Tree')
plt.show()

# Experiment: Trying a Different Model
# We will try a Random Forest Classifier and compare the results.

from sklearn.ensemble import RandomForestClassifier

# Initialize the model
rf_model = RandomForestClassifier(random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Evaluate the model
rf_y_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print(f"Random Forest Model Accuracy: {rf_accuracy * 100:.2f}%")

# Deployment (Optional)
# If you wish to deploy the model as a web app, tools like Streamlit make it easy.

# 1. Install Streamlit: pip install streamlit
# 2. Create a Simple App:

import streamlit as st

st.title("Iris Flower Species Prediction")
st.write("Enter the features of the iris flower below:")

# Inputs
sepal_length = st.number_input("Sepal Length", 0.0, 10.0, step=0.1)
sepal_width = st.number_input("Sepal Width", 0.0, 10.0, step=0.1)
petal_length = st.number_input("Petal Length", 0.0, 10.0, step=0.1)
petal_width = st.number_input("Petal Width", 0.0, 10.0, step=0.1)

if st.button("Predict"):
    # Prepare the input
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    sample_scaled = scaler.transform(sample)
    
    # Make prediction
    prediction = model.predict(sample_scaled)
    species = iris.target_names[prediction][0]
    
    st.write(f"The predicted species is: **{species}**")

# 3. Run the App: streamlit run app.py
