import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Small dataset example
data = {
    'Feature1': [5.1, 4.9, 4.7, 4.6, 5.0],
    'Feature2': [3.5, 3.0, 3.2, 3.1, 3.6],
    'Feature3': [1.4, 1.4, 1.3, 1.5, 1.4],
    'Feature4': [0.2, 0.2, 0.2, 0.2, 0.2],
    'Target': [0, 0, 0, 0, 0]  # Example target values
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Prepare features and target for training
X = df[['Feature1', 'Feature2', 'Feature3', 'Feature4']].values
y = df['Target'].values

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train a RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Streamlit app
st.title('Simple ML Model with User Input')

# Input fields for the features
input_feature1 = st.number_input("Enter a value for Feature1:", value=
