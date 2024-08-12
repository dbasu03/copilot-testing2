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
input_feature1 = st.number_input("Enter a value for Feature1:", value=0.0)
input_feature2 = st.number_input("Enter a value for Feature2:", value=0.0)
input_feature3 = st.number_input("Enter a value for Feature3:", value=0.0)
input_feature4 = st.number_input("Enter a value for Feature4:", value=0.0)

# Predict for the user input
if st.button('Predict'):
    user_input = np.array([[input_feature1, input_feature2, input_feature3, input_feature4]])
    user_prediction = model.predict(user_input)
    st.write(f"Prediction for input features {user_input}: {user_prediction[0]}")

# Display dataset and model accuracy
st.write("Small Dataset:")
st.write(df)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, predictions)
st.write(f"Model Accuracy: {accuracy:.2f}")
