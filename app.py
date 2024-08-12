import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple model
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Streamlit app
st.title("Iris Dataset Classifier")

st.write(f"Accuracy of the model: {accuracy:.2f}")

# User input for prediction
sepal_length = st.slider("Sepal Length", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.slider("Sepal Width", min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.slider("Petal Length", min_value=0.0, max_value=10.0, value=4.0)
petal_width = st.slider("Petal Width", min_value=0.0, max_value=10.0, value=1.0)

# Making a prediction based on user input
user_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = clf.predict(user_data)
prediction_proba = clf.predict_proba(user_data)

st.write(f"Predicted class: {iris.target_names[prediction][0]}")
st.write(f"Prediction probability: {prediction_proba[0]}")
