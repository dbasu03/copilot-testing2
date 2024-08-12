import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Convert to pandas DataFrame for easier manipulation
df = pd.DataFrame(data=X, columns=iris.feature_names)
df['target'] = y

# Print dataset description
print("Dataset Description:")
print(iris.DESCR)

# Print the first few rows of the dataset
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Print the target labels for the first 5 rows
print("\nTarget labels for the first 5 rows:")
print(df['target'].head())

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train a RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"\nModel Accuracy: {accuracy:.2f}")

# Convert predictions to pandas DataFrame for easier viewing
predictions_df = pd.DataFrame(data={'Actual': y_test, 'Predicted': predictions})

# Print a few rows of the predictions DataFrame
print("\nSample of Predictions:")
print(predictions_df.head())
