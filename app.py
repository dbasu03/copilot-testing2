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

# Print dataset
print("Small Dataset:")
print(df)

# Input example
input_feature1 = float(input("Enter a value for Feature1: "))
input_feature2 = float(input("Enter a value for Feature2: "))
input_feature3 = float(input("Enter a value for Feature3: "))
input_feature4 = float(input("Enter a value for Feature4: "))

# Prepare features and target for training
X = df[['Feature1', 'Feature2', 'Feature3', 'Feature4']].values
y = df['Target'].values

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

# Predict for the user input
user_input = np.array([[input_feature1, input_feature2, input_feature3, input_feature4]])
user_prediction = model.predict(user_input)

print(f"\nPrediction for input features {user_input}: {user_prediction[0]}")
