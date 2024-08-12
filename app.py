import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
# Title of the app
st.title('Simple Streamlit App with NumPy')

# Input fields for numbers
num1 = st.number_input("Enter the first number:", value=0)
num2 = st.number_input("Enter the second number:", value=0)

# Calculate the sum using NumPy
result = np.add(num1, num2)

# Display the result
st.write("The sum of the two numbers is:", result)
