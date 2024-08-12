import streamlit as st
import pandas as pd
import numpy as np

# Load the CSV file from the current directory
@st.cache_data
def load_data():
    data = pd.read_csv('sample_data.csv')
    return data

# Streamlit app
st.title("Simple CSV Viewer")

data = load_data()

st.write("Here's the data from the CSV file:")
st.dataframe(data)

