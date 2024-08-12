import streamlit as st
import pandas as pd
import numpy as np
from pynndescent import NNDescent
import tensorflow_hub as hub

# Constants
DATA_FILE_PATH = 'OrderedWorkflows.csv'

# Load the Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Functions

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def create_embeddings(text_list):
    embeddings = embed(text_list).numpy()
    return embeddings

def build_pynndescent_index(embeddings):
    # Create PyNNDescent index
    pynnd_index = NNDescent(embeddings, metric='cosine', n_neighbors=10)
    return pynnd_index

def search_pynndescent_index(pynnd_index, input_embedding, k):
    indices, distances = pynnd_index.query([input_embedding], k=k)
    return distances[0], indices[0]

# Streamlit app code

@st.cache
def initialize():
    df = load_data(DATA_FILE_PATH)
    embeddings = create_embeddings(df['Workflow'].tolist())
    pynnd_index = build_pynndescent_index(embeddings)
    return df, pynnd_index

df, pynnd_index = initialize()

st.title('Workflow Similarity Search')

user_input = st.text_input("Enter your query:")

if user_input:
    # Create embedding for user input
    input_embedding = create_embeddings([user_input])[0]

    # Perform similarity search
    k = 10
    distances, indices = search_pynndescent_index(pynnd_index, input_embedding, k)

    # Display results
    st.write("Matching Workflows:")
    matching_workflows = df.iloc[indices]['Workflow'].tolist()
    for workflow in matching_workflows:
        st.write(workflow)
