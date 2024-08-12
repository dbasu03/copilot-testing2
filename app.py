import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
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

def build_nn_model(embeddings, n_neighbors=10):
    nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    nn_model.fit(embeddings)
    return nn_model

def search_nn_model(nn_model, input_embedding, k):
    distances, indices = nn_model.kneighbors([input_embedding], n_neighbors=k)
    return distances, indices

# Streamlit app code

@st.cache
def initialize():
    df = load_data(DATA_FILE_PATH)
    embeddings = create_embeddings(df['Workflow'].tolist())
    nn_model = build_nn_model(embeddings)
    return df, nn_model

df, nn_model = initialize()

st.title('Workflow Similarity Search')

user_input = st.text_input("Enter your query:")

if user_input:
    # Create embedding for user input
    input_embedding = create_embeddings([user_input])[0]

    # Perform similarity search
    k = 10
    distances, indices = search_nn_model(nn_model, input_embedding, k)

    # Display results
    st.write("Matching Workflows:")
    matching_workflows = df.iloc[indices[0]]['Workflow'].tolist()
    for workflow in matching_workflows:
        st.write(workflow)
