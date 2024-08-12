import streamlit as st
import pandas as pd
import numpy as np
from annoy import AnnoyIndex
import tensorflow_hub as hub

# Constants
DATA_FILE_PATH = 'OrderedWorkflows.csv'
ANNOY_INDEX_FILE_PATH = 'annoy_index.ann'

# Load the Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Functions

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def create_embeddings(text_list):
    embeddings = embed(text_list).numpy()
    return embeddings

def build_annoy_index(embeddings, num_trees=10):
    dimension = embeddings.shape[1]
    annoy_index = AnnoyIndex(dimension, metric='angular')
    for i, embedding in enumerate(embeddings):
        annoy_index.add_item(i, embedding)
    annoy_index.build(num_trees)
    annoy_index.save(ANNOY_INDEX_FILE_PATH)
    return annoy_index

def load_annoy_index(embedding_dim):
    annoy_index = AnnoyIndex(embedding_dim, metric='angular')
    annoy_index.load(ANNOY_INDEX_FILE_PATH)
    return annoy_index

def search_annoy_index(annoy_index, input_embedding, k):
    indices = annoy_index.get_nns_by_vector(input_embedding, k, include_distances=True)
    return indices

# Streamlit app code

@st.cache
def initialize():
    df = load_data(DATA_FILE_PATH)
    embeddings = create_embeddings(df['Workflow'].tolist())
    annoy_index = build_annoy_index(embeddings)
    return df, annoy_index

df, annoy_index = initialize()

st.title('Workflow Similarity Search')

user_input = st.text_input("Enter your query:")

if user_input:
    # Create embedding for user input
    input_embedding = create_embeddings([user_input])[0]

    # Perform similarity search
    k = 10
    indices = search_annoy_index(annoy_index, input_embedding, k)

    # Display results
    st.write("Matching Workflows:")
    matching_workflows = df.iloc[indices[0]]['Workflow'].tolist()
    for workflow in matching_workflows:
        st.write(workflow)
