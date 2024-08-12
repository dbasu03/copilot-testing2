import streamlit as st
import pandas as pd
import numpy as np
import nmslib
import tensorflow_hub as hub

# Constants
DATA_FILE_PATH = 'OrderedWorkflows.csv'
NMSLIB_INDEX_FILE_PATH = 'nmslib_index.nms'

# Load the Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Functions

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def create_embeddings(text_list):
    embeddings = embed(text_list).numpy()
    return embeddings

def build_nmslib_index(embeddings):
    # Create NMSLIB index
    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.addDataPointBatch(embeddings)
    index.createIndex({'post': 2}, print_progress=True)
    index.saveIndex(NMSLIB_INDEX_FILE_PATH)
    return index

def load_nmslib_index(embedding_dim):
    # Load NMSLIB index
    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.loadIndex(NMSLIB_INDEX_FILE_PATH, load_data=True)
    return index

def search_nmslib_index(index, input_embedding, k):
    indices, distances = index.knnQuery(input_embedding, k=k)
    return distances, indices

# Streamlit app code

@st.cache
def initialize():
    df = load_data(DATA_FILE_PATH)
    embeddings = create_embeddings(df['Workflow'].tolist())
    index = build_nmslib_index(embeddings)
    return df, index

df, index = initialize()

st.title('Workflow Similarity Search')

user_input = st.text_input("Enter your query:")

if user_input:
    # Create embedding for user input
    input_embedding = create_embeddings([user_input])[0]

    # Perform similarity search
    k = 10
    distances, indices = search_nmslib_index(index, input_embedding, k)

    # Display results
    st.write("Matching Workflows:")
    matching_workflows = df.iloc[indices]['Workflow'].tolist()
    for workflow in matching_workflows:
        st.write(workflow)
