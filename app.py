import streamlit as st
import pandas as pd
import numpy as np
import hnswlib
import tensorflow_hub as hub

# Constants
DATA_FILE_PATH = 'OrderedWorkflows.csv'
HNSW_INDEX_FILE_PATH = 'hnsw_index.bin'

# Load the Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Functions

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def create_embeddings(text_list):
    embeddings = embed(text_list).numpy()
    return embeddings

def build_hnsw_index(embeddings):
    dimension = embeddings.shape[1]
    num_elements = embeddings.shape[0]

    # Initialize HNSWlib index
    hnsw_index = hnswlib.Index(space='cosine', dim=dimension)
    hnsw_index.init_index(max_elements=num_elements, ef_construction=200, M=16)
    
    # Add items to the index
    hnsw_index.add_items(embeddings)
    
    # Save the index to a file
    hnsw_index.save_index(HNSW_INDEX_FILE_PATH)
    return hnsw_index

def load_hnsw_index(embedding_dim):
    hnsw_index = hnswlib.Index(space='cosine', dim
