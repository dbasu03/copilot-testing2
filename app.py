import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np
import io

# Constants
INDEX_FILE_PATH = 'faiss_index.index'

# Functions from pyfile.py

def create_embeddings(text_list):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text_list)
    return embeddings

def build_faiss_index(embeddings):
    embeddings_np = np.array(embeddings)
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    return index

def save_faiss_index(index, index_file_path):
    faiss.write_index(index, index_file_path)

def load_faiss_index(index_file_path):
    return faiss.read_index(index_file_path)

def search_index(index, input_embedding, k):
    distances, indices = index.search(np.array([input_embedding]), k)
    return distances, indices

# Main function
def main():
    st.title('Workflow Similarity Search')

    # File uploader widget
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

    if uploaded_file:
        # Load data from uploaded file
        df = pd.read_csv(uploaded_file)
        
        if 'Workflow' not in df.columns:
            st.error("The uploaded file must contain a 'Workflow' column.")
            return
        
        embeddings = create_embeddings(df['Workflow'].tolist())
        index = build_faiss_index(embeddings)
        save_faiss_index(index, INDEX_FILE_PATH)

        st.write("File uploaded and index created successfully!")

        user_input = st.text_input("Enter your query:")

        if user_input:
            # Create embedding for user input
            input_embedding = create_embeddings([user_input])[0]

            # Perform similarity search
            k = 10
            distances, indices = search_index(index, input_embedding, k)

            # Display results
            st.write("Matching Workflows:")
            matching_workflows = df.iloc[indices[0]]['Workflow'].tolist()
            for workflow in matching_workflows:
                st.write(workflow)
    else:
        st.info("Please upload a CSV file.")

# Run the app
if __name__ == "__main__":
    main()
