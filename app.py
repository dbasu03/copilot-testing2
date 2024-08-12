# app.py

import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd

# Load a pre-trained model from sentence-transformers
@st.cache(allow_output_mutation=True)
def load_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

# Load model
model = load_model()

# Streamlit app
def main():
    st.title('Text Similarity with Sentence Transformers')
    st.write('Enter a piece of text to calculate its similarity with texts from a CSV file:')
    
    # Upload CSV file
    csv_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    if csv_file is not None:
        # Load CSV file
        df = pd.read_csv(csv_file)
        
        # Check if the CSV has a 'text' column
        if 'Workflow' not in df.columns:
            st.error("CSV file must contain a 'text' column.")
            return
        
        # Display CSV data
        st.write("CSV Data:")
        st.dataframe(df)

        # Text input
        user_text = st.text_area("Input Text", "Type your text here...")

        if st.button('Calculate Similarity'):
            if user_text:
                # Compute embeddings
                user_embedding = model.encode(user_text, convert_to_tensor=True)
                
                # Compute embeddings for each text in the CSV
                df['embedding'] = df['Workflow'].apply(lambda x: model.encode(x, convert_to_tensor=True))
                
                # Calculate cosine similarity for each row
                df['similarity'] = df['embedding'].apply(lambda emb: util.pytorch_cos_sim(user_embedding, emb).item())
                
                # Display results
                st.write("Similarity Scores:")
                st.dataframe(df[['Workflow', 'similarity']].sort_values(by='similarity', ascending=False))
            else:
                st.write("Please enter some text.")
    else:
        st.write("Please upload a CSV file.")

if __name__ == "__main__":
    main()
