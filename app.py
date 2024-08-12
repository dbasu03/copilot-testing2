import streamlit as st
from sentence_transformers import SentenceTransformer

def main():
    # Initialize the SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    st.title('Text Embedding Generator')

    # User input
    user_input = st.text_input("Enter some text:")

    if user_input:
        # Generate embedding
        embedding = model.encode(user_input)
        
        # Display the embedding
        st.write("Text Embedding:")
        st.write(embedding)

if __name__ == "__main__":
    main()
