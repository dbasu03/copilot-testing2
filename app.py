import streamlit as st
from sentence_transformers import SentenceTransformer,util

def main():
    st.title('Simple Streamlit App')
    
    # User input
    user_input = st.text_input("Enter some text:")
    
    if user_input:
        # Display user input
        st.write(f"You entered: {user_input}")

if __name__ == "__main__":
    main()
