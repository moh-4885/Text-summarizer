import streamlit as st
from model import Summarizer
summarizer=Summarizer()



# Define the Streamlit app
def main():
    st.title(":red[Text] Summarization")

    # Text input for user
    input_text = st.text_area("Enter your text here:")
        # Generate summary when button is clicked
    if st.button("Generate Summary"):
            st.spinner("Summarizing...")
            # Tokenize the input text
            summary = summarizer.decode_sequence(input_text)
            st.write("Summary:", summary)

if __name__ == "__main__":
    main()
