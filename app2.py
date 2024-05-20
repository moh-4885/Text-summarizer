import streamlit as st
from model import Summarizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import pipeline

summarizer=Summarizer()



# Define the Streamlit app
def main():
    st.title(":red[Text] Summarization")

    # Text input for user
    input_text = st.text_area("Enter your text here:")
        # Generate summary when button is clicked
    if st.button("Generate Summary"):
            st.spinner("Summarizing...")

            model_name = "t5-base"  # Adjust model name as needed
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(model_name)

            # text = "This is a long piece of text that needs to be summarized."
            input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

# Generate the summary
            summary_ids = model.generate(input_ids, max_length=100, min_length=10, length_penalty=2.0, num_beams=4, early_stopping=True)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            st.write("Summary:", summary)

if __name__ == "__main__":
    main()
