from preprocess import data_preprocess_pipeline

import streamlit as st
import pickle

# preprocess
# vectorize 
# predict using the model
# display

try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model files not found. Please check your file paths.")
    st.stop()

st.title("Email/SMS Spam Classifier")
st.write("Enter a message below to check if it's spam or not.")

input_text = st.text_area("Message:")

if st.button('Predict'):
    if input_text == "":
        st.warning("Please enter a message to classify.")
    else:
        # preprocess
        transformed_text = data_preprocess_pipeline(input_text)
        # vectorize 
        vector_input = tfidf.transform([transformed_text])
        result = model.predict(vector_input)[0]
        if result == 1:
            st.header("🚨 Spam")
        else:
            st.header("✅ Not Spam")