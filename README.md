# 📧 Email & SMS Spam Classifier

A machine learning web application built with Python and Streamlit that accurately classifies text messages and emails as either "Spam" or "Not Spam". 

This project takes raw text input, processes it through a pipeline, and uses a trained machine learning model to make real-time predictions.

## 🚀 Features
* **Custom NLP Pipeline:** Utilizes `nltk` for word tokenization, removing alphanumeric characters, stopword removal, and stemming.
* **Machine Learning Model:** Powered by a pre-trained `scikit-learn` model and TF-IDF vectorizer.
* **Interactive UI:** A clean, responsive web interface built entirely in Python using Streamlit.

## 🛠️ Tech Stack
* **Language:** Python
* **Frontend:** Streamlit
* **Machine Learning:** scikit-learn
* **NLP:** NLTK (Natural Language Toolkit)

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Arpan2411/email-spam-classifier.git
   cd email-spam-classifier

2. Create Virtual Environment.
   ```bash
   python3 -m venv venv

   Activate the virtual environment.
   ```bash
   source venv/bin/activate   
   
   On Windows use 
   ```bash
   venv\Scripts\activate

3. Install the requirements.
   ```bash
   pip install streamlit scikit-learn pandas nltk

4. Run the project.
   ```bash
   streamlit run app.py