import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Page Configuration
st.set_page_config(
    page_title="Email/SMS Spam Classifier",
    layout="wide"
)

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    # Lowercase
    text = text.lower()
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove special characters (only alphabetic)
    tokens = [word for word in tokens if word.isalpha()]
    
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    
    # Stemming
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]
    
    return ' '.join(tokens)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")
input_text = st.text_area("Enter your message")

if st.button("Predict"):
    # preprocess the input text
    transformed_input = transform_text(input_text)

    # vectorize
    vector_input = tfidf.transform([transformed_input]).toarray()

    # make prediction
    prediction = model.predict(vector_input)[0]

    # display result
    if prediction == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")


