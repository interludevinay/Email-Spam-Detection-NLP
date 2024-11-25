import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# import nltk
# nltk.download('punkt')
import sklearn


tfidf = pickle.load(open("vectorizer.pkl","rb"))
model = pickle.load(open("model.pkl","rb"))

st.title("Email Spam Classifier")

input_sms = st.text_area("Enter the Email/Sms")

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))


    return " ".join(y)



# Display
if st.button("Predict"):
    # Preprocess
    transformed_text = transform_text(input_sms)

    # Vectorized
    vector_int = tfidf.transform([transformed_text])

    # Predict
    res = model.predict(vector_int)[0]

    if res == 1:
        st.header("The following Email/SMS is Spammed.")
    else:
        st.header("The following Email/SMS is Not Spammed.")