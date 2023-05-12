import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

def train_model():
    review = pd.read_csv('reviews.csv')
    review = review.rename(columns = {'text':'review'},inplace = False)
    x = review.review
    y = review.polarity
    x_train , x_test , y_train , y_test = train_test_split(x , y , train_size = 0.6, random_state = 1) 
    vector = CountVectorizer(stop_words = 'english',lowercase=False)
    vector.fit(x_train)
    x_transformed = vector.transform(x_train)
    x_transformed.toarray()
    x_test_transformed = vector.transform(x_test)
    naivebayes = MultinomialNB()
    naivebayes.fit(x_transformed,y_train)
    saved_model = pickle.dumps(naivebayes)
    s = pickle.loads(saved_model)
    st.header('SENTIMENT ANALYSIS')
    input = st.text_area('ENTER THE TEXT FOR ANALYSIS')
    vec = vector.transform([input]).toarray()
def senti_analysis():
    st.write((str(list(s.predict(vec))[0]).replace('0', 'NEGATIVE').replace('1', 'POSITIVE')))
train_model()
if st.button('ANALYSE'):
    senti_analysis()

