import streamlit as st
import joblib
import numpy as np

#load train model and vectorizer
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

st.title("True and Fake News Prediction Model")

st.image(r"C:\Users\User1\Documents\Python_Project\Skill_Harvest\ftimage.jpeg", use_column_width=True)

text_input = st.text_area("Enter the news text:", key = "unique_text_input")

#Preprae input
if text_input:
    transformed_input = vectorizer.transform({text_input})


    prediction = model.predict(transformed_input)

    if prediction == 1:
       st.write("Prediction: True News")
    else:
       st.write("Prediction: Fake News")
    