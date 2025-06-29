import streamlit as st
import joblib

vectorizer = joblib.load("vectorizer.jb")
modle = joblib.load("lr_modle.jb")

st.title("FAKE NEWS DETECTION")
st.write("Enter the news article to check whether it is Fake or Real")

news_input = st.text_area("News Article:", "")

if st.button("Check News"):
    if news_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        transform_input = vectorizer.transform([news_input])
        prediction = modle.predict(transform_input)

        if prediction[0] == 1:
            st.success("The News is Fake")
        else:
            st.error("The News is Real")