import streamlit as st
from test import test_bert_endpoint

endpoint = "finance-news-sentiments-classifier"

st.title("Lajan")
st.write("This model predicts the sentiment of a news article on the stock market.")

finance_news: str = st.text_input("Enter the finance news below")

if st.button("Get sentiment"):
    response = test_bert_endpoint(endpoint, finance_news, False)["predictions"]
    probabilities = list(response["probabilities"].values())
    index_max = probabilities.index(max(probabilities))
    sentiment = "negative" if index_max == 0 else "neutral" if index_max == 1 else "positive"
    probability = round(probabilities[index_max] * 100)

    if sentiment == "negative":
        st.markdown(f":red[This news is most likely **{sentiment}**, with a confidence of **{probability}%**.]")
    elif sentiment == "positive":
        st.markdown(f":green[This news is most likely **{sentiment}**, with a confidence of **{probability}%**.]")
    else:
        st.markdown(f"This news is most likely **{sentiment}**, with a confidence of **{probability}%**.")
