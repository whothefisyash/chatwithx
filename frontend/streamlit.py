import streamlit as st
import requests

API_URL = "http://localhost:5000/chat"

st.title("Chat with X")

source = st.selectbox("Choose a data source", ["GitHub Repo", "PDF", "Research Paper", "YouTube Video"])
query = st.text_area("Ask your question:")

if st.button("Submit"):
    response = requests.post(API_URL, json={"source": source, "query": query})
    if response.status_code == 200:
        result = response.json()
        if "answer" in result:
            st.write("Answer:", result["answer"])
        else:
            st.write("Summary:", result["summary"])
    else:
        st.error("Error: Unable to process the request.")
