import streamlit as st
import requests
import dill as pickle
import io

@st.cache_resource
def load_model_from_drive():
    url = "https://drive.google.com/uc?export=download&id=1qBFI1hvBwzKDf4630MIebcjbVm-9U7df"
    response = requests.get(url)
    with io.BytesIO(response.content) as f:
        return pickle.load(f)

# Load model
model_data = load_model_from_drive()

# Unpack komponen jika disimpan sebagai dict
cosine_similarities = model_data["cosine_similarities"]
indices = model_data["indices"]
netflix_title = model_data["netflix_title"]
content_recommender = model_data["content_recommender"]

# Streamlit UI
st.title("Netflix Recommender System")
title = st.text_input("Enter a movie title")

if title:
    try:
        recommendations = content_recommender(title)
        st.write("Recommended Titles:")
        st.write(recommendations)
    except KeyError:
        st.error("Title not found. Please try a different one.")
