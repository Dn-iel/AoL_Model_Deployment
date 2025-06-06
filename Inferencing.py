import streamlit as st
import dill as pickle
import gdown
import os

@st.cache_resource
def load_model_from_drive():
    file_id = "1qBFI1hvBwzKDf4630MIebcjbVm-9U7df"
    output_path = "recommender_model.pkl"

    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)

    with open(output_path, "rb") as f:
        return pickle.load(f)

# Load model
model_data = load_model_from_drive()

# Unpack
cosine_similarities = model_data["cosine_similarities"]
indices = model_data["indices"]
netflix_title = model_data["netflix_title"]
content_recommender = model_data["content_recommender"]

# UI
st.title("Netflix Recommender System")
title = st.text_input("Enter a movie title")

if title:
    try:
        recommendations = content_recommender(title)
        st.write("Recommended Titles:")
        st.write(recommendations)
    except KeyError:
        st.error("Title not found. Please try a different one.")
