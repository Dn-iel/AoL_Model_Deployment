import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import sklearn
import scipy
import dill

import dill as pickle

@st.cache_resource
def load_recommender():
    url = "https://drive.google.com/uc?export=download&id=1qBFI1hvBwzKDf4630MIebcjbVm-9U7df"
    response = requests.get(url)
    with gzip.open(io.BytesIO(response.content), 'rb') as f:
        return pickle.load(f)

def main():
    # Load model dan komponen
    recommender_data = load_recommender()

    cosine_similarities = recommender_data["cosine_similarities"]
    indices = recommender_data["indices"]
    netflix_title = recommender_data["netflix_title"]
    content_recommender = recommender_data["content_recommender"]

    # Streamlit interface
    st.title("🎬 Netflix Recommender System")
    st.write("Masukkan judul film/serial Netflix dan dapatkan rekomendasi berdasarkan deskripsinya.")

    title = st.text_input("Masukkan judul film:")

    if title:
        if title in indices:
            recommendations = content_recommender(title)
            st.subheader("Rekomendasi untuk Anda:")
            st.write(recommendations)
        else:
            st.error("Judul tidak ditemukan. Coba judul lain.")

if __name__ == '__main__':
    main()

