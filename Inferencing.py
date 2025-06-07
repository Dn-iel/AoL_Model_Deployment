import streamlit as st
import joblib
import gdown
import os
import pandas as pd

# === SETTING FILE ID GOOGLE DRIVE ===
SIMILARITY_FILE_ID = "1fSDXnCN_b1AjZmrFQ-CjLqX9snuRf9cK"   # similarity_data.pkl
DATASET_PATH = "netflix_preprocessed.csv"  # Pastikan file ini tersedia

# === DOWNLOAD & LOAD SIMILARITY DATA ===
@st.cache_resource
def load_similarity_data():
    path = "similarity_data.pkl"
    if not os.path.exists(path):
        gdown.download(f"https://drive.google.com/uc?id={SIMILARITY_FILE_ID}", path, quiet=False)
    return joblib.load(path)

# === LOAD DATASET ===
@st.cache_data
def load_full_dataset():
    return pd.read_csv(DATASET_PATH)

# === FUNGSI RECOMMENDER (BUAT SENDIRI) ===
def content_recommender(title, cosine_similarities, indices, top_n=5):
    # Ambil indeks film yang dicari
    if title not in indices:
        return []
    idx = indices[title]
    
    # Ambil daftar similarity scores film lain terhadap film tersebut
    sim_scores = list(enumerate(cosine_similarities[idx]))
    
    # Urutkan berdasarkan similarity tertinggi, kecuali film sendiri (idx)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Ambil top_n rekomendasi setelah film itu sendiri
    top_scores = sim_scores[1:top_n+1]
    
    # Ambil judul film berdasarkan indeks
    recommended_titles = [indices.index[i[0]] for i in top_scores]
    return recommended_titles

# === LOAD DATA ===
similarity_data = load_similarity_data()
full_df = load_full_dataset()

cosine_similarities = similarity_data["cosine_similarities"]
indices = similarity_data["indices"]
netflix_title = similarity_data["netflix_title"]

# === UI STREAMLIT ===
st.title("Netflix Recommender System 🎬")
st.markdown("Enter a Netflix movie title below to get similar movie recommendations.")

title = st.text_input("Enter a movie title:")
search_clicked = st.button("Get Recommended Movies")

# Kolom yang akan ditampilkan
columns_to_show = [
    'type', 'title', 'director', 'cast', 'country', 'date_added',
    'release_year', 'rating', 'listed_in', 'description',
    'duration_minutes', 'duration_seasons'
]

if search_clicked and title:
    if title in set(netflix_title):
        movie_details_df = full_df[full_df['title'] == title][columns_to_show]
        if movie_details_df.empty:
            st.warning("Details not found in the full dataset.")
        else:
            st.subheader("Selected Movie Details")
            st.dataframe(movie_details_df, use_container_width=True)

        st.subheader("Recommended Titles:")
        try:
            recommendations = content_recommender(title, cosine_similarities, indices)
            for i, rec_title in enumerate(recommendations, 1):
                with st.expander(f"{i}. {rec_title}"):
                    rec_details_df = full_df[full_df['title'] == rec_title][columns_to_show]
                    if not rec_details_df.empty:
                        st.dataframe(rec_details_df, use_container_width=True)
                    else:
                        st.warning(f"Details for '{rec_title}' not found.")
        except Exception as e:
            st.error(f"❌ Error while generating recommendations: {e}")
    else:
        st.error("❌ Movie title not found in model title list.")
