import streamlit as st
import joblib
import gdown
import os
import pandas as pd

# === CONFIG ===
RECOMMENDER_FILE_ID = "1_nrAO1xH_RfSjhWMYMXM8HmnSxgU2bt1"  # perbarui jika perlu
SIMILARITY_FILE_ID = "1LFwFkzQtrebYr5jjWphkUkFyGnl3EjOw"    # perbarui jika perlu
DATASET_PATH = "netflix_preprocessed.csv"

# === DOWNLOAD & LOAD PICKLES ===
@st.cache_resource
def load_recommender_function():
    output_path = "recommender.pkl"
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={RECOMMENDER_FILE_ID}"
        gdown.download(url, output_path, quiet=False)
    return joblib.load(output_path)

@st.cache_resource
def load_similarity_data():
    output_path = "data_similarity.pkl"
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={SIMILARITY_FILE_ID}"
        gdown.download(url, output_path, quiet=False)
    return joblib.load(output_path)

@st.cache_data
def load_full_dataset():
    return pd.read_csv(DATASET_PATH)

# === LOAD MODELS & DATA ===
content_recommender = load_recommender_function()
similarity_data = load_similarity_data()
full_df = load_full_dataset()

cosine_similarities = similarity_data.get("cosine_similarities")
indices = similarity_data.get("indices")
netflix_title = similarity_data.get("netflix_title")

columns_to_show = [
    'type', 'title', 'director', 'cast', 'country', 'date_added',
    'release_year', 'rating', 'listed_in', 'description',
    'duration_minutes', 'duration_seasons'
]

# === STREAMLIT UI ===
st.title("Netflix Recommender System 🎬")
st.markdown("Enter a Netflix movie title below to get similar movie recommendations.")

title = st.text_input("Enter a movie title:")
search_clicked = st.button("Get Recommended Movies")

if search_clicked and title:
    if title in set(netflix_title):
        # Show selected movie details
        movie_details_df = full_df[full_df['title'] == title][columns_to_show]
        if movie_details_df.empty:
            st.warning("Details not found in the full dataset.")
        else:
            st.subheader("Selected Movie Details")
            st.dataframe(movie_details_df, use_container_width=True)

        # Show recommendations
        st.subheader("Recommended Titles:")
        try:
            # Panggil fungsi rekomendasi dengan parameter lengkap
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
