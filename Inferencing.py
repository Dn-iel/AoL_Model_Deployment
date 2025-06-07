import streamlit as st
import dill as pickle  # Gunakan dill karena menyimpan fungsi di pkl
import gdown
import os
import pandas as pd

# Download dan load model dari Google Drive
@st.cache_resource
def load_recommender():
    file_id = "1uARTcSmf--15RMbvBxwP7TJFONlISYvK"
    output_path = "recommender_model.pkl"

    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)

    with open(output_path, "rb") as f:
        return pickle.load(f)

# Load dataset lengkap dari CSV
@st.cache_data
def load_full_dataset():
    df = pd.read_csv("netflix_preprocessed.csv")  # Ganti path jika perlu
    return df

# Kolom yang akan ditampilkan
columns_to_show = [
    'type', 'title', 'director', 'cast', 'country', 'date_added',
    'release_year', 'rating', 'listed_in', 'description',
    'duration_minutes', 'duration_seasons'
]

# Load model dan data
recommender_data = load_recommender()
cosine_similarities = recommender_data["cosine_similarities"]
indices = recommender_data["indices"]
netflix_title = recommender_data["netflix_title"]
content_recommender = recommender_data["content_recommender"]

full_df = load_full_dataset()

# Streamlit UI
st.title("Netflix Recommender System 🎬")
st.markdown("Enter a Netflix movie title below to get similar movie recommendations.")

title = st.text_input("Enter a movie title:")
search_clicked = st.button("Get Recommended Movies")

if search_clicked and title:
    if title in set(netflix_title):
        # Tampilkan detail film
        movie_details_df = full_df[full_df['title'] == title][columns_to_show]
        if movie_details_df.empty:
            st.warning("Details not found in the full dataset.")
        else:
            st.subheader("Selected Movie Details")
            st.dataframe(movie_details_df, use_container_width=True)

        # Tampilkan rekomendasi
        st.subheader("Recommended Titles:")
        try:
            recommendations = content_recommender(title)  # Fungsi dari .pkl
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
