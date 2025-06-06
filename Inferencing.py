import streamlit as st
import dill as pickle
import gdown
import os

@st.cache_resource
def load_model_from_drive():
    file_id = "1uARTcSmf--15RMbvBxwP7TJFONlISYvK"
    output_path = "recommender_model.pkl"

    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)

    with open(output_path, "rb") as f:
        return pickle.load(f)

# Load model
model_data = load_model_from_drive()
cosine_similarities = model_data["cosine_similarities"]
indices = model_data["indices"]
netflix_df = model_data["netflix_title"]
content_recommender = model_data["content_recommender"]

# UI
st.title("🎬 Netflix Movie Recommender")
st.markdown("Enter a Netflix movie title below to get detailed information and similar movie recommendations.")

title = st.text_input("Enter a movie title:")
search_clicked = st.button("Get Recommended Movies")  # Tombol SELALU muncul

if search_clicked and title:
    if title in indices:
    index = indices[title]
    movie_details_df = netflix_df.loc[[index]]  # Ini hasilnya pasti DataFrame
    st.subheader("🎥 Selected Movie Details")
    st.table(movie_details_df)

    st.subheader("📺 Recommended Titles with Details:")
    recommendations = content_recommender(title)

        for i, rec_title in enumerate(recommendations, 1):
            with st.expander(f"{i}. {rec_title}"):
                if rec_title in indices:
                    rec_index = indices[rec_title]
                    rec_details_df = netflix_df.loc[[rec_index]]
                    st.table(rec_details_df)
                else:
                    st.warning(f"Details for '{rec_title}' not found.")
    else:
        st.error("❌ Movie title not found in dataset.")
