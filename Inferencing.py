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

# Unpack model components
cosine_similarities = model_data["cosine_similarities"]
indices = model_data["indices"]
netflix_df = model_data["netflix_title"]  # This should be a DataFrame
content_recommender = model_data["content_recommender"]

# UI
st.title("🎬 Netflix Movie Recommender")
st.markdown("Enter a Netflix movie title below to get detailed information and similar movie recommendations.")

title = st.text_input("Enter a movie title:")
if st.button("Get Recommended Movies") and title:
    try:
        # Show input movie details
        index = indices[title]
        movie_details = netflix_df.loc[index]

        st.subheader("🎥 Selected Movie Details")
        st.table(movie_details.to_frame().T)

        # Get recommendations
        recommendations = content_recommender(title)

        st.subheader("🔍 Recommended Movies:")
        for i, rec_title in enumerate(recommendations, 1):
            try:
                rec_index = indices[rec_title]
                rec_details = netflix_df.loc[rec_index]
                with st.expander(f"{i}. {rec_title}"):
                    st.table(rec_details.to_frame().T)
            except KeyError:
                st.warning(f"Details for '{rec_title}' not found in dataset.")
    except KeyError:
        st.error("❌ Movie title not found. Please try a different one.")
