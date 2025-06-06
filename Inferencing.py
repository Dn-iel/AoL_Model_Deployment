import streamlit as st
import dill as pickle
import gdown
import os

@st.cache_resource
def load_model_from_drive():
    file_id = "1uARTcSmf--15RMbvBxwP7TJFONlISYvK"  # <- Ganti dengan ID modelmu
    output_path = "recommender_model.pkl"

    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)

    with open(output_path, "rb") as f:
        return pickle.load(f)

# Load model data
model_data = load_model_from_drive()

# Unpack model components
cosine_similarities = model_data["cosine_similarities"]
indices = model_data["indices"]
netflix_title = model_data["netflix_title"]
content_recommender = model_data["content_recommender"]

# Streamlit UI
st.title("🎬 Netflix Movie Recommender")
st.markdown("Enter a Netflix movie title below and get similar recommendations!")

title = st.text_input("Enter a movie title:")

if title:
    try:
        recommendations = content_recommender(title)
        st.subheader("📺 Recommended Titles:")
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
    except KeyError:
        st.error("❌ Movie title not found in the database. Please try a different one.")
