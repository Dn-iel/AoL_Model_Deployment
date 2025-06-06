import streamlit as st
import dill as pickle  # or use 'import pickle' if not using functions

@st.cache_resource
def load_recommender():
    with open("recommender_model.pkl", "rb") as f:
        return pickle.load(f)

# Load the model
recommender_data = load_recommender()

# Unpack the components
cosine_similarities = recommender_data["cosine_similarities"]
indices = recommender_data["indices"]
netflix_title = recommender_data["netflix_title"]
content_recommender = recommender_data["content_recommender"]

# Example usage in Streamlit
st.title("Netflix Recommender System")

title = st.text_input("Enter a movie title")

if title:
    try:
        recommendations = content_recommender(title)
        st.write("Recommended Titles:")
        st.write(recommendations)
    except KeyError:
        st.error("Title not found. Please try a different one.")
