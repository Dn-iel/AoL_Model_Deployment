import streamlit as st
import dill as pickle
import gdown
import os
import pandas as pd

# Load model dari Google Drive
@st.cache_resource
def load_model_from_drive():
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
    df = pd.read_csv("netflix_preprocessed.csv")  # Ganti jika path berbeda
    return df

def display_recommendations(recommended_titles, df):
    if recommended_titles.empty:
        st.warning("Tidak ada hasil rekomendasi ditemukan.")
        return

    for title in recommended_titles['title']:
        row = df[df['title'] == title].iloc[0]

        st.subheader(f"🎞️ {row['title']}")
        st.markdown(f"""
        - **Type**: {row['type']}
        - **Director**: {row['director']}
        - **Cast**: {row['cast']}
        - **Country**: {row['country']}
        - **Date Added**: {row['date_added']}
        - **Release Year**: {row['release_year']}
        - **Rating**: {row['rating']}
        - **Genre**: {row['listed_in']}
        - **Description**: {row['description']}
        - **Duration (minutes)**: {row['duration_minutes']}
        - **Duration (seasons)**: {row['duration_seasons']}
        """)
        st.markdown("---")

def main():
    st.set_page_config(page_title="Netflix Recommender", layout="centered")
    st.title("🎬 Netflix Content Recommender")
    st.markdown("Masukkan judul film/seri yang kamu sukai untuk mendapatkan rekomendasi konten serupa.")

    # Load model dan dataset
    model_data = load_model_from_drive()
    df = load_full_dataset()

    content_recommender = model_data['content_recommender']
    title_list = model_data['netflix_title']

    # Input user
    input_title = st.text_input("Masukkan judul film/seri (case-sensitive):", "")

    if st.button("Tampilkan Rekomendasi"):
        if input_title in title_list:
            recommended_titles = content_recommender(input_title)

            st.success(f"Hasil rekomendasi untuk: **{input_title}**")
            display_recommendations(recommended_titles, df)

        else:
            st.error("Judul tidak ditemukan dalam basis data. Pastikan penulisan sesuai (case-sensitive).")

if __name__ == "__main__":
    main()
