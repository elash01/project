import streamlit as st
import pandas as pd
import os

# Load Dataset
data_file = (
    "spotify_dataset.csv"  # Ensure this file is in the same directory as the script
)
if os.path.exists("spotify_dataset.csv"):
    spotify_data = pd.read_csv("spotify_dataset.csv")
else:
    st.error(f"Error: {data_file} not found in the project folder.")
    st.stop()  # Stop the app if the file isn't found

# Clean and Convert Data
numeric_columns = [
    "valence",
    "danceability",
    "energy",
    "tempo",
    "popularity",
    "duration_ms",
    "acousticness",
    "liveness",
]
for col in numeric_columns:
    spotify_data[col] = pd.to_numeric(
        spotify_data[col], errors="coerce"
    )  # Convert to numeric, handle errors
spotify_data["artists"] = (
    spotify_data["artists"].str.strip("[]").str.replace("'", "")
)  # Clean artist names
spotify_data = spotify_data.dropna(
    subset=numeric_columns
)  # Drop rows with invalid numeric data

# Streamlit App UI
st.title("🎶 Music Recommendation System")

# Section 1: Search for a Song
st.subheader("🔍 Search for a Song by Name")
song_name = st.text_input("Enter song name:")
if st.button("Search"):
    if song_name:
        search_results = spotify_data[
            spotify_data["name"].str.contains(song_name, case=False, na=False)
        ]
        if not search_results.empty:
            st.write("Search Results:")
            for idx, row in search_results.iterrows():
                st.write(f"{idx + 1}. {row['name']} by {row['artists']}")
                st.write(
                    f"Popularity: {row['popularity']} | Release Year: {row['year']}"
                )
                st.write("---")
        else:
            st.write("No songs found matching your search query.")

# Section 2: Get Recommendations
st.subheader("🎧 Get Song Recommendations by Song ID")
song_id = st.text_input("Enter Song ID for Recommendations:")
if st.button("Get Recommendations"):
    if song_id in spotify_data["id"].values:
        selected_song = spotify_data[spotify_data["id"] == song_id].iloc[0]
        recommendations = (
            spotify_data[
                (spotify_data["id"] != song_id)  # Exclude the selected song
                & (
                    (abs(spotify_data["valence"] - selected_song["valence"]) < 0.1)
                    | (
                        abs(
                            spotify_data["danceability"] - selected_song["danceability"]
                        )
                        < 0.1
                    )
                    | (abs(spotify_data["energy"] - selected_song["energy"]) < 0.1)
                )
            ]
            .sort_values(by="popularity", ascending=False)
            .head(5)
        )

        if not recommendations.empty:
            st.write("Recommended Songs:")
            for idx, row in recommendations.iterrows():
                st.write(f"{idx + 1}. {row['name']} by {row['artists']}")
                st.write(
                    f"Popularity: {row['popularity']} | Release Year: {row['year']}"
                )
                st.write("---")
        else:
            st.write("No recommendations available for this song.")
    else:
        st.write("Song ID not found in the dataset.")

# Section 3: Mood Matcher
st.subheader("😌 Match Songs to Your Mood")
mood = st.selectbox("Choose your mood:", ["Happy", "Sad", "Energetic", "Calm"])
if st.button("Get Mood-Matched Songs"):
    mood_filters = {
        "Happy": spotify_data["valence"] > 0.7,
        "Sad": spotify_data["valence"] < 0.3,
        "Energetic": spotify_data["energy"] > 0.7,
        "Calm": spotify_data["energy"] < 0.3,
    }
    if mood in mood_filters:
        matched_songs = (
            spotify_data[mood_filters[mood]]
            .sort_values(by="popularity", ascending=False)
            .head(10)
        )
        if not matched_songs.empty:
            st.write(f"Songs that match your mood ({mood}):")
            for idx, row in matched_songs.iterrows():
                st.write(f"{idx + 1}. {row['name']} by {row['artists']}")
        else:
            st.write(f"No songs found matching the {mood} mood.")
