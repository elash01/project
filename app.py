import streamlit as st
from recommender import recommend_songs, spotify_data, search_song_details
from utils import (
    generate_wordcloud,
    mood_matcher,
)  # Assume these helper functions are defined in utils.py
import random
import os
from dotenv import load_dotenv
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials

# Load Spotify credentials
load_dotenv()
client_id = os.getenv("a30eb90686c74f0a9e7268bc42c9eb61")
client_secret = os.getenv("9eb5cf8151634a79a1278380bfa4689d")

# Initialize Spotipy
sp = Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=client_id, client_secret=client_secret
    )
)

# Streamlit App UI
st.title("🎶 Music Recommendation System")

# Section 1: Search for a Song
st.subheader("🔍 Search for a Song by Name")
song_name = st.text_input("Enter song name:")
if st.button("Search"):
    if song_name:
        results = search_song_details(song_name)
        st.write("Search Results:")
        for idx, track in enumerate(results):
            st.write(f"{idx + 1}. {track['name']}")
            st.write(f"Song ID: {track['id']}")
            if track["album_cover"]:
                st.image(track["album_cover"], width=100)
            if track["preview_url"]:
                st.audio(track["preview_url"])
            st.write("---")

# Section 2: Get Recommendations
st.subheader("🎧 Get Song Recommendations by Song ID")
song_id = st.text_input("Enter Song ID for Recommendations:")
if st.button("Get Recommendations"):
    if song_id in spotify_data["id"].values:
        recommendations = recommend_songs(song_id=song_id)
        st.write("Recommended Songs:")
        for song in recommendations:
            st.write(song)
    else:
        st.write("Song ID not found in the dataset.")

# Section 3: Mood Matcher
st.subheader("😌 Match Songs to Your Mood")
mood = st.selectbox("Choose your mood:", ["Happy", "Sad", "Energetic", "Calm"])
if st.button("Get Mood-Matched Songs"):
    matched_songs = mood_matcher(mood, spotify_data)
    st.write("Songs that match your mood:")
    for song in matched_songs["name"]:
        st.write(song)
