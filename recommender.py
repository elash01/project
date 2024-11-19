# recommender.py

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import os

# Load data and environment variables
spotify_data = pd.read_csv(
    "spotify_dataset.csv"
)  # Ensure this file is present in your directory

# Load Spotify API credentials from .env file
load_dotenv()
client_id = os.getenv("a30eb90686c74f0a9e7268bc42c9eb61")
client_secret = os.getenv("9eb5cf8151634a79a1278380bfa4689d")

# Initialize Spotipy with client credentials
sp = Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=client_id, client_secret=client_secret
    )
)

# Prepare features for matrix factorization
X = spotify_data[
    [
        "acousticness",
        "danceability",
        "energy",
        "instrumentalness",
        "liveness",
        "valence",
    ]
]
# TruncatedSVD with reduced components based on feature count to avoid ValueError
n_components = min(5, X.shape[1])
svd = TruncatedSVD(n_components=n_components)
X_reduced = svd.fit_transform(X)


# Recommendation function
def recommend_songs(song_id, top_n=10):
    """
    Recommend similar songs based on a given song ID.

    Parameters:
        song_id (str): The ID of the song to base recommendations on.
        top_n (int): Number of similar songs to recommend.

    Returns:
        list: List of recommended song names.
    """
    try:
        # Find the index of the given song ID
        song_idx = spotify_data[spotify_data["id"] == song_id].index[0]
        song_vector = X_reduced[song_idx]

        # Compute similarity as the dot product between song vector and all song vectors
        similarity = np.dot(X_reduced, song_vector)

        # Get indices of top N similar songs (excluding the song itself)
        similar_songs = np.argsort(similarity)[::-1][1 : top_n + 1]

        # Return the names of the recommended songs
        return spotify_data.iloc[similar_songs]["name"].tolist()

    except IndexError:
        print(f"Song ID {song_id} not found in dataset.")
        return []


# Example function to search for song details with Spotipy
def search_song_details(song_name):
    """
    Search for song details by name using Spotipy.

    Parameters:
        song_name (str): Name of the song to search for.

    Returns:
        list: List of dictionaries with song details, including name, ID, and album cover.
    """
    results = sp.search(q=song_name, type="track", limit=5)
    song_details = []

    for track in results["tracks"]["items"]:
        song_details.append(
            {
                "name": track["name"],
                "id": track["id"],
                "album_cover": track["album"]["images"][0]["url"]
                if track["album"]["images"]
                else None,
                "preview_url": track["preview_url"],  # Audio preview link if available
            }
        )

    return song_details
