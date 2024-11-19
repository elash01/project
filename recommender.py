import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

# Load data
spotify_data = pd.read_csv(
    "spotify_dataset.csv"
)  # Ensure this file is in your directory

# Prepare features for matrix factorization
numeric_features = [
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "valence",
]
spotify_data[numeric_features] = spotify_data[numeric_features].apply(
    pd.to_numeric, errors="coerce"
)
spotify_data = spotify_data.dropna(
    subset=numeric_features
)  # Remove rows with missing values

# Perform dimensionality reduction for recommendation
n_components = min(5, len(numeric_features))  # Use fewer components for simplicity
svd = TruncatedSVD(n_components=n_components)
X_reduced = svd.fit_transform(spotify_data[numeric_features])


# Recommendation function
def recommend_songs(song_id, top_n=10):
    """
    Recommend similar songs based on a given song ID.

    Parameters:
        song_id (str): The ID of the song to base recommendations on.
        top_n (int): Number of similar songs to recommend.

    Returns:
        list: List of dictionaries with recommended song details.
    """
    try:
        # Find the index of the given song ID
        song_idx = spotify_data[spotify_data["id"] == song_id].index[0]
        song_vector = X_reduced[song_idx]

        # Compute similarity as the dot product between song vector and all song vectors
        similarity = np.dot(X_reduced, song_vector)

        # Get indices of top N similar songs (excluding the song itself)
        similar_songs_idx = np.argsort(similarity)[::-1][1 : top_n + 1]

        # Retrieve the recommended songs' details
        recommendations = spotify_data.iloc[similar_songs_idx][
            ["name", "artists", "popularity", "id"]
        ]
        return recommendations.to_dict("records")

    except IndexError:
        print(f"Song ID {song_id} not found in dataset.")
        return []


# Example function to search for songs in the dataset
def search_song_details(song_name):
    """
    Search for song details by name in the local dataset.

    Parameters:
        song_name (str): Name of the song to search for.

    Returns:
        list: List of dictionaries with song details.
    """
    search_results = spotify_data[
        spotify_data["name"].str.contains(song_name, case=False, na=False)
    ]
    return search_results[["name", "artists", "id", "popularity"]].to_dict("records")
