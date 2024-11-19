import pandas as pd

spotify_data = pd.read_csv("spotify_dataset.csv")


print(spotify_data.info())
print(spotify_data.head())
