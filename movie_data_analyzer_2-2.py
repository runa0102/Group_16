import os
import requests
import tarfile
import pandas as pd
import ast
from typing import Optional
from pydantic import validate_arguments

class MovieDataAnalyzer:
    """
    A class to analyze movies using the CMU Movie Summaries dataset.
    """

    DATASET_URL = "http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz"
    DATA_DIR = os.path.join(os.path.expanduser("~"), "downloads")
    FILE_NAME = "MovieSummaries.tar.gz"

    def __init__(self):
        """
        Initializes the MovieDataAnalyzer class.
        """
        os.makedirs(self.DATA_DIR, exist_ok=True)
        file_path = os.path.join(self.DATA_DIR, self.FILE_NAME)

        if not os.path.exists(file_path):
            self._download_data(file_path)

        self._extract_data()
        self._load_data()

    def _download_data(self, file_path: str) -> None:
        """Downloads the dataset if it does not exist."""
        print("Downloading dataset...")
        response = requests.get(self.DATASET_URL, stream=True)
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")

    def _extract_data(self) -> None:
        """Extracts the dataset."""
        file_path = os.path.join(self.DATA_DIR, self.FILE_NAME)
        if file_path.endswith("tar.gz"):
            print("Extracting dataset...")
            with tarfile.open(file_path, "r:gz") as tar:
                tar.extractall(path=self.DATA_DIR)
            print("Extraction complete.")

    def _load_data(self) -> None:
        """Loads the extracted dataset into Pandas DataFrames."""
        movie_file = os.path.join(self.DATA_DIR, "MovieSummaries", "movie.metadata.tsv")
        character_file = os.path.join(self.DATA_DIR, "MovieSummaries", "character.metadata.tsv")

        if os.path.exists(movie_file):
            print("Loading movie data...")
            self.movies = pd.read_csv(movie_file, sep="\t", header=None)
            print("Movie data loaded.")
        else:
            print("Error: Movie data file not found.")

        if os.path.exists(character_file):
            print("Loading character data...")
            self.characters = pd.read_csv(character_file, sep="\t", header=None)
            print("Character data loaded.")
        else:
            print("Error: Character data file not found.")

    def movie_type(self, N: int = 10) -> pd.DataFrame:
        """
        Returns the N most common movie genres.

        Args:
        - N (int, optional): Number of genres to return. Default is 10.

        Returns:
        - pd.DataFrame: A DataFrame containing the most common movie genres and their counts.
        """
        if not isinstance(N, int):
            raise ValueError("N must be an integer.")

        genre_column = self.movies[8].dropna()
        genre_column = genre_column[genre_column.apply(lambda x: isinstance(x, str))]
        genre_column = genre_column.apply(ast.literal_eval)

        genre_counts = {}
        for genres in genre_column:
            for genre in genres.values():
                genre_counts[genre] = genre_counts.get(genre, 0) + 1

        genre_df = pd.DataFrame(genre_counts.items(), columns=["Movie_Type", "Count"])
        genre_df = genre_df.sort_values(by="Count", ascending=False).head(N)

        return genre_df

    def actor_count(self) -> pd.DataFrame:
        """
        Returns a DataFrame with the number of actors per movie.
        """
        if not hasattr(self, 'characters'):
            raise ValueError("Character data is not loaded.")

        actor_counts = self.characters.groupby(0).size()
        actor_histogram = actor_counts.value_counts().reset_index()
        actor_histogram.columns = ["Number of Actors", "Movie Count"]
        actor_histogram = actor_histogram.sort_values(by="Number of Actors")

        return actor_histogram

    @validate_arguments
    def actor_distributions(self, gender: str, min_height: float, max_height: float, plot: bool = False) -> pd.DataFrame:
        """
        Returns a DataFrame with actor height distributions filtered by gender.

        Args:
        - gender (str): "All" or a specific gender from the dataset.
        - min_height (float): Minimum height filter.
        - max_height (float): Maximum height filter.
        - plot (bool, optional): If True, generates a histogram plot. Default is False.

        Returns:
        - pd.DataFrame: A DataFrame containing filtered actor heights.
        """
        if not hasattr(self, 'characters'):
            raise ValueError("Character data is not loaded.")

        df = self.characters[[5, 6]].dropna()
        df.columns = ["Gender", "Height"]
        df["Height"] = pd.to_numeric(df["Height"], errors="coerce")
        df["Gender"] = df["Gender"].astype(str)

        if min_height > max_height:
            raise ValueError("min_height must be less than or equal to max_height.")

        valid_genders = df["Gender"].unique()
        if gender.lower() != "all":
            if gender.lower() not in map(str.lower, valid_genders):
                raise ValueError(f"Invalid gender. Choose from: {list(valid_genders)} or 'All'.")
            df = df[df["Gender"].str.lower() == gender.lower()]

        df = df[(df["Height"] >= min_height) & (df["Height"] <= max_height)]

        if plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 5))
            plt.hist(df["Height"], bins=20, edgecolor="black", alpha=0.7)
            plt.xlabel("Height (meters)")
            plt.ylabel("Count")
            plt.title(f"Height Distribution for Gender: {gender}")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.show()

        return df

    def releases(self, genre: Optional[str] = None) -> pd.DataFrame:
        """
        Returns a DataFrame with the number of movie releases per year.
        """
        df = self.movies[[3, 8]].dropna()
        df.columns = ["Year", "Genres"]
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

        if genre:
            df["Genres"] = df["Genres"].apply(ast.literal_eval)
            df = df[df["Genres"].apply(lambda g: genre in g.values())]

        release_counts = df["Year"].value_counts().reset_index()
        release_counts.columns = ["Year", "Movie Count"]
        release_counts = release_counts.sort_values(by="Year")

        return release_counts

    def ages(self, mode: str = "Y") -> pd.DataFrame:
        """
        Returns a DataFrame with the number of births per year or month.
        """
        if not hasattr(self, 'characters'):
            raise ValueError("Character data is not loaded.")

        if mode.upper() == "M":
            df = self.characters[[7]].dropna()
            df.columns = ["Month"]
            df["Month"] = df["Month"].astype(int)
            births = df["Month"].value_counts().reset_index()
            births.columns = ["Month", "Birth Count"]
            births = births.sort_values(by="Month")
        else:
            df = self.characters[[6]].dropna()
            df.columns = ["Year"]
            df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
            births = df["Year"].value_counts().reset_index()
            births.columns = ["Year", "Birth Count"]
            births = births.sort_values(by="Year")

        return births


