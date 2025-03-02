import os
import requests
import tarfile
import pandas as pd
from typing import Optional
from pydantic import validate_arguments

class MovieDataAnalyzer:
    """
    A class to analyze movies using the CMU Movie Summaries dataset.

    Attributes:
    - DATASET_URL: URL of the dataset.
    - DATA_DIR: Directory where the dataset is stored.
    - FILE_NAME: Name of the dataset file.
    - movies: DataFrame containing movie data.
    """

    DATASET_URL = "http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz"
    DATA_DIR = os.path.join(os.path.expanduser("~"), "downloads")
    FILE_NAME = "MovieSummaries.tar.gz"

    def __init__(self):
        """
        Initializes the MovieDataAnalyzer class:
        - Creates the `downloads` directory if not present.
        - Downloads the dataset if it does not exist.
        - Extracts the dataset.
        - Loads the movie data into a Pandas DataFrame.
        """
        os.makedirs(self.DATA_DIR, exist_ok=True)
        file_path = os.path.join(self.DATA_DIR, self.FILE_NAME)

        if not os.path.exists(file_path):
            self._download_data(file_path)

        self._extract_data()
        self._load_data()

    def _download_data(self, file_path: str) -> None:
        """
        Downloads the dataset if it does not exist.

        Args:
        - file_path (str): Path where the dataset will be saved.
        """
        print("Downloading dataset...")
        response = requests.get(self.DATASET_URL, stream=True)
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")

    def _extract_data(self) -> None:
        """
        Extracts the dataset if not already extracted.
        """
        file_path = os.path.join(self.DATA_DIR, self.FILE_NAME)
        if file_path.endswith("tar.gz"):
            print("Extracting dataset...")
            with tarfile.open(file_path, "r:gz") as tar:
                tar.extractall(path=self.DATA_DIR)
            print("Extraction complete.")

    def _load_data(self) -> None:
        """
        Loads the extracted dataset into Pandas DataFrames.
        """
        movie_file = os.path.join(self.DATA_DIR, "MovieSummaries", "movie.metadata.tsv")
        if os.path.exists(movie_file):
            print("Loading movie data...")
            self.movies = pd.read_csv(movie_file, sep="\t", header=None)
            print("Movie data loaded.")
        else:
            print("Error: Movie data file not found.")

        
        character_file = os.path.join(self.DATA_DIR, "MovieSummaries", "character.metadata.tsv")
        if os.path.exists(character_file):
            print("Loading character data...")
            self.characters = pd.read_csv(character_file, sep="\t", header=None)
            print("Character data loaded.")
        else:
            print("Error: Character data file not found.")


# part 1: Develop a first method for your class called __movie_type__ 
    
    def movie_type(self, N: int = 10) -> pd.DataFrame:
         """
         Returns the N most common movie genres.
 
         Args:
         - N (int, optional): Number of genres to return. Default is 10.
 
         Returns:
         - pd.DataFrame: A DataFrame containing the most common movie genres and their counts.

         Raises:
         - ValueError: If N is not an integer.
         """
         if not isinstance(N, int):
             raise ValueError("N must be an integer.")

         # Extract the column containing genres (Column 8)
         genre_column = self.movies[8].dropna()

         # Ensure all values are strings before applying eval()
         genre_column = genre_column[genre_column.apply(lambda x: isinstance(x, str))]

         # Convert JSON-like strings to dictionaries
         genre_column = genre_column.apply(eval)
 
         # Count occurrences of each genre (using values instead of keys)
         genre_counts = {}
         for genres in genre_column:
             for genre in genres.values():  # Extract actual genre names instead of IDs
                 genre_counts[genre] = genre_counts.get(genre, 0) + 1

         # Create a DataFrame with the most common genres
         genre_df = pd.DataFrame(genre_counts.items(), columns=["Movie_Type", "Count"])
         genre_df = genre_df.sort_values(by="Count", ascending=False).head(N)

         return genre_df
        
    # part 2, Develop a second method called __actor_count__
    def actor_count(self) -> pd.DataFrame:
        """
        Returns a DataFrame with the number of actors per movie.

        Returns:
        - pd.DataFrame: A histogram of "number of actors" vs "movie count".
        """
        if not hasattr(self, 'characters'):
            raise ValueError("Character data is not loaded.")

        # Extract movie IDs and count the number of actors per movie
        actor_counts = self.characters.groupby(0).size()

        # Convert to a histogram-like DataFrame
        actor_histogram = actor_counts.value_counts().reset_index()
        actor_histogram.columns = ["Number of Actors", "Movie Count"]
        actor_histogram = actor_histogram.sort_values(by="Number of Actors")

        return actor_histogram


    # part 3, develop third method called actor_distributions
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

        Raises:
        - ValueError: If gender is not in the dataset or heights are invalid.
        """
        if not hasattr(self, 'characters'):
            raise ValueError("Character data is not loaded.")

        # Extract relevant columns: gender (6) and height (7)
        GENDER_COL = 5  # Gender is in column 5
        HEIGHT_COL = 6  # Height is in column 6

        df = self.characters[[GENDER_COL, HEIGHT_COL]].dropna()
        df.columns = ["Gender", "Height"]

        # Convert height column to float
        df["Height"] = pd.to_numeric(df["Height"], errors="coerce")

        # Ensure Gender column only contains valid strings
        df["Gender"] = df["Gender"].astype(str)

        # Ensure min_height is ≤ max_height
        if min_height > max_height:
            raise ValueError("min_height must be less than or equal to max_height.")

        # Apply gender filter
        valid_genders = df["Gender"].unique()
        if gender.lower() != "all":
            if gender.lower() not in map(str.lower, valid_genders):
                raise ValueError(f"Invalid gender. Choose from: {list(valid_genders)} or 'All'.")
            df = df[df["Gender"].str.lower() == gender.lower()]

        # Apply height range filter
        df = df[(df["Height"] >= min_height) & (df["Height"] <= max_height)]

        # Only plot if `plot=True`
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
