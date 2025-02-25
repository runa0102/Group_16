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
    DATA_DIR = "downloads/"
    FILE_NAME = "MovieSummaries.tar.gz"

    def __init__(self):
        """
        Initializes the MovieDataAnalyzer class:
        - Creates the `downloads/` directory if not present.
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
        movie_file = os.path.join(self.DATA_DIR, "movie.metadata.tsv")
        if os.path.exists(movie_file):
            print("Loading movie data...")
            self.movies = pd.read_csv(movie_file, sep="\t", header=None)
            print("Movie data loaded.")
        else:
            print("Error: Movie data file not found.")
