"""Movie Data Analyzer.

This script analyzes movie data by extracting insights from movie titles and plot summaries.
Using an AI pipeline, it also predicts the genres of movies based on their plot summaries, titles, and release years.
"""
import os
import tarfile
import ast
import pandas as pd
import requests
import matplotlib.pyplot as plt
import ollama
from pydantic import validate_arguments
class MovieDataAnalyzer:
    """
    A class to analyze movies using the CMU Movie Summaries dataset.
    """
    DATASET_URL = "http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz"
    DATA_DIR = os.path.join(os.getcwd(), "MovieData")
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
        self._load_plot_summaries()


    def _download_data(self, file_path: str) -> None:
        """Downloads the dataset if it does not exist."""        
        print("Downloading dataset...")
        response = requests.get(self.DATASET_URL, stream=True, timeout=10)

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
        print("\nExtracting dataset...")

        movie_file = os.path.join(self.DATA_DIR, "MovieSummaries", "movie.metadata.tsv")
        character_file = os.path.join(self.DATA_DIR, "MovieSummaries", "character.metadata.tsv")

        if os.path.exists(movie_file):
            print("Loading movie data...")
            self.movies = pd.read_csv(movie_file, sep="\t", header=None)
            print("Movie data loaded successfully.")
        else:
            print("Error: Movie metadata file not found.")

        if os.path.exists(character_file):
            print("Loading character data...")
            self.characters = pd.read_csv(character_file, sep="\t", header=None)
            print("Character data loaded successfully.")
        else:
            print("Error: Character metadata file not found.")

    def _load_plot_summaries(self) -> None:
        """Loads the plot summaries from plot_summaries.txt into a dictionary."""
        plot_file = os.path.join(self.DATA_DIR, "MovieSummaries", "plot_summaries.txt")
        self.plot_summaries = {}
        current_id = None
        current_summary_lines = []

        with open(plot_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() == "":
                    continue
                parts = line.split("\t", 1)
                if len(parts) == 2 and parts[0].isdigit():
                    if current_id is not None:
                        self.plot_summaries[current_id] = "".join(current_summary_lines).strip()
                    current_id = parts[0]
                    current_summary_lines = [parts[1]]
                else:
                    current_summary_lines.append(line)
            if current_id is not None:
                self.plot_summaries[current_id] = "".join(current_summary_lines).strip()


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
    def actor_distributions(self, gender: str, min_height: float,
                            max_height: float, plot: bool = False) -> pd.DataFrame:
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
            plt.figure(figsize=(8, 5))
            plt.hist(df["Height"], bins=20, edgecolor="black", alpha=0.7)
            plt.xlabel("Height (meters)")
            plt.ylabel("Count")
            plt.title(f"Height Distribution for Gender: {gender}")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.show()

        return df

    def releases(self, genre: str = None) -> pd.DataFrame:
        """
        Returns a DataFrame with the number of movie releases per year.
        Args:
        - genre (str, optional): The genre to filter by. If None, return all movies.
        Returns:
        - pd.DataFrame: A DataFrame with movie counts per year.
        """
        if self.movies is None:
            raise ValueError("Movie data is not loaded.")

        df = self.movies[[3, 8]].dropna()
        df.columns = ["Year", "Genres"]

        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

        all_genres = df["Genres"].dropna().apply(ast.literal_eval)
        genre_counts = {}
        for genre_dict in all_genres:
            for genre_name in genre_dict.values():
                genre_counts[genre_name] = genre_counts.get(genre_name, 0) + 1

        top_genres = sorted(genre_counts, key=genre_counts.get, reverse=True)[:10]

        if genre and genre in top_genres:
            df["Genres"] = df["Genres"].apply(ast.literal_eval)
            df = df[df["Genres"].apply(lambda g: genre in g.values())]

        release_counts = df["Year"].value_counts().reset_index()
        release_counts.columns = ["Year", "Movie Count"]
        release_counts = release_counts.sort_values(by="Year")

        return release_counts, top_genres

    def ages(self, mode: str = "Y") -> pd.DataFrame:
        """
        Returns a DataFrame with the number of births per year or month.
        """
        if not hasattr(self, 'characters'):
            raise ValueError("Character data is not loaded.")

        dob_column_index = 4
        if dob_column_index >= len(self.characters.columns):
            raise ValueError("Actor DOB column index is out of range.")

        dob_column = self.characters.columns[dob_column_index]

        self.characters[dob_column] = pd.to_datetime(self.characters[dob_column], errors='coerce')

        if mode.upper() == "M":
            self.characters["Month"] = self.characters[dob_column].dt.month
            births = self.characters["Month"].value_counts().reset_index()
            births.columns = ["Month", "Birth Count"]
            births = births.sort_values(by="Month")
        else:
            self.characters["Year"] = self.characters[dob_column].dt.year
            births = self.characters["Year"].value_counts().reset_index()
            births.columns = ["Year", "Birth Count"]
            births = births.sort_values(by="Year")

        return births


    def get_random_movie(self):
        """Returns a random movie title, its short summary, the plot summary, and genre data."""
        if self.movies is None or self.movies.empty:
            raise ValueError("Movie data is not loaded.")

        random_movie = self.movies.sample(1).iloc[0]
        movie_id = str(random_movie[0])
        title = random_movie[2]
        year = random_movie[3]
        year_str = str(year)[:4]

        summary = f"\"{title}\" ({year_str})"
        plot_summary = self.plot_summaries.get(movie_id, "Plot summary not available.")

        return title, summary, plot_summary, random_movie[8]


    def extract_genres(self, genre_column):
        """Extracts the real genre names from the dictionary-like genre column."""
        try:
            genre_dict = ast.literal_eval(genre_column)
            return list(genre_dict.values())

        except (ValueError, SyntaxError):
            return ["Unknown"]



    def classify_genre_with_llm(self, summary):
        """Sends a movie summary to Ollama (Mistral) and gets genre classification."""
        prompt = (
            "Analyze the following movie title and plot summary. Return only the genres as a comma-separated list."
            "Emphasize the plot summary. "
            "If the plot summary is not given 'Plot Summary: Plot summary not available.', then only focus on the title. "
            "Movies with a release year before 1930 are always Black-and-white (this is a genre). Movies with a release year before 1965 are more likely to be Black-and-white (this is a genre). Afterwards there are almost exclusively colour movies (this is not an explicit genre) "
            "Movies with a release year before 1930 are more likely to be Silent film (this is a genre). "
            "The predicted genre must not be a year (e.g. 1970). "
            "Example output: Action, Drama, Thriller. "
            "Do not include extra text or explanations.\n\n"
            f"{summary}"
        )

        response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"].strip()

    def check_genre_match(self, real_genres, predicted_genres):
        """Compares the actual genres with the predicted ones, ignoring the word 'film'."""
        def clean_genre(genre):
            return genre.lower().replace("film", "").strip()

        real_set = {clean_genre(g) for g in real_genres}
        predicted_set = {clean_genre(g) for g in predicted_genres.split(", ")}

        if real_set & predicted_set:
            return "✅ LLM correctly identified at least one genre!"
        return "❌ No match with actual genres."
