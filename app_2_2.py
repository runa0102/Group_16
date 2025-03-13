import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from movie_data_analyzer_2_2 import MovieDataAnalyzer

# Initialize movie analyzer
analyzer = MovieDataAnalyzer()

# Page selection
page = st.sidebar.radio("Select Page", ["Movie Genres & Actors", "Chronological Data", "Genre Classification"])

if page == "Movie Genres & Actors":
    st.title("Movie Data Analysis App")
    st.write("Analyze the CMU Movie Summaries dataset")

    # Section 1: Top N Movie Genres
    st.header("Most Common Movie Genres")
    N = st.slider("Select number of genres:", min_value=1, max_value=20, value=10)
    genre_df = analyzer.movie_type(N)

    # Display table and histogram
    st.dataframe(genre_df)

    fig, ax = plt.subplots()
    ax.hist(genre_df["Count"], bins=10, edgecolor='black')
    ax.set_xlabel("Number of Movies")
    ax.set_ylabel("Movie Genre")
    ax.set_title("Top Movie Genres")
    st.pyplot(fig)

    # Section 2: Actor Count Histogram
    st.header("Actor Count Distribution")
    actor_df = analyzer.actor_count()

    if actor_df.empty:
        st.warning("No actor data available.")
    else:
        st.dataframe(actor_df)

        fig, ax = plt.subplots()
        ax.hist(actor_df["Number of Actors"], bins=10, edgecolor='black')
        ax.set_xlabel("Number of Actors")
        ax.set_ylabel("Number of Movies")
        ax.set_title("Movies by Actor Count")
        st.pyplot(fig)

elif page == "Chronological Data":
    st.title("Movie Chronology Analysis")

    # Section 1: Releases per Year
    # Fetch release data and top genres
    release_df, top_genres = analyzer.releases()

    # Add genre selection dropdown in Streamlit
    genre = st.selectbox("Select Genre", ["All"] + top_genres)

    # Get filtered movie release data
    release_df, _ = analyzer.releases(None if genre == "All" else genre)

    if not release_df.empty:
        st.dataframe(release_df)

        # Bar plot for movie releases over time
        fig, ax = plt.subplots()
        ax.bar(release_df["Year"], release_df["Movie Count"], color="blue", edgecolor="black")
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Movies")
        ax.set_title(f"Movie Releases ({genre}) Over Time")
        st.pyplot(fig)

    # Section 2: Births per Year or Month
    st.header("Actor Births Distribution")
    birth_mode = st.selectbox("Select Time Unit", ["Year", "Month"])
    birth_df = analyzer.ages("Y" if birth_mode == "Year" else "M")

    if not birth_df.empty:
        st.dataframe(birth_df)
        fig, ax = plt.subplots()
        ax.bar(birth_df.iloc[:, 0], birth_df["Birth Count"], color="green", edgecolor="black")
        ax.set_xlabel(birth_mode)
        ax.set_ylabel("Number of Births")
        ax.set_title(f"Actor Births Per {birth_mode}")
        st.pyplot(fig)

  elif page == "Genre Classification":
    st.title("🎬 Genre Classification with AI")

    # Load a random movie from the dataset
    if st.button("🔀 Shuffle"):
        movie = analyzer.get_random_movie()

        if movie is not None:
            # Extract movie details
            title, summary, genre_column = movie
            real_genres = analyzer.extract_genres(genre_column)

            # Get LLM genre classification
            predicted_genres = analyzer.classify_genre_with_llm(summary)

            # Compare real and predicted genres
            match_result = analyzer.check_genre_match(real_genres, predicted_genres)

            # Display information
            st.subheader("🎥 Random Movie")
            st.text_area("Title & Summary", summary, height=100)

            st.subheader("📂 Actual Genres (Database)")
            st.text_area("Real Genres", ", ".join(real_genres), height=50)

            st.subheader("🤖 LLM Predicted Genres")
            st.text_area("Predicted Genres", predicted_genres, height=50)

            st.subheader("🔍 Genre Match")
            st.write(match_result)
