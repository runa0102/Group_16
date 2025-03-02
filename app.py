import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from movie_data_analyzer import MovieDataAnalyzer

# Initialize movie analyzer
analyzer = MovieDataAnalyzer()

# Streamlit UI
st.title("🎬 Movie Data Analysis App")
st.write("Analyze the CMU Movie Summaries dataset")

# Section 1: Top N Movie Genres
st.header("Most Common Movie Genres")
N = st.slider("Select number of genres:", min_value=1, max_value=20, value=10)
genre_df = analyzer.movie_type(N)

# Display table and bar chart
st.dataframe(genre_df)

fig, ax = plt.subplots()
ax.barh(genre_df["Movie_Type"], genre_df["Count"])
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
    ax.bar(actor_df["Number of Actors"], actor_df["Movie Count"])
    ax.set_xlabel("Number of Actors")
    ax.set_ylabel("Number of Movies")
    ax.set_title("Movies by Actor Count")
    st.pyplot(fig)

st.write("🎥 Data sourced from the CMU Movie Summaries dataset.")

