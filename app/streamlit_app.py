import os
import sys
import zipfile
import urllib.request
from pathlib import Path

# Add project root directory (movie_recsys) to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import numpy as np
import pandas as pd
import streamlit as st

# Data directory configuration
DATA_DIR = BASE_DIR / "data" / "ml-latest-small"
MOVIES_PATH = DATA_DIR / "movies.csv"
RATINGS_PATH = DATA_DIR / "ratings.csv"


def ensure_data_exists():
    """Auto-downloads and extracts MovieLens dataset if CSVs are missing."""
    if not MOVIES_PATH.exists() or not RATINGS_PATH.exists():
        st.info("Downloading MovieLens dataset for first-time setup...")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = BASE_DIR / "data" / "ml-latest-small.zip"

        url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
        urllib.request.urlretrieve(url, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(BASE_DIR / "data")

        if zip_path.exists():
            os.remove(zip_path)


# Download data if absent on Streamlit Cloud
ensure_data_exists()

from onboarding import (
    load_cf_model, get_popular_movies, fold_in_user, get_candidates, rank,
    get_bandit_for_user, record_feedback
)

st.set_page_config(page_title="Movie Recommender")
st.title("Movie Recommender")


@st.cache_data
def load_movies():
    return pd.read_csv(MOVIES_PATH)


@st.cache_data
def load_ratings():
    return pd.read_csv(RATINGS_PATH)


@st.cache_resource
def get_model():
    return load_cf_model()


@st.cache_data
def all_genres():
    movies = load_movies()
    genres = set()
    for g in movies["genres"].str.split("|"):
        genres.update(g)
    genres.discard("(no genres listed)")
    return sorted(genres)


def genre_filtered_movies(genres, n=50, min_ratings=20):
    movies = load_movies()
    ratings = load_ratings()
    stats = ratings.groupby("movieId")["rating"].agg(["count", "mean"])
    stats = stats[stats["count"] >= min_ratings]
    ids = stats.sort_values(["mean", "count"], ascending=False).index

    mask = movies["genres"].apply(lambda g: any(genre in g for genre in genres))
    filtered = movies[movies["movieId"].isin(ids) & mask]
    return filtered.head(n)


if "stage" not in st.session_state:
    st.session_state.stage = "genres"
if "ratings" not in st.session_state:
    st.session_state.ratings = {}
if "dismissed" not in st.session_state:
    st.session_state.dismissed = set()
if "user_id" not in st.session_state:
    st.session_state.user_id = int(np.random.randint(100000, 999999))
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = False


# Stage 1: Select Genres
if st.session_state.stage == "genres":
    st.subheader("Pick 3 genres you enjoy")
    selected = st.multiselect("Genres", all_genres(), max_selections=3)
    if len(selected) == 3 and st.button("Continue"):
        st.session_state.genres = selected
        st.session_state.stage = "rate_required"
        st.rerun()

# Stage 2: Rate 10 Movies with "Haven't Seen" Swap
elif st.session_state.stage == "rate_required":
    REQUIRED_COUNT = 10
    st.subheader(f"Rate {REQUIRED_COUNT} movies to calibrate your profile")

    pool = genre_filtered_movies(st.session_state.genres, n=50)
    available_candidates = pool[~pool["movieId"].isin(st.session_state.dismissed)]
    candidates = available_candidates.head(REQUIRED_COUNT)

    current_ratings = {}

    for idx, (_, row) in enumerate(candidates.iterrows(), 1):
        movie_id = row["movieId"]
        title = row["title"]

        col_title, col_btn = st.columns([3, 1])
        col_title.markdown(f"**{idx}. {title}**")

        if col_btn.button("Haven't seen", key=f"swap_{movie_id}"):
            st.session_state.dismissed.add(movie_id)
            st.rerun()

        val = st.slider(
            f"Rating for {title}",
            min_value=0.5,
            max_value=5.0,
            value=3.0,
            step=0.5,
            key=f"req_{movie_id}",
            label_visibility="collapsed"
        )
        current_ratings[movie_id] = val
        st.write("---")

    displayed_count = len(candidates)
    st.progress(displayed_count / REQUIRED_COUNT)
    st.caption(f"Showing {displayed_count} of {REQUIRED_COUNT} movies")

    if displayed_count < REQUIRED_COUNT:
        st.warning("You've dismissed too many movies! Try adding more genres or starting over.")

    if st.button("Continue to Recommendations") and displayed_count == REQUIRED_COUNT:
        st.session_state.ratings.update(current_ratings)
        st.session_state.stage = "rate_more"
        st.rerun()

# Stage 3: Rate Optional Movies
elif st.session_state.stage == "rate_more":
    st.subheader("Optional: rate more movies for even better precision")
    movies = load_movies()
    pool_ids = get_popular_movies(n=20)
    already = set(st.session_state.ratings.keys()).union(st.session_state.dismissed)
    pool = movies[movies["movieId"].isin(pool_ids) & ~movies["movieId"].isin(already)]

    for _, row in pool.iterrows():
        seen = st.checkbox(f"I've seen: {row['title']}", key=f"seen_{row['movieId']}")
        if seen:
            rating = st.slider(f"Your rating for {row['title']}", 0.5, 5.0, 3.0, 0.5, key=f"more_{row['movieId']}")
            st.session_state.ratings[row["movieId"]] = rating

    if st.button("Get my recommendations"):
        st.session_state.stage = "recommend"
        st.rerun()

# Stage 4: Recommendations
elif st.session_state.stage == "recommend":
    st.subheader("Your recommendations")

    rated_movies = list(st.session_state.ratings.items())
    model = get_model()
    pu = fold_in_user(model, rated_movies)

    rated_ids = set(m for m, _ in rated_movies).union(st.session_state.dismissed)
    top_movie_id = max(rated_movies, key=lambda x: x[1])[0]

    candidates, similarity_lookup = get_candidates(model, pu, rated_ids, top_movie_id)
    ranked = rank(model, pu, candidates, similarity_lookup)
    top_set = ranked[:10]

    if "picked" not in st.session_state:
        bandit = get_bandit_for_user(st.session_state.user_id, context_dim=model.qi.shape[1])
        st.session_state.picked = bandit.select(top_set, pu)
        st.session_state.last_context = pu

    picked = st.session_state.picked
    movies = load_movies().set_index("movieId")

    st.write(f"### Top pick: {movies.loc[picked, 'title']}")
    st.caption(movies.loc[picked, "genres"])

    st.write("Other strong candidates:")
    for movie_id in top_set:
        if movie_id == picked:
            continue
        st.write(f"- {movies.loc[movie_id, 'title']}  ({movies.loc[movie_id, 'genres']})")

    st.divider()
    st.write("Did you like the top pick?")
    col1, col2 = st.columns(2)

    if not st.session_state.feedback_given:
        if col1.button("👍 Yes"):
            record_feedback(st.session_state.user_id, picked, pu, 4.5)
            st.session_state.feedback_given = True
            st.success("Thanks! This will improve future picks.")
            st.rerun()

        if col2.button("👎 No"):
            record_feedback(st.session_state.user_id, picked, pu, 1.5)
            st.session_state.feedback_given = True
            st.success("Thanks! This will improve future picks.")
            st.rerun()
    else:
        st.info("Feedback recorded!")

    if st.button("Start over"):
        st.session_state.clear()
        st.rerun()