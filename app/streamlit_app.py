import sys
import os
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from onboarding import (
    load_cf_model,
    fold_in_user,
    predict_with_folded_in,
    get_candidates_from_picks,
    rank,
    get_bandit_for_user,
    record_feedback
)

MOVIES_PATH = "data/ml-latest-small/movies.csv"
RATINGS_PATH = "data/ml-latest-small/ratings.csv"

st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide"
)

# --- Resource Caching ---
@st.cache_resource
def get_cf_model():
    return load_cf_model()

@st.cache_data
def load_movie_data():
    return pd.read_csv(MOVIES_PATH)

@st.cache_data
def get_popular_movies(min_ratings=30, top_n=50):
    ratings = pd.read_csv(RATINGS_PATH)
    stats = ratings.groupby("movieId")["rating"].agg(["count", "mean"])
    stats = stats[stats["count"] >= min_ratings]
    stats = stats.sort_values(["mean", "count"], ascending=False)
    return stats.head(top_n).index.tolist()

cf_model = get_cf_model()
movies_df = load_movie_data()
popular_ids = get_popular_movies()

movie_lookup = movies_df.set_index("movieId")["title"].to_dict()
genre_lookup = movies_df.set_index("movieId")["genres"].to_dict()

st.title("🎬 Hybrid Movie Recommender System")
st.markdown("Pick at least 3 movies, rate them, and explore personalized hybrid recommendations driven by LinUCB.")

st.divider()

# --- Step 1: Movie Selection ---
st.header("Step 1: Pick at least 3 movies")

popular_options = movies_df[movies_df["movieId"].isin(popular_ids)]["title"].tolist()
all_options = movies_df["title"].tolist()

selection_mode = st.radio("Selection Mode:", ["Pick from Popular Movies", "Search All Movies"], horizontal=True)

if selection_mode == "Pick from Popular Movies":
    selected_titles = st.multiselect("Select movies:", options=popular_options, placeholder="Choose 3 or more movies...")
else:
    selected_titles = st.multiselect("Search movies:", options=all_options, placeholder="Type to search...")

selected_movies_df = movies_df[movies_df["title"].isin(selected_titles)]

if len(selected_titles) < 3:
    st.info(f" Select at least **3** movies to unlock recommendations. Currently selected: {len(selected_titles)}")
else:
    st.success(f"✓ Selected {len(selected_titles)} movies.")
    st.divider()

    # --- Step 2: Rate Selected Movies ---
    st.header("Step 2: Rate your choices")
    user_ratings = []
    cols = st.columns(min(3, len(selected_movies_df)))
    
    for idx, (_, row) in enumerate(selected_movies_df.iterrows()):
        col = cols[idx % len(cols)]
        with col:
            st.markdown(f"**{row['title']}**")
            rating = st.slider("Rating", 1.0, 5.0, 4.0, 0.5, key=f"rating_{row['movieId']}")
            user_ratings.append((row["movieId"], rating))

    st.divider()

    # --- Step 3: Recommendation & LinUCB Selection ---
    st.header("Step 3: Get Recommendations")

    if st.button(" Generate Recommendations", type="primary"):
        st.session_state["generated"] = True
        
        with st.spinner("Computing hybrid vector recommendations..."):
            pu = fold_in_user(cf_model, user_ratings)
            picked_ids = [m[0] for m in user_ratings]
            candidates, similarity_lookup = get_candidates_from_picks(cf_model, pu, picked_ids, k=30)
            ranked_movie_ids = rank(cf_model, pu, candidates, similarity_lookup)
            
            st.session_state["pu"] = pu
            st.session_state["ranked_movie_ids"] = ranked_movie_ids
            st.session_state["similarity_lookup"] = similarity_lookup

    if st.session_state.get("generated", False):
        pu = st.session_state["pu"]
        top_candidates = st.session_state["ranked_movie_ids"][:10]
        similarity_lookup = st.session_state["similarity_lookup"]

        # Fetch trained LinUCB bandit
        bandit = get_bandit_for_user("default_user", context_dim=len(pu))
        hero_pick_id = bandit.select(top_candidates, pu)

        # --- Hero Pick Display ---
        st.subheader(" Bandit's #1 Pick")
        hero_title = movie_lookup.get(hero_pick_id, f"Movie #{hero_pick_id}")
        hero_genres = genre_lookup.get(hero_pick_id, "").replace("|", ", ")

        with st.container(border=True):
            st.markdown(f"##  {hero_title}")
            st.caption(f"**Genres:** {hero_genres}")
            st.write("Selected dynamically via **LinUCB Contextual Bandit** exploration/exploitation.")
            
            c1, c2 = st.columns(2)
            with c1:
                if st.button(" Loved it (Reward = 1.0)", key="reward_high"):
                    record_feedback("default_user", hero_pick_id, pu, reward=1.0)
                    st.toast("Updated LinUCB weights with positive feedback!")
            with c2:
                if st.button(" Not for me (Reward = 0.0)", key="reward_low"):
                    record_feedback("default_user", hero_pick_id, pu, reward=0.0)
                    st.toast("Updated LinUCB weights with negative feedback!")

        st.divider()

        # --- Full Recommendations List ---
        st.subheader(" Other Top Recommendations")
        rec_cols = st.columns(2)
        for idx, movie_id in enumerate(top_candidates):
            if movie_id == hero_pick_id:
                continue
            
            col = rec_cols[idx % 2]
            title = movie_lookup.get(movie_id, f"Movie #{movie_id}")
            genres = genre_lookup.get(movie_id, "").replace("|", ", ")
            pred_score = predict_with_folded_in(cf_model, pu, movie_id)

            with col:
                with st.container(border=True):
                    st.markdown(f"### {idx + 1}. {title}")
                    st.caption(f"**Genres:** {genres}")
                    st.write(f" **Predicted rating:** `{pred_score:.2f} / 5.0`")