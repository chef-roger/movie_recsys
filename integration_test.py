import pickle
import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader

from content_model.build_embeddings import get_similar_movies

REAL_MOVIES_PATH = "data/ml-latest-small/movies.csv"
REAL_RATINGS_PATH = "data/ml-latest-small/ratings.csv"
SYNTH_RATINGS_PATH = "data/synthetic_ratings.csv"
SYNTH_CF_MODEL_PATH = "data/synthetic_cf_model.pkl"

N_USERS = 10
RATINGS_PER_USER = 5
POPULAR_POOL_SIZE = 500


def get_popular_movies(n=POPULAR_POOL_SIZE, min_ratings=20):
    ratings = pd.read_csv(REAL_RATINGS_PATH)
    stats = ratings.groupby("movieId")["rating"].agg(["count", "mean"])
    stats = stats[stats["count"] >= min_ratings]
    stats = stats.sort_values("count", ascending=False)
    return stats.head(n).index.tolist()


def generate_synthetic_users():
    movies = pd.read_csv(REAL_MOVIES_PATH)
    popular_ids = get_popular_movies()
    popular = movies[movies["movieId"].isin(popular_ids)]

    genres = popular["genres"].str.split("|").explode().dropna().unique()
    genres = [g for g in genres if g not in ("(no genres listed)",)]

    rng = np.random.default_rng(0)
    rows = []
    assigned_genre = {}
    for user_id in range(1001, 1001 + N_USERS):
        preferred_genre = rng.choice(genres)
        assigned_genre[user_id] = preferred_genre
        liked = popular[popular["genres"].str.contains(preferred_genre, regex=False)]
        others = popular[~popular["genres"].str.contains(preferred_genre, regex=False)]

        n_liked = min(len(liked), RATINGS_PER_USER // 2)
        n_others = min(len(others), RATINGS_PER_USER - n_liked)

        liked_sample = liked.sample(n_liked, random_state=user_id)
        other_sample = others.sample(n_others, random_state=user_id)

        for _, row in liked_sample.iterrows():
            rows.append({"userId": user_id, "movieId": row["movieId"],
                         "rating": rng.choice([4.0, 4.5, 5.0])})
        for _, row in other_sample.iterrows():
            rows.append({"userId": user_id, "movieId": row["movieId"],
                         "rating": rng.choice([1.0, 2.0, 2.5, 3.0])})

    df = pd.DataFrame(rows)
    df.to_csv(SYNTH_RATINGS_PATH, index=False)
    return df, assigned_genre


def train_synthetic_cf(ratings_df):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings_df[["userId", "movieId", "rating"]], reader)
    trainset = data.build_full_trainset()
    model = SVD(n_factors=20, reg_all=0.08, random_state=42)
    model.fit(trainset)
    with open(SYNTH_CF_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    return model


def get_user_context(model, user_id):
    inner_uid = model.trainset.to_inner_uid(user_id)
    return model.pu[inner_uid]


def get_candidates_synthetic(model, ratings_df, user_id, k=30):
    all_movie_ids = ratings_df["movieId"].unique()
    already_rated = set(ratings_df[ratings_df["userId"] == user_id]["movieId"])
    unrated = [m for m in all_movie_ids if m not in already_rated]

    cf_scores = [(m, model.predict(user_id, m).est) for m in unrated]
    cf_scores.sort(key=lambda x: x[1], reverse=True)
    cf_ids = [m for m, _ in cf_scores[:k // 2]]

    top_movie = ratings_df[ratings_df["userId"] == user_id].sort_values(
        "rating", ascending=False).iloc[0]["movieId"]
    content_results = get_similar_movies(int(top_movie), k=k // 2)
    content_ids = [m for m, _ in content_results]
    similarity_lookup = {m: s for m, s in content_results}

    combined = []
    seen = set()
    for m in cf_ids + content_ids:
        if m not in seen:
            seen.add(m)
            combined.append(m)
    return combined, similarity_lookup


def rank_candidates(model, ratings_df, user_id, candidates, similarity_lookup):
    n_ratings = len(ratings_df[ratings_df["userId"] == user_id])
    cf_weight = min(0.7, 0.2 + 0.02 * n_ratings)
    content_weight = 1 - cf_weight

    scored = []
    for movie_id in candidates:
        cf_score = model.predict(user_id, movie_id).est
        if movie_id in similarity_lookup:
            content_score = similarity_lookup[movie_id] * 4 + 1
            score = cf_weight * cf_score + content_weight * content_score
        else:
            score = cf_score
        scored.append((movie_id, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [m for m, _ in scored]


def run():
    ratings_df, assigned_genre = generate_synthetic_users()
    model = train_synthetic_cf(ratings_df)
    movies_df = pd.read_csv(REAL_MOVIES_PATH).set_index("movieId")
    movies = movies_df["title"]

    for user_id in range(1001, 1001 + N_USERS):
        candidates, similarity_lookup = get_candidates_synthetic(model, ratings_df, user_id, k=30)
        ranked = rank_candidates(model, ratings_df, user_id, candidates, similarity_lookup)
        picks = ranked[:5]

        print(f"\nUser {user_id} (preferred genre: {assigned_genre[user_id]}):")
        context = get_user_context(model, user_id)
        print(f"  context vector: {np.round(context, 2)}")
        for movie_id in picks:
            title = movies.loc[movie_id]
            movie_genres = movies_df.loc[movie_id, "genres"]
            print(f"  {title}  [{movie_genres}]")


if __name__ == "__main__":
    run()