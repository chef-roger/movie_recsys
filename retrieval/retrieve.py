from cf_model.train_cf import top_cf_candidates, load_ratings
from content_model.build_embeddings import get_similar_movies


def get_candidates(user_id: int, k: int = 200) -> list[int]:

    cf_results = top_cf_candidates(user_id, k // 2)
    cf_ids = [movie_id for movie_id, _ in cf_results]

    similar_ids = []
    ratings = load_ratings()
    user_ratings = ratings[ratings["userId"] == user_id]
    if not user_ratings.empty:
        top_movie_id = int(
            user_ratings.sort_values("rating", ascending=False).iloc[0]["movieId"]
        )
        similar_results = get_similar_movies(top_movie_id, k=k // 2)
        similar_ids = [movie_id for movie_id, _ in similar_results]

    combined = []
    seen = set()
    for movie_id in cf_ids + similar_ids:
        if movie_id not in seen:
            seen.add(movie_id)
            combined.append(movie_id)

    return combined[:k]


def _debug_get_candidates_with_source(user_id: int, k: int = 200):

    cf_results = top_cf_candidates(user_id, k // 2)
    cf_ids = [m for m, _ in cf_results]

    similar_ids = []
    ratings = load_ratings()
    user_ratings = ratings[ratings["userId"] == user_id]
    top_movie_id = None
    if not user_ratings.empty:
        top_movie_id = int(
            user_ratings.sort_values("rating", ascending=False).iloc[0]["movieId"]
        )
        similar_results = get_similar_movies(top_movie_id, k=k // 2)
        similar_ids = [m for m, _ in similar_results]

    combined = []
    seen = set()
    for movie_id in cf_ids:
        if movie_id not in seen:
            seen.add(movie_id)
            combined.append((movie_id, "CF"))
    for movie_id in similar_ids:
        if movie_id not in seen:
            seen.add(movie_id)
            combined.append((movie_id, "content"))

    return combined, top_movie_id


if __name__ == "__main__":
    import pandas as pd

    test_user = 1
    movies = pd.read_csv("data/ml-latest-small/movies.csv").set_index("movieId")

    labeled, seed_movie_id = _debug_get_candidates_with_source(test_user, k=200)

    if seed_movie_id is not None:
        print(f"User {test_user}'s top-rated movie: {movies.loc[seed_movie_id, 'title']}\n")

    print(f"First 10 CF candidates for user {test_user}:")
    for movie_id, src in [x for x in labeled if x[1] == "CF"][:10]:
        print(f"  [{src}] {movies.loc[movie_id, 'title']}")

    print(f"\nFirst 10 content-based candidates for user {test_user}:")
    for movie_id, src in [x for x in labeled if x[1] == "content"][:10]:
        print(f"  [{src}] {movies.loc[movie_id, 'title']}")