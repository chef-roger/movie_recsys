"""
Person A owns this file (Phase 4).

Blends CF score + content similarity into a single ranked list.
Starts simple (weighted sum), can graduate to LightGBM later.
"""
from cf_model.train_cf import predict_rating
from content_model.build_embeddings import get_similar_movies
from retrieval.retrieve import get_candidates
from cf_model.train_cf import load_ratings


def rank(user_id: int, candidate_movie_ids: list[int]) -> list[int]:
    """
    Score every candidate on one common scale, then sort descending.

    CF candidates already have a predicted rating (1-5 scale) - use it
    directly. Content-only candidates only have a similarity score
    (0-1 scale) - stretch it onto the same 1-5 scale so the two are
    comparable: similarity 0 -> ~1, similarity 1 -> 5.
    """
    ratings = load_ratings()
    already_rated = set(ratings[ratings["userId"] == user_id]["movieId"])

    # Get the user's top movie + its similarity scores, so we can look up
    # a similarity score for any content-only candidate in our list.
    user_ratings = ratings[ratings["userId"] == user_id]
    similarity_lookup = {}
    if not user_ratings.empty:
        top_movie_id = int(
            user_ratings.sort_values("rating", ascending=False).iloc[0]["movieId"]
        )
        for movie_id, sim in get_similar_movies(top_movie_id, k=len(candidate_movie_ids)):
            similarity_lookup[movie_id] = sim

    scored = []
    for movie_id in candidate_movie_ids:
        if movie_id in already_rated:
            continue  # never recommend something they've already seen

        # Prefer a real CF prediction, blended with content similarity when
        # we have it; otherwise use CF alone.
        cf_score = predict_rating(user_id, movie_id)
        if movie_id in similarity_lookup:
            content_score = similarity_lookup[movie_id] * 4 + 1  # 0-1 -> 1-5
            # Blend: trust CF more since it's personalized, content less
            score = 0.7 * cf_score + 0.3 * content_score
        else:
            score = cf_score

        scored.append((movie_id, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [movie_id for movie_id, _ in scored]


if __name__ == "__main__":
    import pandas as pd
    from cf_model.train_cf import predict_rating

    test_user = 1
    movies = pd.read_csv("data/ml-latest-small/movies.csv").set_index("movieId")

    candidates = get_candidates(test_user, k=50)
    ranked = rank(test_user, candidates)

    print(f"Top 15 ranked recommendations for user {test_user}:")
    for movie_id in ranked[:15]:
        cf_score = predict_rating(test_user, movie_id)
        print(f"  {movies.loc[movie_id, 'title']}  cf={cf_score:.2f}")