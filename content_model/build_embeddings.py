import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

MOVIES_PATH = "data/ml-latest-small/movies.csv"
TAGS_PATH = "data/ml-latest-small/tags.csv"
EMB_OUT = "data/content_embeddings.npy"
INDEX_OUT = "data/faiss.index"
IDS_OUT = "data/movie_ids.json"

MODEL_NAME = "all-MiniLM-L6-v2"


def load_movies():
    return pd.read_csv(MOVIES_PATH)


def _build_text_per_movie():

    movies = load_movies()
    tags = pd.read_csv(TAGS_PATH)

    # Collapse all tags for a movie into one space-separated string,
    # e.g. movieId 1 -> "pixar funny animated"
    tags_per_movie = (
        tags.groupby("movieId")["tag"]
        .apply(lambda t: " ".join(t.astype(str)))
        .rename("tags")
    )

    movies = movies.merge(tags_per_movie, on="movieId", how="left")
    movies["tags"] = movies["tags"].fillna("")
    movies["genres_clean"] = movies["genres"].str.replace("|", " ", regex=False)

    movies["text"] = (
        movies["title"] + " " + movies["genres_clean"] + " " + movies["tags"]
    )
    return movies[["movieId", "text"]]


def build():

    movie_text = _build_text_per_movie()
    print(f"Building embeddings for {len(movie_text)} movies")

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        movie_text["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    # Normalize so inner product == cosine similarity
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    np.save(EMB_OUT, embeddings)
    faiss.write_index(index, INDEX_OUT)
    with open(IDS_OUT, "w") as f:
        json.dump(movie_text["movieId"].tolist(), f)

    print(f"Saved embeddings {embeddings.shape} -> {EMB_OUT}")
    print(f"Saved FAISS index -> {INDEX_OUT}")


def get_similar_movies(movie_id: int, k: int = 10) -> list[tuple[int, float]]:

    embeddings = np.load(EMB_OUT)
    with open(IDS_OUT) as f:
        movie_ids = json.load(f)
    index = faiss.read_index(INDEX_OUT)

    row = movie_ids.index(movie_id)
    query = embeddings[row : row + 1].copy()
    faiss.normalize_L2(query)

    scores, neighbors = index.search(query, k + 1)  # +1 to drop self-match
    results = [
        (movie_ids[i], float(s))
        for i, s in zip(neighbors[0], scores[0])
        if movie_ids[i] != movie_id
    ]
    return results[:k]


if __name__ == "__main__":
    build()

    # quick sanity check after building
    test_movie_id = 1  # Toy Story
    movies = load_movies().set_index("movieId")["title"]
    print(f"\nMovies similar to '{movies.loc[test_movie_id]}':")
    for movie_id, score in get_similar_movies(test_movie_id, k=5):
        print(f"  {movies.loc[movie_id]}  (score={score:.3f})")