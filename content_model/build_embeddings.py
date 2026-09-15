"""
Person B owns this file.

Builds content embeddings from movies.csv (genres/tags) and a FAISS index,
implementing get_similar_movies() from interfaces.py.

Usage: python content_model/build_embeddings.py
"""
import pandas as pd
# from sentence_transformers import SentenceTransformer
# import faiss

MOVIES_PATH = "data/ml-latest-small/movies.csv"
TAGS_PATH = "data/ml-latest-small/tags.csv"
EMB_OUT = "data/content_embeddings.npy"
INDEX_OUT = "data/faiss.index"
IDS_OUT = "data/movie_ids.json"


def load_movies():
    return pd.read_csv(MOVIES_PATH)


def build():
    """
    TODO (Person B):
    1. Load movies.csv, optionally merge in tags.csv (concat tags per movie_id)
    2. Build a text string per movie: title + genres (+ tags)
    3. Embed with sentence-transformers (or TfidfVectorizer if going non-neural)
    4. Normalize + build a FAISS IndexFlatIP
    5. Save embeddings, index, and movie_id order to disk
    """
    raise NotImplementedError


def get_similar_movies(movie_id: int, k: int = 10) -> list[tuple[int, float]]:
    """TODO: load FAISS index, look up movie_id's vector, search for k nearest."""
    raise NotImplementedError


if __name__ == "__main__":
    build()
