import json
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

EMBEDDINGS_PATH = "data/content_embeddings.npy"
INDEX_PATH = "data/faiss.index"
MOVIE_IDS_PATH = "data/movie_ids.json"
MOVIES_PATH = "data/ml-latest-small/movies.csv"

# Global lazy-loaded cache
_INDEX = None
_MOVIE_IDS = None
_ID_TO_INDEX = None


def load_content_resources():
    global _INDEX, _MOVIE_IDS, _ID_TO_INDEX
    if _INDEX is None:
        _INDEX = faiss.read_index(INDEX_PATH)
        with open(MOVIE_IDS_PATH, "r") as f:
            _MOVIE_IDS = json.load(f)
        _ID_TO_INDEX = {movie_id: idx for idx, movie_id in enumerate(_MOVIE_IDS)}
    return _INDEX, _MOVIE_IDS, _ID_TO_INDEX


def get_similar_movies(movie_id, k=15):
    index, movie_ids, id_to_index = load_content_resources()
    
    if movie_id not in id_to_index:
        return []
    
    idx = id_to_index[movie_id]
    query_vector = index.reconstruct(idx).reshape(1, -1)
    
    # Query FAISS index (fetch k+1 to exclude the query movie itself)
    similarities, indices = index.search(query_vector, k + 1)
    
    results = []
    for sim, i in zip(similarities[0], indices[0]):
        cand_id = movie_ids[i]
        if cand_id != movie_id:
            results.append((cand_id, float(sim)))
            if len(results) == k:
                break
                
    return results


def build_embeddings():
    movies = pd.read_csv(MOVIES_PATH)
    movies["genres_clean"] = movies["genres"].str.replace("|", " ", regex=False)
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(movies["genres_clean"].tolist(), show_progress_bar=True, convert_to_numpy=True)
    
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    
    np.save(EMBEDDINGS_PATH, embeddings)
    faiss.write_index(index, INDEX_PATH)
    with open(MOVIE_IDS_PATH, "w") as f:
        json.dump(movies["movieId"].tolist(), f)


if __name__ == "__main__":
    build_embeddings()