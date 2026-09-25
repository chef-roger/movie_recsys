# 🎬 Hybrid Movie Recommender System

A content-based, collaborative filtering, and contextual bandit (LinUCB) hybrid movie recommendation engine built with Streamlit, Scikit-Surprise, and FAISS.

---

## 🚀 Quickstart (Local Setup)

### 1. Clone the Repository
git clone https://github.com/chef-roger/movie_recsys.git
cd movie_recsys

### 2. Set Up Virtual Environment & Dependencies
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt

### 3. Run the Streamlit App
streamlit run app/streamlit_app.py

The app will launch automatically in your web browser at http://localhost:8501.

---

## 🛠️ How It Works

1. User Pick & Rating Stage: Users select at least 3 movies and assign ratings (1.0–5.0).
2. User Fold-In: Estimates user latent factor vectors (pu) dynamically using ridge regression without retraining the underlying SVD model.
3. Candidate Search & Hybrid Ranking: Merges sentence-transformer cosine vector similarities (via FAISS) with Collaborative Filtering rating predictions.
4. LinUCB Hero Selection: Dynamically surfaces a #1 recommendation using Upper Confidence Bound contextual exploitation/exploration with live feedback recording.