# Movie Recommender (CF + Content + Bandit)

## What's here
- `interfaces.py` — the shared contract. Read this first, every time you're
  unsure who owns what or how pieces connect.
- `cf_model/`, `ranking/`, `evaluation/eval_cf.py` — **Person A**
- `content_model/`, `bandit/`, `evaluation/eval_bandit.py` — **Person B**
- `retrieval/`, `app/` — built jointly, at the checkpoints noted in the code
- `data/download.py` — run this after cloning, don't commit the raw CSVs

## First-time setup (both people)
```bash
git clone <repo-url>
cd movie_recsys
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python data/download.py
```

## Git workflow, step by step

**One person only — create the repo:**
1. Go to github.com → New repository → name it `movie_recsys` → don't
   initialize with a README (we already have one) → Create.
2. On your machine, in this folder:
   ```bash
   git init
   git add .
   git commit -m "Initial scaffold: folder structure + interface contract"
   git branch -M main
   git remote add origin <the-url-github-gave-you>
   git push -u origin main
   ```
3. On GitHub: Settings → Collaborators → add your partner's GitHub username.

**Both people, for every piece of work after that:**
```bash
git checkout main
git pull                          # get latest before starting new work
git checkout -b feature/cf-model  # or feature/content-model, etc.

# ... do your work, e.g. filling in cf_model/train_cf.py ...

git add .
git commit -m "Implement CF training and predict_rating"
git push -u origin feature/cf-model
```
Then on GitHub: open a Pull Request from your branch into `main`, the other
person reviews/merges it. This is what stops you from silently overwriting
each other's work.

**Rule of thumb:** never commit directly to `main`. Always branch, push,
open a PR, merge. Takes an extra minute, saves you a merge-conflict
afternoon.

## Build order
See the docstrings in each file — every stub has a numbered TODO explaining
exactly what to implement and in what order. Work through them in this
sequence: `cf_model` / `content_model` (parallel) → `evaluation/eval_cf.py`
→ `retrieval` (joint) → `ranking` → `bandit` → `evaluation/eval_bandit.py`
→ `app/streamlit_app.py` (joint).
