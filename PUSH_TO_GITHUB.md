# Push this repo to GitHub (I can’t log in as you — run these locally)

## Option A — GitHub CLI (`gh`)

```bash
cd /Users/drashtipatel/Desktop/Vectera
gh auth login
gh repo create YOUR-REPO-NAME --private --source=. --remote=origin --push
```

## Option B — Web UI

1. Create a **new empty repository** on GitHub (no README).
2. Then:

```bash
cd /Users/drashtipatel/Desktop/Vectera
git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
git branch -M main
git push -u origin main
```

`.env` stays **local** (gitignored). Reviewers clone and run `cp .env.example .env`.

## Postgres on your machine

**Docker was not available** in the automated setup environment. To use the assessment Postgres path:

1. Install **Docker Desktop** for Mac.
2. `docker compose up -d`
3. Uncomment `DATABASE_URL` in `.env`, restart Streamlit, **re-index** your PDFs.

Until then, the app uses **SQLite + FAISS** under `data/` (valid for local demos).
