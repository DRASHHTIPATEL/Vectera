# Pushing to GitHub

I can’t authenticate as you, so this has to run on your machine.

**Using GitHub CLI** (if you use `gh`):

```bash
cd /Users/drashtipatel/Desktop/Vectera
gh auth login
gh repo create YOUR-REPO-NAME --private --source=. --remote=origin --push
```

**Using the website** — create an empty repo (no README), then:

```bash
cd /Users/drashtipatel/Desktop/Vectera
git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
git branch -M main
git push -u origin main
```

`.env` is gitignored so secrets stay local. Anyone cloning should run `cp .env.example .env` and fill in LLM + optional DB.

---

**Postgres note:** install Docker Desktop if you want `docker compose up -d` and set `DATABASE_URL` in `.env`.

Until then, SQLite + FAISS under `data/` is fine for demos — just say that in the video if asked.
