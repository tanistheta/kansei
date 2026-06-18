# Kansei (感性)

> Discover your aesthetic identity through visual preference.

**[kansei.up.railway.app](https://kansei.up.railway.app)**

Kansei maps your taste into CLIP's semantic embedding space through 25 image pair comparisons, then tells you where you live — your blend of 16 aesthetic identities, visualized in 3D.

---

## How it works

1. You pick between 25 image pairs
2. Each pick accumulates a preference vector in CLIP's 512-dimensional embedding space
3. Your vector is compared against 16 aesthetic cluster centroids via cosine similarity
4. You get a primary aesthetic + full breakdown with percentages
5. Your position is visualized in a 3D UMAP projection of the entire image space
6. kNN finds the 5 individual images closest to your taste vector
7. Rejection analysis tracks what you actively avoided

---

## Features

- **Quiz** — 25 image pairs, centroids preloaded at init for zero-latency choices
- **Image classify** — drop any photo, CLIP reads its visual DNA instantly
- **Result page** — aesthetic DNA bar, 3D UMAP plot, Fibonacci score grid, consistency score, rejection analysis, nearest images
- **Explore** — browse all 16 aesthetics with descriptions and images
- **Share card** — receipt-style downloadable card with your aesthetic breakdown

---

## Aesthetics

Afrofuturism · Boho · Brutalism · Coastal Cool · Cottagecore · Cyber Minimalism · Cyberpunk · Dark Academia · Ethereal · Glam Maximalism · Quiet Luxury · Solarpunk · Terracotta Modernism · Vintage Americana · Wabi Sabi · Zen Modern

---

## Stack

| Layer | Tech |
|---|---|
| Embeddings | CLIP (ViT-B/32) via `open-clip-torch` |
| Dimensionality reduction | UMAP (3D projection) |
| Scoring | Cosine similarity + cubic amplification curve |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS |
| Visualization | Plotly (3D scatter) |
| Deployment | Railway |

---

## Local setup

```bash
git clone https://github.com/tanistheta/kansei.git
cd kansei/backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000`

### Image structure

Images live in `backend/images/<aesthetic>/`. The CLIP embeddings are precomputed and stored in `backend/kansei_embeddings.json`. To recompute after adding images:

```bash
python scripts/precompute.py
```

---

## API

| Endpoint | Method | Description |
|---|---|---|
| `/api/pairs?n=25` | GET | Random image pairs for quiz |
| `/api/centroid/{aesthetic}` | GET | CLIP centroid vector for an aesthetic |
| `/api/score` | POST | Score user vectors against all 16 centroids |
| `/api/nearest` | POST | kNN — find closest images to user vector |
| `/api/umap` | POST | 3D UMAP projection of user position |
| `/api/classify` | POST | Classify an uploaded image |
| `/api/images/{aesthetic}` | GET | List images for an aesthetic |

---

## Project context

Built as a portfolio project exploring the intersection of aesthetic theory and ML embedding spaces. Uses CLIP's visual-semantic representations as a proxy for subjective taste — the hypothesis being that aesthetic preference clusters meaningfully in embedding space even without explicit labels.

Both the quiz and classify flows use the same underlying pipeline: user preference → mean CLIP vector → cosine similarity against aesthetic centroids → ranked output.