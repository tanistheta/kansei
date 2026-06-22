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
- **Image classify** — drop any photo, CLIP reads its visual DNA instantly, served by an isolated inference microservice (see [Architecture](#architecture) below)
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
| Main app deployment | Railway |
| Classify inference | Dockerized FastAPI microservice on GCP Compute Engine (e2-micro) |

---

## Architecture

The quiz flow and the classify flow share the same scoring pipeline (mean CLIP vector → cosine similarity → ranked aesthetics) but run on physically separate infrastructure, for a reason worth documenting rather than hiding.

**The problem:** CLIP's full load — model weights plus PyTorch's runtime overhead — peaks at measured **918MB RSS** (profiled stage-by-stage: imports alone cost ~316MB before any model touches memory; the fp32 weight load is the single largest jump, ~580MB in one step). Railway's container gives the whole app a shared 1GB ceiling. Loading CLIP inside the same process as the main app reliably triggered an OOM kill.

**What didn't work:** fp16 quantization (halves weight size on paper, but PyTorch's CPU build doesn't guarantee fp16 ops stay fp16 internally, and it doesn't touch the ~316MB import overhead anyway). Subprocess isolation alone (the OS reclaims subprocess memory on exit, but doesn't help if the *peak* during execution still exceeds the shared ceiling).

**What worked:** moving classify into its own container, on its own machine, with its own memory budget — fully decoupled from the main app's ceiling.

```
┌─────────────────┐         HTTPS, auth header        ┌──────────────────────┐
│  Railway          │ ──────────────────────────────▶ │  GCP e2-micro VM      │
│  (main FastAPI app)│                                  │  Docker container     │
│  quiz · result ·  │ ◀────────────────────────────── │  CLIP ViT-B/32 (fp16) │
│  explore · score  │      embedding (512-d JSON)       │  FastAPI + uvicorn    │
└─────────────────┘                                     └──────────────────────┘
```

Notes on the deployment, stated plainly rather than glossed over:

- The VM runs on GCP's free tier (e2-micro, 1GB RAM). That's *tighter* than the 918MB peak with zero margin for the OS and Docker daemon's own overhead, so the service runs with a 2GB swap file to absorb the difference. This trades occasional latency (a request that hits swapped memory can take longer) for not OOM-crashing — a deliberate, documented tradeoff rather than an oversight.
- The classify endpoint is gated by a shared-secret header, separate from the open `0.0.0.0/0` firewall rule — the firewall has to stay open since Railway's free tier has no static outbound IP to allowlist against instead, so the auth check is what actually keeps the endpoint from being usable by the constant bot traffic that scans every public IP on the internet.
- A real packaging bug surfaced during containerization: installing `torch` and `open-clip-torch` in separate `pip install` steps let pip's resolver silently pull in the full CUDA toolkit (~6GB of `nvidia-*` packages) to satisfy a loose version constraint, even though the service is CPU-only. Fixed by resolving all torch-related packages in a single `pip install` call against PyTorch's CPU-only wheel index — dropped the image from 9.2GB to 1.62GB.

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

### Running the classify service locally

The classify microservice lives in `classify_service/` as a separate Docker build:

```bash
cd classify_service
docker build -t kansei-classify .
docker run -d -p 8001:8001 -e CLASSIFY_AUTH_TOKEN="your-token-here" --name kansei-classify-container kansei-classify
```

The main app's `core.py` expects `CLASSIFY_SERVICE_URL` to point at wherever this is running, and `CLASSIFY_AUTH_TOKEN` (set as an environment variable, not hardcoded) to match.

---

## API

### Main app (`backend/`)

| Endpoint | Method | Description |
|---|---|---|
| `/api/pairs?n=25` | GET | Random image pairs for quiz |
| `/api/centroid/{aesthetic}` | GET | CLIP centroid vector for an aesthetic |
| `/api/score` | POST | Score user vectors against all 16 centroids |
| `/api/nearest` | POST | kNN — find closest images to user vector |
| `/api/umap` | POST | 3D UMAP projection of user position |
| `/api/classify` | POST | Classify an uploaded image (proxies to the classify service) |
| `/api/images/{aesthetic}` | GET | List images for an aesthetic |

### Classify service (`classify_service/`)

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/health` | GET | none | Liveness check |
| `/classify` | POST | `X-Kansei-Auth` header | CLIP inference on an uploaded image, returns a 512-d embedding |

---

## Project context

Built as a portfolio project exploring the intersection of aesthetic theory and ML embedding spaces. Uses CLIP's visual-semantic representations as a proxy for subjective taste — the hypothesis being that aesthetic preference clusters meaningfully in embedding space even without explicit labels.

Both the quiz and classify flows use the same underlying pipeline: user preference → mean CLIP vector → cosine similarity against aesthetic centroids → ranked output. They differ only in how that vector is produced, and — as of the classify feature's rebuild — in the infrastructure each one runs on.
