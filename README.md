# Kansei (感性)

Discover your aesthetic identity through a 25-image-pair quiz. Choices are encoded with CLIP, mapped against 16 aesthetic centroids in embedding space, and visualized with UMAP. You can also upload any image and have it classified directly.

**Live:** [kansei.duckdns.org](https://kansei.duckdns.org)

---

## What this is

Kansei takes a simple idea — "what pulls you in, visually?" — and backs it with real embedding-space math instead of a quiz-app multiple-choice scoring trick. Every choice you make is a CLIP vector. Your final result is a weighted mean of those vectors, scored by cosine similarity against 16 aesthetic centroids built from a 102-image dataset. The same pipeline runs in reverse for the "classify an image" feature: upload any photo, get its CLIP embedding, see which aesthetic it's closest to.

The interesting parts of this project were never the quiz UI. They were the infrastructure problems underneath it — a CLIP model is heavy, free-tier hosting is light, and making those two facts coexist is most of what this README is actually about.

---

## Architecture

Kansei runs as two independent services, deliberately split apart rather than as one monolith:

```
┌─────────────────────────┐         ┌──────────────────────────┐
│   Main App (FastAPI)    │  HTTP   │  Classify Service         │
│   GCP e2-micro VM       │ ──────► │  (CLIP inference)         │
│   :8000, behind Caddy   │         │  same VM, :8001            │
│                          │         │                            │
│   - quiz / explore /    │         │  - loads CLIP ViT-B/32     │
│     result pages        │         │  - gated by shared-secret  │
│   - UMAP projection      │         │    auth header             │
│     (cached on disk)     │         │  - swap-backed under load  │
│   - SQLite analytics     │         │                            │
└─────────────────────────┘         └──────────────────────────┘
        │                                      │
        └──────────── /data volume ────────────┘
         (analytics.db, umap_reducer.joblib)
```

Both containers run on a single **GCP e2-micro VM** (free tier: 1 vCPU burstable, 964MB RAM, 20GB disk). Caddy sits in front of the main app, terminating HTTPS and proxying to `localhost:8000`. The classify service is exposed directly on `:8001`, gated by a shared-secret header rather than left open.

This used to be two different platforms — the main app on Railway, classify on this same GCP VM — calling each other over the public internet. It's now fully consolidated onto one box. The "why" for that move, and everything that broke along the way, is below.

---

## Why it moved off Railway

Railway's free trial ran out. The obvious fix was to pay for Hobby tier and move on. The choice made instead was to migrate onto a free-tier GCP VM and stay there deliberately — not because I was looking to save up a few bucks, but because **the constraint was the point**. Anyone can rent more RAM. Making a CLIP model, a UMAP fit, and a web app all coexist on a 964MB box that GCP itself doesn't recommend for production is a different kind of problem, and a more honest test of whether the engineering actually holds up. Every fix below exists because of that choice, not despite it.

The migration surfaced real problems that a clean "lift and shift" never does:

### `torch` and `open-clip-torch` were dead weight in the main app

The main app's `requirements.txt` still listed `torch` and `open-clip-torch` from before the classify feature was split into its own service. Nothing in the live request path imported them — a `grep` across the codebase confirmed every usage lived in `classify_worker.py` (the old subprocess approach, now dead code) and a one-off profiling script. Removing both dropped the main app's Docker image from a multi-gigabyte build down to **1.01GB**, with no functional change.

### Disk ran out mid-build

The VM's original 10GB disk had **2.3GB free** after the classify image was already on it — tight enough that adding a second image risked running out mid-build. Rather than fight for space, the disk was resized live to 20GB (`growpart` + `resize2fs`, no downtime, still within GCP's always-free persistent-disk allowance).

### HTTPS wasn't optional — it was load-bearing

The quiz initially failed silently in production with `crypto.randomUUID is not a function`. The cause: that browser API is restricted to secure contexts (HTTPS or localhost), and the app was being served over plain HTTP at `http://<ip>:8000`. This wasn't a nice-to-have — without HTTPS, the session-tracking code that the whole quiz flow depends on simply doesn't run in modern browsers.

Fixed with a **free domain (DuckDNS) + Caddy + Let's Encrypt**, fully automated, zero ongoing cost. Caddy also collapsed the port number out of the URL — `kansei.duckdns.org`, not `kansei.duckdns.org:8000`.

### The classify auth token never made the jump

Railway had `CLASSIFY_AUTH_TOKEN` set in its dashboard. The VM's `docker run` command for the new container didn't pass it, so the running container had an empty token and every classify request was rejected with a clean, correctly-functioning 401 — the auth gate working exactly as designed, just against the wrong service. Fixed by passing the real token explicitly via `-e CLASSIFY_AUTH_TOKEN=...` on container start.

### A real, previously-latent bug in the UMAP projection code

`get_umap_projection()` enforced a fixed minimum-choices threshold (3) with no way to override it — fine for the quiz, but the classify endpoint calls the same function with a single embedding, which used to silently route through a different, already-overridable function (`score()`) but hit the *un*-overridable one for UMAP specifically. The result: classify would return a valid score, then crash the result page rendering `null.x` in the UMAP visualization. Fixed by adding the same `min_required` override to `get_umap_projection()` that `score()` already had, and passing `min_required=1` from the classify endpoint.

This bug existed before tonight — it just never had a code path that exercised it until classify actually started working end-to-end again.

### Refitting UMAP on every restart was real, avoidable cost

Every container restart re-fit UMAP from scratch against the same 102-image dataset — wasted, deterministic work. On this VM's CPU budget, that refit was severe enough to make a restart look hung rather than just slow. Fixed by fitting once locally (on a machine with real headroom), caching the result to disk (`joblib.dump`), and loading the cached fit on every subsequent boot. The cache file lives on the same persistent volume as the analytics database, so it survives restarts and rebuilds.

### The e2-micro's CPU ceiling is real, and it's not a bug

Classify timing varied wildly in testing — 50 seconds one run, two minutes the next, with memory usage staying flat the whole time. The cause isn't memory: **GCP's e2-micro is explicitly documented to sustain only 25% CPU time total**, bursting above that only when the shared host has spare capacity. CLIP inference is CPU-bound matrix math, and that 25% ceiling is the actual constraint — not something `docker run` flags or memory tuning can fix.

Rather than chase a problem with the wrong tool, the timeout was raised to a value that reflects observed reality (90s → 150s), and the loading state in the UI was updated to honestly say so ("can take up to 2 min") instead of pretending it's instant.

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| Backend | FastAPI | Async-friendly, typed, fast to iterate on |
| Embeddings | CLIP ViT-B/32 (open_clip) | Strong general-purpose visual embeddings, no fine-tuning needed |
| Dimensionality reduction | UMAP | Preserves local structure better than PCA/t-SNE for this dataset size |
| Storage | SQLite | Zero-ops, file-based, fine at this scale; fitted UMAP reducer cached the same way |
| Frontend | Vanilla HTML/CSS/JS | No build step, no framework overhead for 4 pages |
| Infrastructure | GCP e2-micro (free tier), Docker, Caddy | Two containers, one box, free HTTPS via Let's Encrypt |
| Domain | DuckDNS (free) | No paid domain needed for Let's Encrypt to issue a real cert |

---

## API

| Endpoint | Method | What it does |
|---|---|---|
| `/api/pairs` | GET | Generate N random aesthetic image pairs for the quiz |
| `/api/score` | POST | Score a set of choice vectors against all 16 centroids |
| `/api/umap` | POST | 3D UMAP projection of the user's result alongside the full dataset |
| `/api/nearest` | POST | k-nearest individual images to the user's result vector |
| `/api/classify` | POST | Upload an image, get it classified via the CLIP inference service |
| `/api/aesthetics` | GET | List all 16 aesthetics with descriptions and colors |
| `/api/event` | POST | Fire-and-forget analytics event logging |
| `/api/event/summary` | GET | Funnel breakdown + share-rate metric |

The classify service (separate container, `:8001`) exposes its own minimal surface: `/health` (open, used for liveness checks) and `/classify` (gated by `X-Kansei-Auth` header).

---

## Local setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # or `source venv/bin/activate` on Linux/Mac
pip install -r requirements.txt
uvicorn main:app --reload
```

The classify feature requires `CLASSIFY_AUTH_TOKEN` to be set to match whatever the classify service expects — without it, `/api/classify` will return a 401 (the auth gate working correctly, just with no credentials supplied). See `.env.example` if present, or set it directly:

```bash
$env:CLASSIFY_AUTH_TOKEN = "your-token-here"   # PowerShell
```

---

## Project context

This started as a way to combine an interest in aesthetics and visual identity with a working CLIP + UMAP pipeline — not as a resume line item first. It became one anyway, because the infrastructure problems that showed up were real engineering, not portfolio theater.

The free-tier constraint specifically was a deliberate choice, not a budget default. Staying on a 964MB VM instead of upgrading meant actually confronting an OOM kill, a CPU-bursting ceiling, and a memory-pressure freeze, rather than paying to make those problems disappear before they taught anything. That's the difference between a project that demonstrates "I can deploy something" and one that demonstrates "I can find out why something breaks and fix the actual cause" — and the second one was always the goal here.

Every tradeoff documented above — swap over a bigger VM, a longer timeout over a rewrite, a free domain over a paid one — was made under that same constraint, on purpose. That's the part worth reading closely if you're evaluating this as more than a quiz app.