from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import threading

from core import (
    load_data,
    get_reducer,
    generate_pairs,
    get_centroid,
    get_moodboard,
    score,
    get_umap_projection,
    classify_image,
    real_consistency,
    rejection_analysis,
    nearest_images,
    AESTHETIC_DESCRIPTIONS,
    AESTHETIC_COLORS,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = Path(__file__).parent.parent
IMAGES_DIR = BASE / "images" if (BASE / "images").exists() else Path(__file__).parent / "images"
FRONTEND_DIR = BASE / "frontend" if (BASE / "frontend").exists() else Path(__file__).parent / "frontend"

app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.on_event("startup")
async def startup():
    def preheat():
        print("Loading embeddings...")
        load_data()
        print("Computing UMAP...")
        get_reducer()
        print("Kansei ready.")
    threading.Thread(target=preheat, daemon=True).start()


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR  / "index.html")

@app.get("/quiz")
def quiz():
    return FileResponse(FRONTEND_DIR / "quiz.html")

@app.get("/explore")
def explore():
    return FileResponse(FRONTEND_DIR / "explore.html")

@app.get("/result")
def result():
    return FileResponse(FRONTEND_DIR  / "result.html")


# ── Models ────────────────────────────────────────────────────────────────────
class ChoicesPayload(BaseModel):
    vectors: list[list[float]]
    choices: list[dict] | None = None  # [{chosen, rejected}, ...]


class NearestPayload(BaseModel):
    vectors: list[list[float]]
    k: int = 5


# ── API ───────────────────────────────────────────────────────────────────────
@app.get("/api/pairs")
def api_pairs(n: int = 25):
    return {"pairs": generate_pairs(n)}


@app.get("/api/centroid/{aesthetic}")
def api_centroid(aesthetic: str):
    return {"vector": get_centroid(aesthetic)}


@app.post("/api/score")
def api_score(payload: ChoicesPayload):
    scores = score(payload.vectors)
    top = list(scores.keys())[0]

    # Real consistency from embedding variance
    consistency = real_consistency(payload.vectors)

    # Rejection analysis if choices provided
    rejections = None
    if payload.choices:
        rejections = rejection_analysis(payload.choices)

    return {
        "scores": scores,
        "top": top,
        "description": AESTHETIC_DESCRIPTIONS.get(top, ""),
        "color": AESTHETIC_COLORS.get(top, "#fff"),
        "moodboard": get_moodboard(top),
        "consistency": consistency,
        "rejections": rejections,
    }


@app.post("/api/nearest")
def api_nearest(payload: NearestPayload):
    results = nearest_images(payload.vectors, k=payload.k)
    return {"nearest": results}


@app.post("/api/umap")
def api_umap(payload: ChoicesPayload):
    return get_umap_projection(payload.vectors)


@app.post("/api/classify")
async def api_classify(file: UploadFile = File(...)):
    contents = await file.read()
    scores = classify_image(contents)
    top = list(scores.keys())[0]
    proxy_vector = get_centroid(top)
    umap_data = get_umap_projection([proxy_vector])
    nearest = nearest_images([proxy_vector], k=5)
    return {
        "scores": scores,
        "top": top,
        "description": AESTHETIC_DESCRIPTIONS.get(top, ""),
        "color": AESTHETIC_COLORS.get(top, "#fff"),
        "moodboard": get_moodboard(top),
        "umap": umap_data,
        "nearest": nearest,
        "consistency": None,
        "rejections": None,
    }


@app.get("/api/aesthetics")
def api_aesthetics():
    return {
        "aesthetics": [
            {
                "name": name,
                "description": AESTHETIC_DESCRIPTIONS.get(name, ""),
                "color": AESTHETIC_COLORS.get(name, "#fff"),
            }
            for name in AESTHETIC_DESCRIPTIONS
        ]
    }


@app.get("/api/images/{aesthetic}")
def api_images(aesthetic: str):
    """Return all image filenames for a given aesthetic."""
    _, image_pool, _, _, _ = load_data()
    images = image_pool.get(aesthetic, [])
    return {"aesthetic": aesthetic, "images": images}