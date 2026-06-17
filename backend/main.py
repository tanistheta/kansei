from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import threading
import io

from core import (
    load_data,
    get_reducer,
    generate_pairs,
    get_centroid,
    get_moodboard,
    score,
    get_umap_projection,
    classify_image,
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

app.mount("/images", StaticFiles(directory=BASE / "images"), name="images")
app.mount("/static", StaticFiles(directory=BASE / "frontend"), name="static")

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
    return FileResponse(BASE / "frontend" / "index.html")

@app.get("/quiz")
def quiz():
    return FileResponse(BASE / "frontend" / "quiz.html")

@app.get("/result")
def result():
    return FileResponse(BASE / "frontend" / "result.html")


class ChoicesPayload(BaseModel):
    vectors: list[list[float]]


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
    return {
        "scores": scores,
        "top": top,
        "description": AESTHETIC_DESCRIPTIONS.get(top, ""),
        "color": AESTHETIC_COLORS.get(top, "#fff"),
        "moodboard": get_moodboard(top),
    }

@app.post("/api/umap")
def api_umap(payload: ChoicesPayload):
    return get_umap_projection(payload.vectors)

@app.post("/api/classify")
async def api_classify(file: UploadFile = File(...)):
    contents = await file.read()
    scores = classify_image(contents)
    top = list(scores.keys())[0]
    from core import get_centroid as gc
    proxy_vector = gc(top)
    umap_data = get_umap_projection([proxy_vector])
    return {
        "scores": scores,
        "top": top,
        "description": AESTHETIC_DESCRIPTIONS.get(top, ""),
        "color": AESTHETIC_COLORS.get(top, "#fff"),
        "moodboard": get_moodboard(top),
        "umap": umap_data,
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