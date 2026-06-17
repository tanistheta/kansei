import json
import numpy as np
import random
import umap as umap_lib
from pathlib import Path
from functools import lru_cache

EMBEDDINGS_PATH = Path(__file__).parent / "kansei_embeddings.json"

AESTHETIC_DESCRIPTIONS = {
    "afrofuturism": "Bold, cosmic, rooted in African tradition and speculative futures.",
    "boho": "Free-spirited, layered, earthy with a wanderer's soul.",
    "brutalism": "Raw, honest, unapologetic — concrete and shadow.",
    "coastal cool": "Effortless, sun-bleached, salt-air calm.",
    "cottagecore": "Soft, pastoral, handmade warmth in a quiet world.",
    "cyber minimalism": "Cold precision, stripped to system — function as form.",
    "cyberpunk": "Neon and rain, high tech low life, beautiful decay.",
    "dark academia": "Candlelit libraries, antiquity, the romance of learning.",
    "ethereal": "Soft light, translucent layers, existing between worlds.",
    "glam maximalism": "More is more — opulent, theatrical, unapologetically excessive.",
    "quiet luxury": "Understated wealth, timeless quality, nothing to prove.",
    "solarpunk": "Hopeful futures, community gardens, sun through green leaves.",
    "teracotta modernism": "Warm clay, curved walls, desert light and organic form.",
    "vintage americana": "Open highways, neon diners, nostalgia for a simpler time.",
    "wabi sabi": "Beauty in imperfection, transience, the worn and the weathered.",
    "zen modern": "Stillness as design — wood, light, intentional emptiness.",
}

AESTHETIC_COLORS = {
    "afrofuturism": "#9B59B6",
    "boho": "#E67E22",
    "brutalism": "#7F8C8D",
    "coastal cool": "#3498DB",
    "cottagecore": "#A8D8A8",
    "cyber minimalism": "#00FFFF",
    "cyberpunk": "#FF00FF",
    "dark academia": "#8B6914",
    "ethereal": "#D7BDE2",
    "glam maximalism": "#F1C40F",
    "quiet luxury": "#BDC3C7",
    "solarpunk": "#2ECC71",
    "teracotta modernism": "#E07B54",
    "vintage americana": "#E74C3C",
    "wabi sabi": "#C9AE8C",
    "zen modern": "#95A5A6",
}


@lru_cache(maxsize=1)
def load_data():
    with open(EMBEDDINGS_PATH) as f:
        raw = json.load(f)

    centroids = {}
    image_pool = {}
    all_vectors = []
    all_labels = []

    for aesthetic, info in raw.items():
        centroids[aesthetic] = np.array(info["centroid"])
        image_pool[aesthetic] = [Path(p).name for p in info["images"]]
        for v in info["vectors"]:
            all_vectors.append(v)
            all_labels.append(aesthetic)

    return centroids, image_pool, np.array(all_vectors), all_labels


@lru_cache(maxsize=1)
def get_reducer():
    _, _, all_vectors, _ = load_data()
    reducer = umap_lib.UMAP(n_components=3, random_state=42, n_neighbors=5, min_dist=0.5)
    reducer.fit(all_vectors)
    return reducer


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def score(user_vectors: list[list[float]]) -> dict:
    centroids, _, _, _ = load_data()
    user_vec = np.mean(user_vectors, axis=0)

    raw = {a: cosine_similarity(user_vec, c) for a, c in centroids.items()}

    # Shift to 0-based (cosine similarity is typically 0.7–1.0 for similar CLIP embeddings)
    # Map actual range to 0–100 so numbers feel meaningful but top isn't always 100
    min_s = min(raw.values())
    max_s = max(raw.values())
    spread = max_s - min_s if max_s - min_s > 0 else 1

    # Scale so top aesthetic is ~85-95%, not always 100%
    # Use: score = min_possible + (raw - min_raw) / spread * range_width
    range_width = 60  # top gets ~85%, bottom gets ~25%
    base = 25
    normalized = {
        a: round(base + (s - min_s) / spread * range_width, 1)
        for a, s in raw.items()
    }

    return dict(sorted(normalized.items(), key=lambda x: x[1], reverse=True))


def get_umap_projection(user_vectors: list[list[float]]) -> dict:
    centroids, _, all_vectors, all_labels = load_data()
    reducer = get_reducer()
    user_vec = np.mean(user_vectors, axis=0)

    centroid_list = list(centroids.values())
    centroid_labels = list(centroids.keys())
    combined = np.vstack([all_vectors, centroid_list, user_vec.reshape(1, -1)])
    projected = reducer.transform(combined)

    n_imgs = len(all_vectors)
    n_centroids = len(centroid_list)

    img_proj = projected[:n_imgs]
    centroid_proj = projected[n_imgs:n_imgs + n_centroids]
    user_proj = projected[-1]

    points = []
    for i, label in enumerate(all_labels):
        points.append({
            "x": float(img_proj[i, 0]),
            "y": float(img_proj[i, 1]),
            "z": float(img_proj[i, 2]),
            "aesthetic": label,
            "color": AESTHETIC_COLORS.get(label, "#888"),
        })

    centers = []
    for i, label in enumerate(centroid_labels):
        centers.append({
            "x": float(centroid_proj[i, 0]),
            "y": float(centroid_proj[i, 1]),
            "z": float(centroid_proj[i, 2]),
            "aesthetic": label,
            "color": AESTHETIC_COLORS.get(label, "#888"),
        })

    return {
        "points": points,
        "centroids": centers,
        "user": {
            "x": float(user_proj[0]),
            "y": float(user_proj[1]),
            "z": float(user_proj[2]),
        }
    }


def generate_pairs(n: int = 25) -> list[dict]:
    centroids, image_pool, _, _ = load_data()
    aesthetics = list(image_pool.keys())
    pairs = []
    for _ in range(n):
        a, b = random.sample(aesthetics, 2)
        img_a = random.choice(image_pool[a])
        img_b = random.choice(image_pool[b])
        pairs.append({
            "left":  {"image": img_a, "aesthetic": a},
            "right": {"image": img_b, "aesthetic": b},
        })
    return pairs


def get_centroid(aesthetic: str) -> list[float]:
    centroids, _, _, _ = load_data()
    return centroids[aesthetic].tolist()


def get_moodboard(aesthetic: str, n: int = 4) -> list[str]:
    _, image_pool, _, _ = load_data()
    images = image_pool.get(aesthetic, [])
    return images[:n]


@lru_cache(maxsize=1)
def get_clip_model():
    import clip
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device


def classify_image(image_bytes: bytes) -> dict:
    """Classify raw image bytes against all aesthetic centroids."""
    import torch
    from PIL import Image
    import io

    model, preprocess, device = get_clip_model()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.encode_image(tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    user_vec = embedding.cpu().numpy()[0]
    return score([user_vec.tolist()])