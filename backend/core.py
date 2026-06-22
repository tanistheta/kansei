import json
import httpx
import os
import numpy as np
import random
import umap as umap_lib
from pathlib import Path
from functools import lru_cache

EMBEDDINGS_PATH = Path(__file__).parent / "kansei_embeddings.json"
CLASSIFY_SERVICE_URL = "http://35.206.123.246:8001/classify"
CLASSIFY_AUTH_TOKEN = os.environ.get("CLASSIFY_AUTH_TOKEN", "")

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
    all_images = []  # track image filenames alongside vectors

    for aesthetic, info in raw.items():
        centroids[aesthetic] = np.array(info["centroid"])
        names = [p.replace('\\', '/').split('/')[-1] for p in info["images"]]
        image_pool[aesthetic] = names
        for i, v in enumerate(info["vectors"]):
            all_vectors.append(v)
            all_labels.append(aesthetic)
            all_images.append(names[i] if i < len(names) else "")

    return centroids, image_pool, np.array(all_vectors), all_labels, all_images


@lru_cache(maxsize=1)
def get_reducer():
    _, _, all_vectors, _, _ = load_data()
    reducer = umap_lib.UMAP(n_components=3, random_state=42, n_neighbors=5, min_dist=0.5)
    reducer.fit(all_vectors)
    return reducer


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def score(
    user_vectors: list[list[float]],
    weights: list[float] | None = None
) -> dict:
    centroids, _, _, _, _ = load_data()

    # Weighted mean — later choices carry more weight
    vecs = np.array(user_vectors)
    if weights is None:
        n = len(vecs)
        # Exponential decay: round 25 counts ~3x more than round 1
        weights = np.exp(np.linspace(0, 1.1, n))
    w = np.array(weights)
    w = w / w.sum()
    user_vec = np.average(vecs, axis=0, weights=w)

    raw = {a: cosine_similarity(user_vec, c) for a, c in centroids.items()}
    min_s = min(raw.values())
    max_s = max(raw.values())
    spread = max_s - min_s if max_s - min_s > 0 else 1

    # Aggressive amplification: cube the relative position
    # This creates a steep drop-off — top stays high, rest falls fast
    normalized = {}
    for a, s in raw.items():
        relative = (s - min_s) / spread   # 0.0 to 1.0
        amplified = relative ** 3          # cubic — top ~85%, second ~50%, rest drops sharply
        normalized[a] = round(10 + amplified * 75, 1)

    return dict(sorted(normalized.items(), key=lambda x: x[1], reverse=True))


def real_consistency(user_vectors: list[list[float]]) -> dict:
    """
    Compute true consistency as 1 - mean pairwise cosine distance.
    Low variance = tight cluster = decisive. High variance = scattered = eclectic.
    Returns score 0-100 and interpretation.
    """
    vecs = np.array(user_vectors)
    if len(vecs) < 2:
        return {"score": 100, "label": "Decisive", "desc": "Not enough choices to measure."}

    # Normalize all vectors
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs_norm = vecs / norms

    # Mean pairwise cosine similarity
    sim_matrix = vecs_norm @ vecs_norm.T
    n = len(vecs)
    upper = sim_matrix[np.triu_indices(n, k=1)]
    mean_sim = float(np.mean(upper))

    # Map to 0-100: mean_sim typically 0.7-1.0
    # 1.0 = perfectly consistent (100%), 0.7 = very scattered (0%)
    score_val = round(max(0, min(100, (mean_sim - 0.70) / 0.30 * 100)), 1)

    if score_val >= 80:
        label = "Decisive"
        desc = "Your choices clustered tightly — you know exactly what you want."
    elif score_val >= 55:
        label = "Eclectic"
        desc = "Your taste draws from multiple aesthetics without losing a center of gravity."
    elif score_val >= 35:
        label = "Wandering"
        desc = "You move between contrasting worlds — contradiction is part of your aesthetic."
    else:
        label = "Undefined"
        desc = "Your choices span the full aesthetic spectrum. You resist easy categorization."

    return {"score": score_val, "label": label, "desc": desc}


def rejection_analysis(choices: list[dict]) -> dict:
    """
    Analyze rejected aesthetics from quiz choices.
    Each choice: {"chosen": "aesthetic_name", "rejected": "aesthetic_name"}
    Returns rejection counts, most avoided, and never chosen.
    """
    _, image_pool, _, _, _ = load_data()
    all_aesthetics = set(image_pool.keys())

    rejection_counts = {a: 0 for a in all_aesthetics}
    chosen_counts = {a: 0 for a in all_aesthetics}

    for c in choices:
        chosen = c.get("chosen", "")
        rejected = c.get("rejected", "")
        if chosen in chosen_counts:
            chosen_counts[chosen] += 1
        if rejected in rejection_counts:
            rejection_counts[rejected] += 1

    # Sort by rejection count
    sorted_rejected = sorted(rejection_counts.items(), key=lambda x: x[1], reverse=True)

    # Never chosen aesthetics
    never_chosen = [a for a, cnt in chosen_counts.items() if cnt == 0]

    # Most avoided = high rejections + low chosen
    avoidance_score = {
        a: rejection_counts[a] - chosen_counts[a]
        for a in all_aesthetics
    }
    most_avoided = sorted(avoidance_score.items(), key=lambda x: x[1], reverse=True)[:3]
    most_avoided = [(a, s) for a, s in most_avoided if s > 0]

    return {
        "rejection_counts": dict(sorted_rejected),
        "chosen_counts": chosen_counts,
        "never_chosen": never_chosen,
        "most_avoided": most_avoided,
        "top_rejected": sorted_rejected[:3],
    }


def nearest_images(user_vectors: list[list[float]], k: int = 5) -> list[dict]:
    """
    kNN: find the k most similar individual images to the user's mean vector.
    Returns image filename, aesthetic label, and similarity score.
    """
    _, _, all_vectors, all_labels, all_images = load_data()

    vecs = np.array(user_vectors)
    n = len(vecs)
    weights = np.exp(np.linspace(0, 1.1, n))
    weights = weights / weights.sum()
    user_vec = np.average(vecs, axis=0, weights=weights)

    # Normalize
    user_vec_norm = user_vec / np.linalg.norm(user_vec)
    all_vecs_norm = all_vectors / np.linalg.norm(all_vectors, axis=1, keepdims=True)

    similarities = all_vecs_norm @ user_vec_norm

    top_k_idx = np.argsort(similarities)[::-1][:k]

    results = []
    for idx in top_k_idx:
        results.append({
            "image": all_images[idx],
            "aesthetic": all_labels[idx],
            "similarity": round(float(similarities[idx]), 4),
            "color": AESTHETIC_COLORS.get(all_labels[idx], "#888"),
        })

    return results


def get_umap_projection(user_vectors: list[list[float]]) -> dict:
    centroids, _, all_vectors, all_labels, _ = load_data()
    reducer = get_reducer()

    vecs = np.array(user_vectors)
    n = len(vecs)
    weights = np.exp(np.linspace(0, 1.1, n))
    weights = weights / weights.sum()
    user_vec = np.average(vecs, axis=0, weights=weights)

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
    centroids, image_pool, _, _, _ = load_data()
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
    centroids, _, _, _, _ = load_data()
    return centroids[aesthetic].tolist()


def get_moodboard(aesthetic: str, n: int = 4) -> list[str]:
    _, image_pool, _, _, _ = load_data()
    images = image_pool.get(aesthetic, [])
    return images[:n]


def classify_image(image_bytes: bytes) -> dict:
    """
    Calls the isolated classify service running on a separate cloud VM,
    instead of spawning a local subprocess. The service has its own
    dedicated memory budget (a GCP e2-micro instance with swap configured
    to absorb CLIP's ~918MB peak RSS) — fully decoupled from this app's
    own memory ceiling, which is the constraint that caused the original
    Railway OOM kills.

    Trade-off, stated plainly: this adds network latency and an external
    dependency (if the VM is down, this feature is down) in exchange for
    removing the subprocess-reload cost on every call and the temp-file
    write/cleanup the old implementation needed. Both are acceptable for
    an occasional feature; neither would be acceptable as the app's hot path.
    """
    try:
        response = httpx.post(
            CLASSIFY_SERVICE_URL,
            files={"image": ("image.jpg", image_bytes, "image/jpeg")},
            headers={"X-Kansei-Auth": CLASSIFY_AUTH_TOKEN},
            timeout=90.0,  # generous — e2-micro leans on disk-backed swap under
            # memory pressure, which can make a single request take 30-60s+;
            # 30s was too tight and caused real timeouts under that condition.
        )
        response.raise_for_status()
    except httpx.TimeoutException:
        raise RuntimeError("classify service timed out — it may be cold-starting, try again shortly")
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"classify service returned {e.response.status_code}: {e.response.text}")
    except httpx.RequestError as e:
        raise RuntimeError(f"could not reach classify service: {e}")

    output = response.json()
    if "embedding" not in output:
        raise RuntimeError(f"classify service produced unexpected response: {output!r}")

    user_vec = output["embedding"]
    return score([user_vec])