"""
classify_worker.py

Standalone script, invoked as a subprocess by core.py's classify_image().
Loads CLIP, embeds one image, prints the embedding as JSON, exits.

Why a separate process instead of importing this logic into core.py directly:
the whole point is that when this process exits, the OS reclaims every byte
it used — the model weights, PyTorch's internal buffers, everything. That's
not true if CLIP is loaded inside the main FastAPI process, where it's either
cached forever (current behavior) or freed in a way Python's own garbage
collector doesn't fully guarantee back to the OS (the actual root cause of
the OOMs we hit).

Usage:
    python classify_worker.py <path_to_image>

Output (stdout): a single line of JSON: {"embedding": [...512 floats...]}
Anything else (model download progress, warnings) goes to stderr, so stdout
stays clean and parseable.
"""

import sys
import json


def main():
    if len(sys.argv) != 2:
        print(json.dumps({"error": "usage: classify_worker.py <image_path>"}), file=sys.stdout)
        sys.exit(1)

    image_path = sys.argv[1]

    # Imports happen inside main(), not at module level — this keeps the
    # script's own startup cheap if it's ever imported elsewhere, and makes
    # the actual heavy load clearly scoped to this function.
    import open_clip
    import torch
    from PIL import Image

    device = "cpu"  # subprocess workers don't need CUDA logic; this app runs CPU-only in production

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    # fp16 instead of the default fp32 — roughly halves the model's resident
    # memory (ViT-B/32 weights go from ~350MB to ~175MB in this dtype alone).
    # This is the actual lever being tested: if it's enough to clear
    # Railway's container memory ceiling, no separate service is needed.
    # Trade-off: fp16 on CPU is not guaranteed to be faster, and on some CPU
    # builds of PyTorch certain fp16 ops fall back to fp32 internally anyway
    # — the memory savings from holding fp16 weights still apply regardless.
    model = model.to(device).half()
    model.eval()

    image = Image.open(image_path).convert("RGB")
    # Input tensor must match the model's dtype, or encode_image raises a
    # dtype-mismatch error (the most common way this kind of change breaks).
    tensor = preprocess(image).unsqueeze(0).to(device).half()

    with torch.no_grad():
        embedding = model.encode_image(tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    # Cast back to float32 before serializing — JSON doesn't have a float16
    # type, and downstream code in core.py (cosine_similarity, score, etc.)
    # expects standard Python floats / numpy float64 anyway.
    vec = embedding.float().cpu().numpy()[0].tolist()

    # Only this one line goes to stdout — the parent process reads exactly this.
    print(json.dumps({"embedding": vec}))


if __name__ == "__main__":
    main()