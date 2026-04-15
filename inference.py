"""
Inference utilities for the doorbell face recognition system.

Enrollment:
    python inference.py enroll --model checkpoints/backbone_final.h5 \
        --name "Alice" --images alice1.jpg alice2.jpg alice3.jpg

Verification:
    python inference.py verify --model checkpoints/backbone_final.h5 \
        --image doorbell_capture.jpg
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf

IMG_SIZE = 112
DEFAULT_DB = "face_db.json"
DEFAULT_THRESHOLD = 0.40   # cosine similarity threshold — tune on your enrolled set


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(image_path: str) -> tf.Tensor:
    """Load and normalize a single face image to (1, 112, 112, 3)."""
    raw = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = (tf.cast(image, tf.float32) - 127.5) / 128.0
    return tf.expand_dims(image, axis=0)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def get_embedding(backbone, image_path: str) -> np.ndarray:
    """Return a raw (embedding_dim,) numpy vector for one image."""
    image = preprocess(image_path)
    emb = backbone(image, training=False)
    return emb.numpy()[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two embedding vectors."""
    a = a / (np.linalg.norm(a) + 1e-7)
    b = b / (np.linalg.norm(b) + 1e-7)
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# Enrollment
# ---------------------------------------------------------------------------

def enroll(backbone, name: str, image_paths: list[str], db_path: str = DEFAULT_DB):
    """
    Register a person by averaging embeddings from multiple images.

    Storing the mean embedding is robust to mild pose/lighting variation.
    The more images provided (5-10 recommended), the more stable the average.
    """
    embeddings = [get_embedding(backbone, p) for p in image_paths]
    mean_emb = np.mean(embeddings, axis=0).tolist()

    db = _load_db(db_path)
    db[name] = mean_emb
    _save_db(db, db_path)

    print(f"Enrolled '{name}' using {len(image_paths)} image(s). DB: {db_path}")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(
    backbone,
    image_path: str,
    db_path: str = DEFAULT_DB,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[str | None, float]:
    """
    Compare a query face against all enrolled identities.

    Returns:
        (matched_name, score) if similarity >= threshold, else (None, best_score).
    """
    db = _load_db(db_path)
    if not db:
        print("Database is empty — enroll some faces first.")
        return None, 0.0

    query_emb = get_embedding(backbone, image_path)

    best_name, best_score = None, -1.0
    for name, stored in db.items():
        score = cosine_similarity(query_emb, np.array(stored))
        if score > best_score:
            best_score = score
            best_name = name

    if best_score >= threshold:
        print(f"ACCESS GRANTED — {best_name}  (similarity={best_score:.3f})")
        return best_name, best_score
    else:
        print(
            f"ACCESS DENIED — no match above threshold {threshold}. "
            f"Closest: {best_name} (similarity={best_score:.3f})"
        )
        return None, best_score


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _load_db(db_path: str) -> dict:
    if os.path.exists(db_path):
        with open(db_path) as f:
            return json.load(f)
    return {}


def _save_db(db: dict, db_path: str):
    with open(db_path, "w") as f:
        json.dump(db, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["enroll", "verify"])
    parser.add_argument("--model", required=True, help="Path to backbone .h5 file")
    parser.add_argument("--db", default=DEFAULT_DB, help="Face database JSON path")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)

    # Enroll args
    parser.add_argument("--name", help="Identity name for enrollment")
    parser.add_argument("--images", nargs="+", help="Image paths for enrollment")

    # Verify args
    parser.add_argument("--image", help="Query image path for verification")

    args = parser.parse_args()

    backbone = tf.keras.models.load_model(args.model, compile=False)

    if args.command == "enroll":
        if not args.name or not args.images:
            parser.error("enroll requires --name and --images")
        enroll(backbone, args.name, args.images, db_path=args.db)

    elif args.command == "verify":
        if not args.image:
            parser.error("verify requires --image")
        verify(backbone, args.image, db_path=args.db, threshold=args.threshold)


if __name__ == "__main__":
    main()
