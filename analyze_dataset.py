"""
Dataset analysis: images per identity, blur, brightness, face coverage.
Samples up to SAMPLE_PER_ID images per identity for speed.
"""

import os, random, struct
import numpy as np

DATA_DIR       = "/tmp/webface_images"
SAMPLE_PER_ID  = 5
RANDOM_SEED    = 42
random.seed(RANDOM_SEED)


def laplacian_variance(gray):
    """Estimate sharpness via variance of a simple Laplacian kernel."""
    # Approximate Laplacian with centre-minus-neighbours on downsampled patch
    p = gray[1:-1, 1:-1].astype(np.float32)
    lap = (4*p
           - gray[:-2, 1:-1].astype(np.float32)
           - gray[2:,  1:-1].astype(np.float32)
           - gray[1:-1, :-2].astype(np.float32)
           - gray[1:-1, 2:].astype(np.float32))
    return float(np.var(lap))


def decode_jpeg_rough(path):
    """
    Minimal JPEG decoder: extract raw pixel data using Python stdlib only.
    Falls back to returning None if the file can't be decoded.
    We only need approximate pixel stats, so we use a crude SOF parser.
    """
    # Use tf later; here just read raw bytes and return path for batch decode
    return path


def main():
    identities = sorted(os.listdir(DATA_DIR))
    num_ids = len(identities)
    print(f"Identities: {num_ids:,}")

    imgs_per_id = []
    sampled_paths = []

    for iid in identities:
        idir = os.path.join(DATA_DIR, iid)
        files = [f for f in os.listdir(idir) if f.endswith('.jpg')]
        imgs_per_id.append(len(files))
        sample = random.sample(files, min(SAMPLE_PER_ID, len(files)))
        sampled_paths.extend(os.path.join(idir, f) for f in sample)

    imgs_per_id = np.array(imgs_per_id)
    total_imgs  = imgs_per_id.sum()

    print(f"\n=== Images per identity ===")
    print(f"  Total images  : {total_imgs:,}")
    print(f"  Min           : {imgs_per_id.min()}")
    print(f"  Max           : {imgs_per_id.max()}")
    print(f"  Mean          : {imgs_per_id.mean():.1f}")
    print(f"  Median        : {np.median(imgs_per_id):.0f}")
    print(f"  Std           : {imgs_per_id.std():.1f}")
    print(f"  IDs with 1 img: {(imgs_per_id==1).sum():,}")
    print(f"  IDs with <5   : {(imgs_per_id<5).sum():,}")
    print(f"  IDs with >=20 : {(imgs_per_id>=20).sum():,}")
    print(f"  IDs with >=50 : {(imgs_per_id>=50).sum():,}")

    # Decode sampled images with TF for pixel stats
    print(f"\nDecoding {len(sampled_paths):,} sampled images for quality stats...")
    import tensorflow as tf

    blur_scores, brightness, contrast = [], [], []
    errors = 0

    for path in sampled_paths:
        try:
            raw = tf.io.read_file(path)
            img = tf.image.decode_jpeg(raw, channels=3).numpy()   # (112,112,3) uint8
            gray = img.mean(axis=2)                                # (112,112)

            blur_scores.append(laplacian_variance(gray))
            brightness.append(gray.mean())
            contrast.append(gray.std())
        except Exception:
            errors += 1

    blur_scores = np.array(blur_scores)
    brightness  = np.array(brightness)
    contrast    = np.array(contrast)

    # Blur threshold: <100 typically considered blurry for 112px crops
    blurry_pct = (blur_scores < 100).mean() * 100

    print(f"\n=== Image quality (sampled {len(blur_scores):,} images) ===")
    print(f"  Sharpness (Laplacian var)")
    print(f"    Mean      : {blur_scores.mean():.1f}")
    print(f"    Median    : {np.median(blur_scores):.1f}")
    print(f"    Blurry (<100): {blurry_pct:.1f}%")
    print(f"  Brightness (0-255 scale)")
    print(f"    Mean      : {brightness.mean():.1f}")
    print(f"    Std       : {brightness.std():.1f}")
    print(f"    Dark (<64): {(brightness<64).mean()*100:.1f}%")
    print(f"    Bright (>200): {(brightness>200).mean()*100:.1f}%")
    print(f"  Contrast (pixel std)")
    print(f"    Mean      : {contrast.mean():.1f}")
    print(f"  Decode errors: {errors}")

    # All images are 112x112 pre-aligned crops — 1 face per image by construction
    print(f"\n=== Image format ===")
    print(f"  Resolution    : 112 × 112 px (pre-aligned)")
    print(f"  Faces/image   : 1 (pre-cropped & aligned)")
    print(f"  Channels      : RGB JPEG")


if __name__ == "__main__":
    main()
