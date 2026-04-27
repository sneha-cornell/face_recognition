"""
Quick subset training to validate the pipeline before full 40-epoch run.
Uses the top-N identities by image count (most data = best ArcFace signal).
Evaluates on LFW/CFP-FP/AgeDB-30 .bin files after training.

Usage:
    python train_subset.py --n_ids 1000 --epochs 5
"""

import argparse, math, os
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, auc

from model import build_face_backbone
from loss import ArcFaceLayer

AUTOTUNE   = tf.data.AUTOTUNE
IMG_SIZE   = 112
BIN_DIR    = "/tmp/webface/faces_webface_112x112"
IMG_DIR    = "/tmp/webface_images"


# ── Dataset ──────────────────────────────────────────────────────────────────

def build_subset(img_dir, n_ids, batch_size):
    """Pick the n_ids identities with most images, build tf.data pipeline."""
    classes = sorted(os.listdir(img_dir))
    counts  = [(len(os.listdir(os.path.join(img_dir, c))), c) for c in classes]
    counts.sort(reverse=True)
    top_ids  = [c for _, c in counts[:n_ids]]
    id_to_idx = {c: i for i, c in enumerate(top_ids)}

    paths, labels = [], []
    for cls in top_ids:
        d = os.path.join(img_dir, cls)
        for f in os.listdir(d):
            if f.endswith(".jpg"):
                paths.append(os.path.join(d, f))
                labels.append(id_to_idx[cls])

    n = len(paths)
    print(f"Subset: {n_ids} identities | {n:,} images")

    def decode(p, lbl):
        img = tf.image.decode_jpeg(tf.io.read_file(p), channels=3)
        img = tf.cast(img, tf.float32)
        img = (img - 127.5) / 128.0
        img = tf.image.random_flip_left_right(img)
        return img, lbl

    ds = (tf.data.Dataset.from_tensor_slices((paths, labels))
          .shuffle(min(n, 20_000), reshuffle_each_iteration=True)
          .map(decode, num_parallel_calls=AUTOTUNE)
          .batch(batch_size, drop_remainder=True)
          .prefetch(AUTOTUNE))

    steps = math.ceil(n / batch_size)
    return ds, n_ids, steps


# ── Eval: read InsightFace .bin pairs ────────────────────────────────────────

def load_bin(bin_path, image_size=112):
    """Load InsightFace .bin eval file (pickle of (jpeg_list, issame_list))."""
    import pickle
    with open(bin_path, "rb") as f:
        jpeg_list, issame_list = pickle.load(f, encoding="bytes")

    images = []
    for jpeg in jpeg_list:
        try:
            img = tf.image.decode_jpeg(jpeg, channels=3)
            img = tf.cast(img, tf.float32)
            img = (img - 127.5) / 128.0
            images.append(img.numpy())
        except Exception:
            images.append(np.zeros((image_size, image_size, 3), np.float32))

    return np.array(images, dtype=np.float32), np.array(issame_list, dtype=bool)


def cosine_sim(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-7)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-7)
    return np.sum(a * b, axis=1)


def evaluate_bin(backbone, bin_path, name, batch_size=512):
    print(f"\n  [{name}] loading pairs...")
    images, issame = load_bin(bin_path)
    n_pairs = len(issame)

    # Embed all images in batches
    embs = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        embs.append(backbone(batch, training=False).numpy())
    embs = np.concatenate(embs, axis=0)

    a = embs[0::2]   # first image of each pair
    b = embs[1::2]   # second image of each pair
    scores = cosine_sim(a, b)

    fpr, tpr, thresholds = roc_curve(issame.astype(int), scores)
    roc_auc = auc(fpr, tpr)

    # Best accuracy
    accs = [np.mean((scores >= t) == issame) for t in thresholds]
    best_acc = float(np.max(accs))

    # TAR @ FAR thresholds
    tar_1pct  = float(tpr[np.searchsorted(fpr, 0.01,  side="right") - 1])
    tar_01pct = float(tpr[np.searchsorted(fpr, 0.001, side="right") - 1])

    print(f"  [{name}] pairs={n_pairs}  acc={best_acc:.4f}  AUC={roc_auc:.4f}  TAR@FAR1%={tar_1pct:.4f}  TAR@FAR0.1%={tar_01pct:.4f}")
    return {"name": name, "acc": best_acc, "auc": roc_auc, "tar_far1": tar_1pct, "tar_far01": tar_01pct}


# ── Training ─────────────────────────────────────────────────────────────────

def main(args):
    print(f"\n=== Subset Training: {args.n_ids} identities | {args.epochs} epochs ===")

    train_ds, num_classes, steps = build_subset(IMG_DIR, args.n_ids, args.batch_size)

    backbone = build_face_backbone(embedding_dim=128)

    face_in  = tf.keras.Input(shape=(112, 112, 3), name="input_face")
    label_in = tf.keras.Input(shape=(), dtype=tf.int32, name="input_label")
    emb      = backbone(face_in, training=True)
    logits   = ArcFaceLayer(num_classes, margin=0.3, scale=32.0)(emb, labels=label_in, training=True)
    train_model = tf.keras.Model(inputs=[face_in, label_in], outputs=logits)

    lr = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.01,
        decay_steps=args.epochs * steps,
        alpha=1e-5,
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, weight_decay=5e-4)
    loss_fn   = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    @tf.function
    def train_step(imgs, lbls):
        with tf.GradientTape() as tape:
            logits = train_model([imgs, lbls], training=True)
            loss   = loss_fn(lbls, logits)
        grads = tape.gradient(loss, train_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, train_model.trainable_variables))
        acc_metric.update_state(lbls, logits)
        return loss

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        acc_metric.reset_state()
        total_loss = 0.0
        for step, (imgs, lbls) in enumerate(train_ds):
            loss = train_step(imgs, lbls)
            total_loss += float(loss)
            if (step + 1) % 100 == 0:
                print(f"  [{epoch+1}/{args.epochs}] step {step+1}/{steps}  loss={total_loss/(step+1):.4f}  acc={acc_metric.result():.4f}")
        print(f"Epoch {epoch+1}/{args.epochs} — loss={total_loss/steps:.4f}  acc={acc_metric.result():.4f}")

    ckpt = os.path.join(args.output_dir, f"backbone_subset{args.n_ids}_ep{args.epochs}.keras")
    backbone.save(ckpt)
    print(f"\nSaved: {ckpt}")

    # ── Evaluate on standard benchmarks ──
    print("\n=== Evaluation ===")
    results = []
    for name, fname in [("LFW", "lfw.bin"), ("CFP-FP", "cfp_fp.bin"), ("AgeDB-30", "agedb_30.bin")]:
        bin_path = os.path.join(BIN_DIR, fname)
        if os.path.exists(bin_path):
            results.append(evaluate_bin(backbone, bin_path, name))
        else:
            print(f"  [{name}] not found at {bin_path}")

    print("\n=== Summary ===")
    for r in results:
        print(f"  {r['name']:12s}  acc={r['acc']:.4f}  AUC={r['auc']:.4f}  TAR@FAR1%={r['tar_far1']:.4f}  TAR@FAR0.1%={r['tar_far01']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_ids",      type=int, default=1000)
    parser.add_argument("--epochs",     type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output_dir", default="/tmp/face_recognition/checkpoints")
    args = parser.parse_args()
    main(args)
