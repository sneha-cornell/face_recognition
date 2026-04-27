"""
End-to-end train + eval on LFW (min 10 images/person).
Produces real verification metrics: accuracy, TAR@FAR=0.01, AUC.
"""

import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

from model import build_face_backbone
from loss import ArcFaceLayer

# ── Config ──────────────────────────────────────────────────────────────────
EMBEDDING_DIM = 128
BATCH_SIZE    = 64
EPOCHS        = 30
LR            = 0.01
MARGIN        = 0.5
SCALE         = 32.0   # lower scale suits small #classes
SEED          = 42
# ────────────────────────────────────────────────────────────────────────────


def load_lfw():
    print("Loading LFW (min 10 images/person)...")
    lfw = fetch_lfw_people(min_faces_per_person=10, resize=1.0, color=True)
    images = lfw.images.astype(np.float32)          # (N, 125, 94, 3)  [0,255]
    labels = lfw.target.astype(np.int32)
    num_classes = len(np.unique(labels))
    print(f"  {len(images)} images | {num_classes} identities")

    # Resize to 112×112 and normalise to [-1, 1]
    images = tf.image.resize(images, [112, 112]).numpy()
    images = (images - 127.5) / 128.0

    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, stratify=labels, random_state=SEED
    )
    return X_train, X_test, y_train, y_test, num_classes


def build_training_model(backbone, num_classes):
    face_in  = tf.keras.Input(shape=(112, 112, 3), name="input_face")
    label_in = tf.keras.Input(shape=(), dtype=tf.int32, name="input_label")
    emb      = backbone(face_in, training=True)
    logits   = ArcFaceLayer(num_classes, margin=MARGIN, scale=SCALE)(
                   emb, labels=label_in, training=True)
    return tf.keras.Model(inputs=[face_in, label_in], outputs=logits)


def make_pairs(X, y, n_pairs=2000, rng=None):
    """Generate balanced genuine/impostor pairs for verification eval."""
    rng = rng or np.random.default_rng(SEED)
    ids = np.unique(y)
    pairs, pair_labels = [], []

    # Genuine pairs (same identity)
    for _ in range(n_pairs // 2):
        cls = rng.choice(ids)
        idx = np.where(y == cls)[0]
        if len(idx) < 2:
            continue
        i, j = rng.choice(idx, 2, replace=False)
        pairs.append((i, j))
        pair_labels.append(1)

    # Impostor pairs (different identity)
    for _ in range(n_pairs // 2):
        c1, c2 = rng.choice(ids, 2, replace=False)
        i = rng.choice(np.where(y == c1)[0])
        j = rng.choice(np.where(y == c2)[0])
        pairs.append((i, j))
        pair_labels.append(0)

    return pairs, np.array(pair_labels)


def cosine_sim(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-7)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-7)
    return np.sum(a * b, axis=1)


def evaluate(backbone, X_test, y_test):
    print("\n── Evaluation ──────────────────────────────────────────")
    embs = backbone(X_test, training=False).numpy()

    pairs, pair_labels = make_pairs(X_test, y_test, n_pairs=2000)
    a = np.array([embs[i] for i, _ in pairs])
    b = np.array([embs[j] for _, j in pairs])
    scores = cosine_sim(a, b)

    # ROC / AUC
    fpr, tpr, thresholds = roc_curve(pair_labels, scores)
    roc_auc = auc(fpr, tpr)

    # TAR @ FAR ≤ 0.01
    tar_at_far1 = float(tpr[np.searchsorted(fpr, 0.01, side="right") - 1])

    # Best verification accuracy (optimal threshold)
    accs = [np.mean((scores >= t) == pair_labels) for t in thresholds]
    best_acc = float(np.max(accs))
    best_thr = float(thresholds[np.argmax(accs)])

    print(f"  Pairs evaluated : {len(pair_labels)} ({pair_labels.sum()} genuine)")
    print(f"  AUC             : {roc_auc:.4f}")
    print(f"  Best accuracy   : {best_acc:.4f}  (threshold={best_thr:.3f})")
    print(f"  TAR @ FAR=1%    : {tar_at_far1:.4f}")
    print("────────────────────────────────────────────────────────")
    return {"auc": roc_auc, "best_acc": best_acc, "tar_at_far1": tar_at_far1,
            "best_threshold": best_thr}


def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    X_train, X_test, y_train, y_test, num_classes = load_lfw()
    print(f"Train: {len(X_train)}  Test: {len(X_test)}  Classes: {num_classes}")

    backbone       = build_face_backbone(embedding_dim=EMBEDDING_DIM)
    training_model = build_training_model(backbone, num_classes)

    loss_fn    = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer  = tf.keras.optimizers.SGD(learning_rate=LR, momentum=0.9, weight_decay=5e-4)
    train_acc  = tf.keras.metrics.SparseCategoricalAccuracy()

    steps = int(np.ceil(len(X_train) / BATCH_SIZE))

    print(f"\nTraining {EPOCHS} epochs | {steps} steps/epoch | {num_classes} classes\n")

    for epoch in range(EPOCHS):
        idx = np.random.permutation(len(X_train))
        X_s, y_s = X_train[idx], y_train[idx]
        train_acc.reset_state()
        total_loss = 0.0

        for step in range(steps):
            xb = X_s[step*BATCH_SIZE:(step+1)*BATCH_SIZE]
            yb = y_s[step*BATCH_SIZE:(step+1)*BATCH_SIZE]
            with tf.GradientTape() as tape:
                logits = training_model([xb, yb], training=True)
                loss   = loss_fn(yb, logits)
            grads = tape.gradient(loss, training_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, training_model.trainable_variables))
            train_acc.update_state(yb, logits)
            total_loss += float(loss)

        avg_loss = total_loss / steps
        print(f"Epoch {epoch+1:02d}/{EPOCHS}  loss={avg_loss:.4f}  acc={train_acc.result():.4f}")

    backbone.save("/tmp/face_recognition/backbone_lfw.h5")
    print("\nSaved: backbone_lfw.h5")

    metrics = evaluate(backbone, X_test, y_test)
    return metrics


if __name__ == "__main__":
    main()
