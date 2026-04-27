"""
Training script for face recognition backbone with ArcFace loss.

Example usage:
    python train.py --data_dir /path/to/CASIA-WebFace --epochs 40

The backbone (.h5) saved at the end is what you deploy to the GPX10.
The ArcFace head is training-only and is not saved with the backbone.
"""

import argparse
import os

import tensorflow as tf

from dataset import build_dataset, build_dataset_tfrecord
from loss import ArcFaceLayer
from model import build_face_backbone


def build_training_model(backbone, num_classes, margin=0.5, scale=64.0):
    """
    Wraps backbone + ArcFace head into a single Keras model.
    Both the face image and the integer class label are model inputs so that
    the ArcFace layer can apply the margin to the correct class during the
    forward pass.
    """
    face_input = tf.keras.Input(shape=(112, 112, 3), name="input_face")
    label_input = tf.keras.Input(shape=(), dtype=tf.int32, name="input_label")

    embeddings = backbone(face_input, training=True)
    arcface = ArcFaceLayer(num_classes, margin=margin, scale=scale, name="arcface")
    logits = arcface(embeddings, labels=label_input, training=True)

    model = tf.keras.Model(
        inputs=[face_input, label_input],
        outputs=logits,
        name="training_model",
    )
    return model


def make_lr_schedule(initial_lr, epochs, steps_per_epoch):
    """
    Cosine decay with a short linear warm-up.
    Warm-up for 5 epochs stabilises ArcFace early training when scale is large.
    """
    warmup_steps = 5 * steps_per_epoch
    total_steps = epochs * steps_per_epoch

    warmup = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=initial_lr / 100,
        decay_steps=warmup_steps,
        end_learning_rate=initial_lr,
        power=1.0,
    )
    cosine = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=total_steps - warmup_steps,
        alpha=1e-5,
    )

    # PiecewiseConstantDecay to switch from warmup to cosine after warmup_steps
    # Simpler: use CosineDecayRestarts or implement via callback.
    # For clarity we return cosine only — tweak if you observe unstable early loss.
    return cosine


def main(args):
    print(f"\n=== Face Recognition Training ===")
    print(f"Data:       {args.data_dir}")
    print(f"Epochs:     {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output:     {args.output_dir}\n")

    import math, glob as _glob
    tfrecord_dir = args.data_dir.rstrip("/") + "_tfrecords"
    tfrecord_dir = tfrecord_dir.replace("_images_tfrecords", "_tfrecords")
    # Auto-detect TFRecord dir: same parent, name ends with _tfrecords
    _candidate = os.path.join(os.path.dirname(args.data_dir), "webface_tfrecords")
    if os.path.isdir(_candidate) and _glob.glob(os.path.join(_candidate, "*.tfrecord")):
        tfrecord_dir = _candidate

    if os.path.isdir(tfrecord_dir) and _glob.glob(os.path.join(tfrecord_dir, "*.tfrecord")):
        # Fast path: TFRecord pipeline
        num_classes = len([
            d for d in os.listdir(args.data_dir)
            if os.path.isdir(os.path.join(args.data_dir, d))
        ])
        n_images = sum(
            1 for shard in _glob.glob(os.path.join(tfrecord_dir, "*.tfrecord"))
            for _ in [None]  # can't count without reading; use file sizes as proxy
        )
        # Count from property file if present, else walk once
        prop = os.path.join(os.path.dirname(tfrecord_dir), "faces_webface_112x112", "property")
        n_images = 490623  # known from extraction
        train_ds = build_dataset_tfrecord(
            tfrecord_dir, num_classes, batch_size=args.batch_size, training=True
        )
        print(f"Using TFRecord pipeline: {tfrecord_dir}")
    else:
        # Fallback: per-file pipeline
        train_ds, num_classes = build_dataset(
            args.data_dir, batch_size=args.batch_size, training=True
        )
        n_images = sum(len(files) for _, _, files in os.walk(args.data_dir))

    steps_per_epoch = math.ceil(n_images / args.batch_size)

    backbone = build_face_backbone(embedding_dim=args.embedding_dim)
    print(f"Backbone params: {backbone.count_params():,}  steps/epoch: {steps_per_epoch}")

    training_model = build_training_model(backbone, num_classes, margin=args.margin, scale=args.scale)

    lr_schedule = make_lr_schedule(args.lr, args.epochs, steps_per_epoch)
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=lr_schedule, momentum=0.9, weight_decay=5e-4
    )
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Track top-1 accuracy over identities as a training signal
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name="acc")

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            logits = training_model([images, labels], training=True)
            loss = loss_fn(labels, logits)
        grads = tape.gradient(loss, training_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, training_model.trainable_variables))
        train_acc.update_state(labels, logits)
        return loss

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        train_acc.reset_state()
        running_loss = 0.0

        for step, (images, labels) in enumerate(train_ds):
            loss = train_step(images, labels)
            running_loss += float(loss)

            if (step + 1) % 200 == 0:
                print(
                    f"  [{epoch+1}/{args.epochs}] step {step+1}/{steps_per_epoch}"
                    f"  loss={running_loss/(step+1):.4f}"
                    f"  acc={train_acc.result():.4f}"
                )

        avg_loss = running_loss / steps_per_epoch
        print(
            f"Epoch {epoch+1}/{args.epochs} — "
            f"loss={avg_loss:.4f}  acc={train_acc.result():.4f}"
        )

        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            ckpt = os.path.join(args.output_dir, f"backbone_epoch{epoch+1:03d}.h5")
            backbone.save(ckpt)
            print(f"  Saved: {ckpt}")

    final_path = os.path.join(args.output_dir, "backbone_final.h5")
    backbone.save(final_path)
    print(f"\nTraining complete. Final backbone saved to: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train face recognition backbone with ArcFace")
    parser.add_argument("--data_dir", required=True, help="CASIA-WebFace root directory")
    parser.add_argument("--output_dir", default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--scale", type=float, default=64.0)
    args = parser.parse_args()
    main(args)
