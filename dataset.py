import os
import glob
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE = 112


# ── TFRecord pipeline (fast) ─────────────────────────────────────────────────

def _parse_tfrecord(serialized):
    features = tf.io.parse_single_example(serialized, {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    })
    image = tf.image.decode_jpeg(features["image"], channels=3)
    image = tf.cast(image, tf.float32)
    image = (image - 127.5) / 128.0          # [-1, 1]
    label = tf.cast(features["label"], tf.int32)
    return image, label


def _augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    image = tf.clip_by_value(image, -1.0, 1.0)
    return image, label


def build_dataset_tfrecord(tfrecord_dir, num_classes, batch_size=512, training=True):
    """
    Fast pipeline: parallel interleaved reads from sharded TFRecords,
    decode on CPU, augment, batch, prefetch to GPU.

    ~10x faster than per-file JPEG loading for large datasets.
    """
    shards = sorted(glob.glob(os.path.join(tfrecord_dir, "*.tfrecord")))
    assert shards, f"No .tfrecord files found in {tfrecord_dir}"
    print(f"Found {len(shards)} TFRecord shards  ({num_classes} classes)")

    ds = tf.data.Dataset.from_tensor_slices(shards)

    if training:
        ds = ds.shuffle(len(shards))

    # Read shards in parallel — cycle_length controls how many open at once
    ds = ds.interleave(
        lambda path: tf.data.TFRecordDataset(path, buffer_size=64 << 20),
        cycle_length=16,
        block_length=16,
        num_parallel_calls=AUTOTUNE,
        deterministic=False,
    )

    if training:
        ds = ds.shuffle(buffer_size=20_000, reshuffle_each_iteration=True)

    ds = ds.map(_parse_tfrecord, num_parallel_calls=AUTOTUNE)

    if training:
        ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=training).prefetch(AUTOTUNE)
    return ds


# ── Legacy per-file pipeline (kept for compatibility) ────────────────────────

def _decode_and_resize(image_path, label):
    raw   = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = (tf.cast(image, tf.float32) - 127.5) / 128.0
    return image, label


def build_dataset(root_dir, batch_size=512, training=True):
    class_names = sorted(
        name for name in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, name))
    )
    num_classes = len(class_names)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    image_paths, labels = [], []
    for cls in class_names:
        cls_dir = os.path.join(root_dir, cls)
        for fname in os.listdir(cls_dir):
            if os.path.splitext(fname)[1].lower() in {".jpg", ".jpeg", ".png"}:
                image_paths.append(os.path.join(cls_dir, fname))
                labels.append(class_to_idx[cls])

    print(f"Found {len(image_paths)} images across {num_classes} identities.")

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if training:
        ds = ds.shuffle(50_000, reshuffle_each_iteration=True)
    ds = ds.map(_decode_and_resize, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=training).prefetch(AUTOTUNE)
    return ds, num_classes
