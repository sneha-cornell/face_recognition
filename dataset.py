import os
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE = 112


def decode_and_resize(image_path, label):
    raw = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    # Normalize to [-1, 1] — standard for ArcFace / InsightFace pipelines
    image = (tf.cast(image, tf.float32) - 127.5) / 128.0
    return image, label


def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    # Mild brightness/contrast jitter — avoid distorting face appearance
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    image = tf.clip_by_value(image, -1.0, 1.0)
    return image, label


def build_dataset(root_dir, batch_size=512, training=True):
    """
    Build a tf.data pipeline for CASIA-WebFace (or any flat identity-folder layout).

    Expected directory structure:
        root_dir/
            <identity_0>/
                img001.jpg
                img002.jpg
            <identity_1>/
                ...

    Args:
        root_dir:   Path to dataset root.
        batch_size: Batch size.
        training:   If True, shuffle + augment.

    Returns:
        dataset:     tf.data.Dataset yielding (image, label) batches.
        num_classes: Total number of identities found.
    """
    class_names = sorted(
        name for name in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, name))
    )
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    num_classes = len(class_names)

    image_paths, labels = [], []
    valid_exts = {".jpg", ".jpeg", ".png"}
    for class_name in class_names:
        class_dir = os.path.join(root_dir, class_name)
        for fname in os.listdir(class_dir):
            if os.path.splitext(fname)[1].lower() in valid_exts:
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(class_to_idx[class_name])

    print(f"Found {len(image_paths)} images across {num_classes} identities.")

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if training:
        dataset = dataset.shuffle(buffer_size=50_000, reshuffle_each_iteration=True)

    dataset = dataset.map(decode_and_resize, num_parallel_calls=AUTOTUNE)

    if training:
        dataset = dataset.map(augment, num_parallel_calls=AUTOTUNE)

    dataset = (
        dataset
        .batch(batch_size, drop_remainder=training)
        .prefetch(AUTOTUNE)
    )

    return dataset, num_classes
