"""
Convert CASIA-WebFace image folders to sharded TFRecord files.
Stores raw JPEG bytes — no re-encoding, minimal overhead.

Output: /tmp/webface_tfrecords/train-NNNNN-of-00128.tfrecord
128 shards allows parallel interleaved reading.
"""

import os, math, random
import tensorflow as tf

IMG_DIR    = "/tmp/webface_images"
OUT_DIR    = "/tmp/webface_tfrecords"
NUM_SHARDS = 128
SEED       = 42

random.seed(SEED)


def make_example(jpeg_bytes: bytes, label: int) -> bytes:
    feature = {
        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[jpeg_bytes])),
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


def main():
    # Collect all (path, label) pairs
    print("Scanning image directory...")
    class_names = sorted(os.listdir(IMG_DIR))
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    samples = []
    for cls in class_names:
        cls_dir = os.path.join(IMG_DIR, cls)
        for fname in os.listdir(cls_dir):
            if fname.endswith(".jpg"):
                samples.append((os.path.join(cls_dir, fname), class_to_idx[cls]))

    random.shuffle(samples)
    n = len(samples)
    print(f"  {n:,} images | {len(class_names):,} classes -> {NUM_SHARDS} shards")

    os.makedirs(OUT_DIR, exist_ok=True)
    shard_size = math.ceil(n / NUM_SHARDS)

    for shard_idx in range(NUM_SHARDS):
        shard_path = os.path.join(OUT_DIR, f"train-{shard_idx:05d}-of-{NUM_SHARDS:05d}.tfrecord")
        start = shard_idx * shard_size
        end   = min(start + shard_size, n)
        shard_samples = samples[start:end]

        with tf.io.TFRecordWriter(shard_path) as writer:
            for path, label in shard_samples:
                with open(path, "rb") as f:
                    jpeg_bytes = f.read()
                writer.write(make_example(jpeg_bytes, label))

        if (shard_idx + 1) % 16 == 0 or shard_idx == NUM_SHARDS - 1:
            print(f"  Shard {shard_idx+1}/{NUM_SHARDS}  ({end:,} images written)")

    print(f"\nDone. TFRecords in {OUT_DIR}")
    total_mb = sum(
        os.path.getsize(os.path.join(OUT_DIR, f))
        for f in os.listdir(OUT_DIR)
    ) / 1e6
    print(f"Total size: {total_mb:.0f} MB")


if __name__ == "__main__":
    main()
