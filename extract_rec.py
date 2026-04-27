"""
Extract MXNet RecordIO (.rec + .idx) to identity-folder structure.
Output: /tmp/webface_images/<class_id>/<img_id>.jpg

RecordIO layout per record:
  4 bytes  magic (0xced7230a)
  4 bytes  cflag<<29 | length
  --- data (length bytes) ---
  4 bytes  flag   (int32)
  4 bytes  label  (float32)   <- class id
  8 bytes  id     (uint64)
  8 bytes  id2    (uint64)
  N bytes  JPEG image
"""

import os
import struct

REC_PATH = "/tmp/webface/faces_webface_112x112/train.rec"
IDX_PATH = "/tmp/webface/faces_webface_112x112/train.idx"
OUT_DIR  = "/tmp/webface_images"

MAGIC   = 0xced7230a
IR_HDR  = struct.Struct("<IfQQ")   # flag, label, id, id2
IR_SIZE = IR_HDR.size              # 24 bytes


def read_idx(idx_path):
    """Returns sorted list of byte offsets (text format: 'rec_id\toffset\n')."""
    offsets = []
    with open(idx_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            offsets.append(int(parts[1]))
    offsets.sort()
    return offsets


def read_record(f):
    """Read one RecordIO record; returns (class_label, jpeg_bytes) or (None, None)."""
    header = f.read(8)
    if len(header) < 8:
        return None, None
    magic, length_flag = struct.unpack("<II", header)
    if magic != MAGIC:
        return None, None
    length = length_flag & 0x1FFFFFFF
    pad    = (4 - length % 4) % 4
    data   = f.read(length + pad)
    flag, label, _, _ = IR_HDR.unpack(data[:IR_SIZE])
    img_bytes = data[IR_SIZE:length]
    return int(label), bytes(img_bytes)


def main():
    print("Reading index...")
    offsets = read_idx(IDX_PATH)
    print(f"  {len(offsets):,} records found")

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Extracting to {OUT_DIR} ...")
    with open(REC_PATH, "rb") as f:
        for i, offset in enumerate(offsets):
            f.seek(offset)
            label, img_bytes = read_record(f)
            if img_bytes is None:
                continue

            class_dir = os.path.join(OUT_DIR, f"{label:07d}")
            os.makedirs(class_dir, exist_ok=True)
            img_path = os.path.join(class_dir, f"{i:07d}.jpg")
            with open(img_path, "wb") as img_f:
                img_f.write(img_bytes)

            if (i + 1) % 50000 == 0:
                print(f"  {i+1:,} / {len(offsets):,}")

    print(f"\nDone. Images in {OUT_DIR}")
    classes = len(os.listdir(OUT_DIR))
    print(f"  Classes: {classes:,}")


if __name__ == "__main__":
    main()
