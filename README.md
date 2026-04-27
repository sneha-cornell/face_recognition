# Face Recognition — Lightweight ArcFace Backbone

A compact face recognition system built for edge deployment. The backbone (~404K parameters, 1.6 MB fp32) is trained with ArcFace loss on CASIA-WebFace and produces 128-dimensional embeddings for cosine-similarity matching.

---

## Model Architecture

**Backbone: Expanded Depthwise-Separable Network**

| Stage | Output Shape | Block | Expansion | Stride |
|-------|-------------|-------|-----------|--------|
| Stem Conv3×3 | 56×56×32 | Conv-BN-ReLU | — | 2 |
| Block 1 | 56×56×32 | Expand-DW-Project | 1 | 1 |
| Block 2 | 28×28×64 | Expand-DW-Project | 2 | 2 |
| Block 3 | 28×28×64 | Expand-DW-Project | 2 | 1 |
| Block 4 | 28×28×128 | Expand-DW-Project | 2 | 1 |
| Block 5 | 14×14×128 | Expand-DW-Project | 2 | 2 |
| Block 6 | 14×14×128 | Expand-DW-Project | 2 | 1 |
| Block 7 | 14×14×128 | Expand-DW-Project | 2 | 1 |
| Block 8 | 7×7×256 | Expand-DW-Project | 2 | 2 |
| GAP | 256 | AveragePool 7×7 | — | — |
| Embedding | **128** | Dense (no bias) | — | — |

Each block uses the pattern: `1×1 expand → DepthwiseConv3×3 → 1×1 linear project (no activation)`. No residual additions — designed for GPX10 compiler compatibility.

The embedding head is `Dense(128) → BatchNormalization`. The BN is critical: without it, embedding norms grow unbounded (~480+ observed without BN), which shrinks gradients through the ArcFace L2-normalization step and causes representation collapse. BN keeps norms bounded so ArcFace training stays well-conditioned throughout.

**Parameter summary (verified by running `model.py`):**

| | Count |
|--|--|
| Total parameters | 404,288 |
| Trainable | 396,416 |
| Non-trainable (BN stats) | 7,872 |
| Model size (fp32) | **1.62 MB** |
| Model size (int8 quantized) | **~0.40 MB** |

---

## Loss Function: ArcFace

ArcFace (Additive Angular Margin Loss, Deng et al. CVPR 2019) is used during training. It adds an angular margin *m* to the ground-truth class angle before softmax, pushing intra-class embeddings tighter and inter-class embeddings further apart on the unit hypersphere.

```
logit_target = scale × cos(θ + m)
logit_others = scale × cos(θ)
```

Default hyperparameters:

| Hyperparameter | Value |
|---|---|
| Margin *m* | 0.5 rad (~28.6°) |
| Scale *s* | 64.0 |

The ArcFace head is training-only and is discarded at inference — only the backbone is deployed.

---

## Training

### Dataset: CASIA-WebFace

Pre-aligned 112×112 face crops in RGB JPEG format. Each image contains exactly **1 face** — the dataset is pre-cropped and landmark-aligned (5-point: eyes, nose, mouth corners) before storage.

**Scale (measured on this dataset):**

| Stat | Value |
|---|---|
| Total images | 490,623 |
| Identities | 10,572 |
| Images per identity — mean | 46.4 |
| Images per identity — median | 27 |
| Images per identity — min / max | 2 / 802 |
| Images per identity — std | 59.3 |
| Identities with ≥ 20 images | 7,502 (71%) |
| Identities with ≥ 50 images | 2,511 (24%) |

**Image quality (measured over 52,855 sampled images):**

| Metric | Value |
|---|---|
| Resolution | 112 × 112 px (fixed, pre-aligned) |
| Faces per image | 1 (pre-cropped) |
| Sharpness — mean Laplacian variance | 492.5 |
| Sharpness — median | 325.0 |
| Blurry images (Laplacian var < 100) | **19.4%** |
| Brightness — mean (0–255) | 107.8 |
| Brightness — std | 31.1 |
| Dark images (mean < 64) | 8.1% |
| Over-exposed (mean > 200) | 0.1% |
| Contrast — mean pixel std | 54.7 |
| Decode errors | 0 |

The 19% blurry fraction reflects real-world conditions in the original CASIA-WebFace scrape: motion blur, low-resolution source frames, and out-of-focus captures. The distribution is long-tailed — most images are sharp (median 325) with a minority of poor-quality examples that add robustness.

**Training configuration:**

| Setting | Value |
|---|---|
| Input resolution | 112 × 112 |
| Normalisation | `(pixel − 127.5) / 128.0` → [−1, 1] |
| Batch size | 512 |
| Epochs | 40 |
| Optimiser | SGD, momentum=0.9, weight decay=5e-4 |
| LR schedule | Cosine decay, initial LR=0.1 |
| Embedding dim | 128 |
| Hardware | Apple M5, Metal GPU (tensorflow-metal 1.2.0 + TF 2.17) |
| ArcFace scale | 32 (subset), 64 (full run) |
| ArcFace margin | 0.3 (subset), 0.5 (full run) |

**Augmentation (training only):** random horizontal flip, brightness jitter ±0.15, contrast jitter 0.85–1.15.

```bash
python train.py --data_dir /path/to/CASIA-WebFace --epochs 40
```

---

## Evaluation & Metrics

Face recognition models are evaluated on two standard tasks:

### 1. Verification (1:1)
Given two face images, predict whether they belong to the same person.

**Primary benchmark: LFW (Labeled Faces in the Wild)**

| Metric | Description |
|---|---|
| Accuracy | Fraction of correctly classified pairs at optimal threshold |
| TAR @ FAR=1e-3 | True Accept Rate when False Accept Rate = 0.1% |
| TAR @ FAR=1e-4 | True Accept Rate when False Accept Rate = 0.01% |
| AUC | Area under ROC curve |

Results are updated after training completes (see below). Reference ranges for lightweight ArcFace at this parameter budget (~400K params, 128-dim):

| Benchmark | Reference Range |
|---|---|
| LFW | ~98.5 – 99.0% |
| CFP-FP (frontal–profile) | ~88 – 92% |
| AgeDB-30 | ~91 – 94% |

> Training in progress on Apple M5 (Metal GPU). Results will be filled in once the 40-epoch run on CASIA-WebFace completes.

### 2. Identification (1:N)
Given a probe image, find the closest match from an enrolled gallery (used by the `verify` command in `inference.py`).

**Metric:** Cosine similarity with threshold (default = 0.40).

| Decision | Meaning |
|---|---|
| similarity ≥ threshold | ACCESS GRANTED — identity matched |
| similarity < threshold | ACCESS DENIED — no match |

Threshold tuning: plot the impostor vs. genuine similarity distributions on a held-out set and choose the operating point that satisfies your FAR budget.

---

## Inference

### Enroll a person
```bash
python inference.py enroll \
  --model checkpoints/backbone_final.h5 \
  --name "Alice" \
  --images alice1.jpg alice2.jpg alice3.jpg
```
Stores the mean embedding of all provided images in `face_db.json`.

### Verify a query image
```bash
python inference.py verify \
  --model checkpoints/backbone_final.h5 \
  --image doorbell_capture.jpg \
  --threshold 0.40
```

---

## File Structure

```
├── model.py          # Backbone definition (Expanded DW+PW blocks)
├── loss.py           # ArcFace loss layer
├── train.py          # Training loop (ArcFace + SGD + cosine LR)
├── dataset.py        # tf.data pipeline for CASIA-WebFace layout
├── inference.py      # Enroll / verify CLI
└── requirements.txt  # tensorflow>=2.13.0, numpy>=1.24.0
```

---

## Requirements

```
tensorflow>=2.13.0
numpy>=1.24.0
```

---

## References

- Deng et al., [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698), CVPR 2019
- Howard et al., MobileNetV2 (inverted residual / linear bottleneck design)
- CASIA-WebFace dataset: Yi et al., 2014
