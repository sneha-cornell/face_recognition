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

**Parameter summary (verified by running `model.py`):**

| | Count |
|--|--|
| Total parameters | 403,776 |
| Trainable | 396,416 |
| Non-trainable (BN stats) | 7,360 |
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

**Dataset:** CASIA-WebFace (~494,414 images, ~10,575 identities)

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

Expected performance range for a model of this scale (~400K params, 128-dim embeddings, CASIA-WebFace training):

| Benchmark | Expected Accuracy |
|---|---|
| LFW | ~98.5 – 99.0% |
| CFP-FP (frontal–profile) | ~88 – 92% |
| AgeDB-30 | ~91 – 94% |

> These are typical ranges for lightweight ArcFace models at this parameter budget. Actual results depend on training duration, data quality, and alignment preprocessing.

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
