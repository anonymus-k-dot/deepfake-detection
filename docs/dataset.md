# VERITAS Dataset Specification & Management

VERITAS models and trains on facial manipulation datasets structured into hierarchical HDF5 (`.h5`) containers, with support for the FaceForensics++ benchmark.

---

## 1. Dataset Origin & Composition

- **Benchmark**: FaceForensics++ (FF++) subset.
- **Classes**:
  - `Real` (Label: `0.0` / Ground Truth: Authentic)
  - `Fake` (Label: `1.0` / Ground Truth: Manipulated / Deepfake)
- **Modality**: Video sequences decomposed into facial crops and inter-frame temporal difference tensors.

---

## 2. Directory Structure

Training and evaluation expect the following directory organization under a preprocessed root folder:

```text
preprocessed_dataset/
├── Training/
│   ├── Real/
│   │   ├── video_001.h5
│   │   ├── video_002.h5
│   │   └── ...
│   └── Fake/
│       ├── video_fake_001.h5
│       ├── video_fake_002.h5
│       └── ...
├── Validation/
│   ├── Real/
│   │   └── ...
│   └── Fake/
│       └── ...
└── Testing/
    ├── Real/
    │   └── ...
    └── Fake/
        └── ...
```

---

## 3. HDF5 File Schema

Each `.h5` file represents a single preprocessed video containing $N$ aligned sliding windows. Every file must contain exactly four dataset keys:

| Dataset Key | Stored Shape | Stored Dtype | Target Tensor Shape | Description |
|---|---|---|---|---|
| `full_face` | $(N, 3, 224, 224)$ | `float32` / `uint8` | $(B, 3, 224, 224)$ | Full face crop, ImageNet normalized |
| `eye_crop` | $(N, 3, 96, 96)$ | `float32` / `uint8` | $(B, 3, 96, 96)$ | Eye region crop, ImageNet normalized |
| `nose_crop` | $(N, 3, 64, 64)$ | `float32` / `uint8` | $(B, 3, 64, 64)$ | Central nose/boundary crop, ImageNet normalized |
| `temporal` | $(N, 2W, 3, H)$ / $(N, 6, 112, 112)$ | `float32` | $(B, 6, 112, 112)$ | Inter-frame difference motion tensor |

---

## 4. Dataset Loading & PyTorch Integration

The dataset is ingested using `DeepfakeH5Dataset` (`phase3_model.py`):

```python
from phase3_model import build_dataloaders

dataloaders = build_dataloaders(
    preprocessed_root="/path/to/preprocessed_dataset",
    batch_size=8,
    num_workers=2
)

train_loader = dataloaders["train"]  # Augmentation enabled, shuffle=True, drop_last=True
val_loader   = dataloaders["val"]    # Augmentation disabled, shuffle=False
test_loader  = dataloaders["test"]   # Augmentation disabled, shuffle=False (final evaluation)
```

### Batch Tensor Output
Each iteration yields a 5-tuple:
1. `ff_t`: `(B, 3, 224, 224)` — Full face tensor
2. `ec_t`: `(B, 3, 96, 96)` — Eye crop tensor
3. `nc_t`: `(B, 3, 64, 64)` — Nose crop tensor
4. `tm_t`: `(B, 6, 112, 112)` — Temporal motion difference tensor
5. `labels`: `(B,)` — Float tensor with values `0.0` (Real) or `1.0` (Fake)

---

## 5. Data Augmentation Policy

Augmentation is applied dynamically in `DeepfakeH5Dataset` during training only:
- **Shared Random Seed**: A single random generator seed is generated per sample to ensure spatial transforms remain synchronized across all spatial and temporal streams.
- **Horizontal Flip**: Applied with $p = 0.5$.
- **90° Discrete Rotations**: Applied with $k \in \{0, 1, 2, 3\}$ ($0^\circ, 90^\circ, 180^\circ, 270^\circ$).
- Validation and Testing splits are loaded without augmentation (`augment=False`).
