# VERITAS Video Preprocessing Pipeline

This document details the video-to-tensor preprocessing pipeline implemented in `preprocess.py` through the `InferencePreprocessor` class, which matches the training data preprocessing.

---

## 1. Pipeline Overview

```text
Input Video File
       │
       ▼
[Frame Extraction]  ───▶  10 FPS sampling (RGB format)
       │
       ▼
[Face Detection]    ───▶  MTCNN (conf ≥ 0.95, 5-point landmarks)
       │
       ├──────────────────────┬──────────────────────┬──────────────────────┐
       ▼                      ▼                      ▼                      ▼
 [Full Face]             [Eye Crop]             [Nose Crop]         [Temporal Triplets]
  (224×224)               (96×96)                (64×64)             (t, t+1, t+2)
       │                      │                      │                      │
       ▼                      ▼                      ▼                      ▼
ImageNet Norm          ImageNet Norm          ImageNet Norm         Diff1 & Diff2 (|Δf|)
(3, 224, 224)           (3, 96, 96)            (3, 64, 64)           (6, 112, 112)
```

---

## 2. Hyperparameters & Configuration

| Parameter | Value | Rationale |
|---|---|---|
| `TARGET_FPS` | `10` | Standardizes video temporal sampling across variable framerates |
| `MIN_FACES` | `3` | Minimum valid frames required to compute temporal motion difference |
| `CONFIDENCE_THRESHOLD` | `0.95` | Eliminates false positive face detections and background noise |
| `CROP_SIZES['full_face']` | `(224, 224)` | ResNet-18 spatial input standard |
| `CROP_SIZES['eye']` | `(96, 96)` | High-resolution focus on blinking and iris artifacts |
| `CROP_SIZES['nose']` | `(64, 64)` | Focused field on facial mask boundary blending seams |
| `TEMPORAL_H`, `TEMPORAL_W`| `(112, 112)` | Spatial resolution for motion difference maps |

---

## 3. Step-by-Step Execution

### Step 1: Video Ingestion & Frame Sampling (`_extract_frames`)
- Opens the video file using `cv2.VideoCapture`.
- Reads the native FPS: $R_{\text{native}}$.
- Computes the frame skip interval:
  $$\text{interval} = \max\left(1, \text{round}\left(\frac{R_{\text{native}}}{10}\right)\right)$$
- Samples every $\text{interval}$-th frame and converts from BGR to RGB.
- Raises `ValueError` if fewer than 3 frames are retrieved.

### Step 2: MTCNN Face Detection & Crop Extraction (`_detect_and_crop`)
Configured MTCNN parameters:
- `image_size=224`, `margin=20`, `select_largest=True`, `post_process=False`, `keep_all=False`.
- For each frame:
  1. Detects bounding box coordinates $[x_1, y_1, x_2, y_2]$ and 5 landmarks (`left_eye`, `right_eye`, `nose`, `mouth_left`, `mouth_right`).
  2. If `prob < 0.95` or bounding box is empty, the frame is skipped.
  3. Crops full face and resizes to **224×224**.
  4. Calls `_extract_specific_crops` to compute:
     - **Eye Crop (96×96)**: Center $(cx, cy)$ computed as the midpoint between left eye and right eye. Extracted as a bounding box $[cx - 48, cx + 48] \times [cy - 48, cy + 48]$.
     - **Nose Crop (64×64)**: Center $(nx, ny)$ positioned directly at the nose landmark. Extracted as a bounding box $[nx - 32, nx + 32] \times [ny - 32, ny + 32]$.
  5. Converts arrays to channel-first float format: `(3, H, W)` normalized to $[0, 1]$.

### Step 3: Temporal Motion Difference Tensor (`_generate_temporal_tensors`)
For every 3 consecutive full-face frames $(f_t, f_{t+1}, f_{t+2})$:
1. Calculates frame-to-frame absolute differences:
   $$\text{diff}_1 = |f_{t+1} - f_t|, \quad \text{diff}_2 = |f_{t+2} - f_{t+1}|$$
2. Concatenates differences along channels to form a 6-channel difference tensor.
3. Resizes spatially to fixed **(112, 112)** resolution via bilinear interpolation.

### Step 4: PyTorch Normalization & Alignment
- **Spatial Streams**: Converted to `torch.FloatTensor` and normalized using ImageNet channel statistics:
  $$\mathbf{x}_{\text{norm}} = \frac{\mathbf{x} - \boldsymbol{\mu}}{\boldsymbol{\sigma}}, \quad \boldsymbol{\mu} = [0.485, 0.456, 0.406], \quad \boldsymbol{\sigma} = [0.229, 0.224, 0.225]$$
- **Temporal Stream**: Output as a `(N, 6, 112, 112)` float32 tensor.
- Returns aligned 4-tuple of PyTorch tensors and total window count $N$.
