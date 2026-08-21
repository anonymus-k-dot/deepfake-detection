# VERITAS Model Weights Directory

This directory holds the PyTorch model checkpoint (`best_model.pt`) used by the VERITAS multi-stream deepfake detection inference pipeline and Flask backend.

---

## Checkpoint Specification

| Property | Value |
|---|---|
| **Expected Filename** | `best_model.pt` |
| **Path** | `models/best_model.pt` |
| **Architecture** | `DeepfakeMultiStreamModel` |
| **Total Parameters** | 28,102,974 (~28.1M) |
| **Trainable Parameters** | 28,102,947 (27 frozen Laplacian weights) |
| **Model Size** | ~337 MB |
| **Inference Device** | CUDA (default if available) or CPU |

---

## Checkpoint Structure

The checkpoint is saved as a PyTorch dictionary containing both model parameters and training state:

```python
checkpoint = {
    "epoch": 42,                         # Epoch at which best validation AUC was achieved
    "model_state": OrderedDict(...),     # Weights for all 4 neural streams + fusion head
    "optimizer_state": dict(...),        # Adam optimizer state (3 parameter groups)
    "scheduler_state": dict(...),        # Learning rate scheduler state
    "val_auc": 0.880473,                 # Validation ROC-AUC (0.8805 / 88.05%)
    "val_f1": 0.780632,                  # Validation F1-score (0.7806 / 78.06%)
    "val_acc": 0.783821                  # Validation Accuracy (0.7838 / 78.38%)
}
```

---

## How the Checkpoint is Loaded

The application loader in `app.py` supports both wrapped metadata dictionaries and raw state dicts:

```python
from pathlib import Path
import torch
from phase3_model import DeepfakeMultiStreamModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = Path("models/best_model.pt")

model = DeepfakeMultiStreamModel().to(device)
ckpt = torch.load(model_path, map_location=device, weights_only=False)

if isinstance(ckpt, dict) and "model_state" in ckpt:
    state = ckpt["model_state"]
else:
    state = ckpt

model.load_state_dict(state)
model.eval()
```

---

## Checkpoint Acquisition & Placement

1. Obtain the trained `best_model.pt` checkpoint from your training run or designated artifact storage.
2. Place the file directly in this folder:
   ```bash
   models/best_model.pt
   ```
3. Verify that the file exists and is readable before launching the Flask server (`python app.py`). If the file is missing, the server will log an error and return HTTP 503 on analysis requests.
