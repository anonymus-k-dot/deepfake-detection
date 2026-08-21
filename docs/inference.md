# VERITAS Inference Guide

This document describes how to execute inference on video files using the Python API and the Flask REST API.

---

## 1. Prerequisites

1. Ensure dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```
2. Verify that `best_model.pt` is present in the `models/` directory:
   ```bash
   models/best_model.pt
   ```

---

## 2. Python Programmatic Inference

You can run end-to-end inference directly in Python using `InferencePreprocessor` and `DeepfakeMultiStreamModel`:

```python
from pathlib import Path
import torch
from phase3_model import DeepfakeMultiStreamModel
from preprocess import InferencePreprocessor

# 1. Device selection
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Load model
model = DeepfakeMultiStreamModel().to(device)
ckpt = torch.load("models/best_model.pt", map_location=device, weights_only=False)
state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
model.load_state_dict(state)
model.eval()

# 3. Preprocess video
preprocessor = InferencePreprocessor(device=device)
video_path = "sample_video.mp4"

(ff_t, ec_t, nc_t, tm_t), n_windows = preprocessor.process_video(video_path)

# 4. Batched inference
batch_size = 32
all_probs = []

with torch.no_grad():
    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        
        ff = ff_t[start:end].to(device)
        ec = ec_t[start:end].to(device)
        nc = nc_t[start:end].to(device)
        tm = tm_t[start:end].to(device)
        
        logits = model(ff, ec, nc, tm)
        probs = torch.sigmoid(logits).squeeze(1)
        all_probs.append(probs.cpu())

mean_prob = torch.cat(all_probs).mean().item()
verdict = "FAKE" if mean_prob >= 0.5 else "REAL"
confidence = round(mean_prob * 100, 1)

print(f"Verdict: {verdict}")
print(f"Confidence: {confidence}%")
print(f"Windows Analyzed: {n_windows}")
```

---

## 3. Flask Web Server & REST API

### Starting the Server
```bash
python app.py --host 0.0.0.0 --port 5000
```

### Health Check Endpoint
```bash
curl http://127.0.0.1:5000/health
```
**Response:**
```json
{
  "status": "ok",
  "device": "cuda",
  "model": "models/best_model.pt"
}
```

### Video Analysis Endpoint (`POST /api/analyze`)

Send a `multipart/form-data` request with the video file attached to field `video`:

```bash
curl -X POST http://127.0.0.1:5000/api/analyze \
     -F "video=@/path/to/test_video.mp4"
```

**Success Response (`200 OK`):**
```json
{
  "verdict": "FAKE",
  "confidence": 87.4,
  "windows": 18,
  "streams": {
    "eye": 34.2,
    "nose": 18.5,
    "face": 29.1,
    "temporal": 18.2
  }
}
```

**HTTP Status Codes & Error Handling:**
- **`200 OK`**: Video successfully processed and classified.
- **`400 Bad Request`**: Missing `video` field in the multipart form payload.
- **`415 Unsupported Media Type`**: File extension not in `.mp4`, `.mov`, `.webm`.
- **`422 Unprocessable Entity`**: Video is too short (< 3 frames) or MTCNN detected fewer than 3 faces.
- **`500 Internal Server Error`**: Unexpected runtime error or CUDA out-of-memory.
- **`503 Service Unavailable`**: Model checkpoint `models/best_model.pt` not found on server startup.
