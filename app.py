
import os
import logging
import tempfile
from pathlib import Path

import torch
import numpy as np
from flask import Flask, request, jsonify,render_template
from flask_cors import CORS

from phase3_model import DeepfakeMultiStreamModel
from preprocess import InferencePreprocessor

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("veritas")

MODEL_PATH      = Path("models/best_model.pt")
INFERENCE_BATCH = 32                          
MAX_UPLOAD_MB   = 500
ALLOWED_EXT     = {".mp4", ".mov", ".webm"}

# Select inference device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f"Inference device: {DEVICE}")


def _load_model(path: Path, device: str) -> DeepfakeMultiStreamModel:
    """
    Load DeepfakeMultiStreamModel from a training checkpoint.
    Accepts both raw state dicts and checkpoints wrapped with training metadata.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"\n{'─'*60}\n"
            f"  Model checkpoint not found: {path.resolve()}\n"
            f"  Place best_model.pt inside the models/ folder:\n"
            f"    {Path('models').resolve()}/best_model.pt\n"
            f"{'─'*60}"
        )

    model = DeepfakeMultiStreamModel().to(device)

    log.info(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Handle both wrapped ({model_state: ...}) and raw state dicts
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state      = ckpt["model_state"]
        epoch_info = f"epoch {ckpt.get('epoch', '?')}  val_AUC={ckpt.get('val_auc', '?')}"
    else:
        state      = ckpt
        epoch_info = "(raw state dict)"

    model.load_state_dict(state)
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    log.info(f"Model ready — {total:,} params  checkpoint: {epoch_info}")
    return model


# Load once at startup — shared across all requests
try:
    model        = _load_model(MODEL_PATH, DEVICE)
    preprocessor = InferencePreprocessor(device=DEVICE)
except FileNotFoundError as e:
    log.error(str(e))
    model        = None
    preprocessor = None


app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# =============================================================================
# INFERENCE
# =============================================================================

def _run_inference(
    full_face_t: torch.Tensor,
    eye_crop_t:  torch.Tensor,
    nose_crop_t: torch.Tensor,
    temporal_t:  torch.Tensor,
):
    """
    Run batched multi-stream inference over all N windows.

    Stream contribution scores are computed as the mean absolute activation
    of each stream's features projected through the first fusion layer —
    a fast approximation of how much each stream influences the final logit.

    Returns:
        mean_prob    float  — mean P(fake) across all windows  ∈ [0, 1]
        stream_pcts  dict   — per-stream contribution  ∈ [0, 100]
    """
    n = full_face_t.shape[0]
    all_probs  = []
    contrib    = {"eye": 0.0, "nose": 0.0, "face": 0.0, "temporal": 0.0}

    model.eval()
    with torch.no_grad():
        for start in range(0, n, INFERENCE_BATCH):
            end = min(start + INFERENCE_BATCH, n)

            ff = full_face_t[start:end].to(DEVICE)
            ec = eye_crop_t[start:end].to(DEVICE)
            nc = nose_crop_t[start:end].to(DEVICE)
            tm = temporal_t[start:end].to(DEVICE)

            # ── Per-stream feature extraction ──────────────────────────────
            feat_eye  = model.eye_stream(ec)            # (B, 256)
            feat_nose = model.nose_stream(nc)           # (B, 256)
            feat_face = model.full_face_stream(ff)      # (B, 512)
            feat_temp = model.temporal_stream(tm)       # (B, 256)

            # ── Fusion → logit → probability ──────────────────────────────
            fused  = torch.cat([feat_eye, feat_nose, feat_face, feat_temp], dim=1)
            logits = model.fusion_head(fused)           # (B, 1)
            probs  = torch.sigmoid(logits).squeeze(1)   # (B,)
            all_probs.append(probs.cpu())

            # ── Stream contribution (weight × activation magnitude) ────────
            # Fusion head first layer: Linear(1280 → 512)
            # Slice weights according to concatenation order:
            #   [0:256]   → eye    [256:512]  → nose
            #   [512:1024] → face  [1024:1280] → temporal
            W = model.fusion_head[0].weight             # (512, 1280)

            def _contrib(w_slice, feat):
                # Mean absolute value of (W_slice @ features^T) across hidden units
                return (w_slice @ feat.T).abs().mean().item()

            contrib["eye"]      += _contrib(W[:, :256],     feat_eye)
            contrib["nose"]     += _contrib(W[:, 256:512],  feat_nose)
            contrib["face"]     += _contrib(W[:, 512:1024], feat_face)
            contrib["temporal"] += _contrib(W[:, 1024:],    feat_temp)

    # Average probability across all windows
    mean_prob = torch.cat(all_probs).mean().item()

    # Normalise stream contributions to percentages
    total = sum(contrib.values()) + 1e-8
    stream_pcts = {k: round(v / total * 100, 1) for k, v in contrib.items()}

    return mean_prob, stream_pcts


# =============================================================================
# ROUTES
# =============================================================================

@app.route("/health", methods=["GET"])
@app.route('/')
def index():
    # Serves index.html from the current directory
    return render_template('index.html')
def health():
    """Liveness check."""
    return jsonify({
        "status": "ok" if model is not None else "model_not_loaded",
        "device": DEVICE,
        "model":  str(MODEL_PATH),
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Analyse a video for deepfake indicators.

    Request: multipart/form-data  field "video"
    Response: JSON verdict object
    """
    # ── Guard: model loaded ────────────────────────────────────────────────────
    if model is None or preprocessor is None:
        return jsonify({
            "error": (
                f"Model not loaded. Place best_model.pt in "
                f"{MODEL_PATH.resolve().parent} and restart the server."
            )
        }), 503

    # ── Validate upload ────────────────────────────────────────────────────────
    if "video" not in request.files:
        return jsonify({
            "error": "No video file provided. Send a multipart/form-data request "
                     "with field name 'video'."
        }), 400

    f   = request.files["video"]
    ext = Path(f.filename or "").suffix.lower()

    if ext not in ALLOWED_EXT:
        return jsonify({
            "error": f"Unsupported format '{ext}'. Accepted: MP4, MOV, WEBM."
        }), 415

    # Save upload to temp file ───────────────────────────────────────────────
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            f.save(tmp.name)
            tmp_path = tmp.name

        size_mb = os.path.getsize(tmp_path) / 1e6
        log.info(f"Received: '{f.filename}'  {size_mb:.1f} MB")

        # ── Preprocessing ──────────────────────────────────────────────────────
        (full_face_t, eye_crop_t, nose_crop_t, temporal_t), n_windows = \
            preprocessor.process_video(tmp_path)

        log.info(f"Preprocessing done — {n_windows} windows")

        # ── Inference ──────────────────────────────────────────────────────────
        mean_prob, stream_pcts = _run_inference(
            full_face_t, eye_crop_t, nose_crop_t, temporal_t
        )

        confidence = round(mean_prob * 100, 1)
        verdict    = "FAKE" if mean_prob >= 0.5 else "REAL"

        log.info(
            f"Result: {verdict}  confidence={confidence}%  "
            f"windows={n_windows}  streams={stream_pcts}"
        )

        return jsonify({
            "verdict":    verdict,
            "confidence": confidence,
            "windows":    n_windows,
            "streams":    stream_pcts,
        })

    except ValueError as e:
        # Raised by preprocessor for short/dark/faceless videos
        log.warning(f"Preprocessing rejected '{f.filename}': {e}")
        return jsonify({"error": str(e)}), 422

    except torch.cuda.OutOfMemoryError:
        log.error("CUDA out of memory — reduce INFERENCE_BATCH or use CPU")
        return jsonify({
            "error": "GPU out of memory. Try a shorter video or restart the server."
        }), 500

    except Exception as e:
        log.exception(f"Unexpected error analysing '{f.filename}'")
        return jsonify({"error": f"Internal server error: {e}"}), 500

    finally:
        # Always clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Veritas deepfake detection server")
    parser.add_argument("--host",  default="0.0.0.0",  help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port",  default=5000, type=int, help="Bind port (default: 5000)")
    parser.add_argument("--debug", action="store_true",    help="Enable Flask debug mode")
    args = parser.parse_args()

    log.info(f"Starting Veritas server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)