

import logging
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from facenet_pytorch import MTCNN

log = logging.getLogger("veritas.preprocess")

# ImageNet normalisation (same as DeepfakeH5Dataset._spatial_to_tensor)
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class InferencePreprocessor:
    """
    Single-video preprocessing for model inference.

    All hyperparameters and data-formatting decisions are kept identical to
    the training pipeline so the model sees the same input distribution.

    Usage:
        preprocessor = InferencePreprocessor(device="cuda")
        (ff, ec, nc, tm), n_windows = preprocessor.process_video("/path/to/video.mp4")
        # ff: (N, 3, 224, 224)   ec: (N, 3, 96, 96)
        # nc: (N, 3, 64, 64)     tm: (N, 6, 112, 112)
    """

    # ── Hyperparameters — must match training ─────────────────────────────────
    TARGET_FPS            = 10
    MIN_FACES             = 3           # minimum detected faces to proceed
    CONFIDENCE_THRESHOLD  = 0.95        # MTCNN face detection threshold
    CROP_SIZES = {
        "full_face": (224, 224),
        "eye":       (96,  96),
        "nose":      (64,  64),
    }
    TEMPORAL_H = 112
    TEMPORAL_W = 112

    def __init__(self, device: str = "cpu"):
        """
        Args:
            device: "cuda" or "cpu". MTCNN runs on this device.
                    Model inference device should match the caller's device.
        """
        self.device = device
        log.info(f"Initialising MTCNN on device={device}")
        self.mtcnn = MTCNN(
            image_size=224,
            margin=20,
            keep_all=False,
            post_process=False,
            device=device,
            select_largest=True,
        )
        log.info("MTCNN ready")

    # ── Public API ────────────────────────────────────────────────────────────

    def process_video(self, video_path: str):
        """
        Process a video file and return aligned tensor windows.

        Args:
            video_path: Absolute or relative path to the video file.

        Returns:
            Tuple:
                (full_face_t, eye_crop_t, nose_crop_t, temporal_t): CPU tensors
                    full_face_t : (N, 3, 224, 224)  ImageNet-normalised
                    eye_crop_t  : (N, 3,  96,  96)  ImageNet-normalised
                    nose_crop_t : (N, 3,  64,  64)  ImageNet-normalised
                    temporal_t  : (N, 6, 112, 112)  motion diffs, raw float32
                n_windows (int): number of valid sliding windows (= N)

        Raises:
            ValueError: too few frames / faces detected, or video unreadable.
        """
        path = Path(video_path)
        log.info(f"Processing: {path.name}")

        # ── Step 1: Frame extraction ──────────────────────────────────────────
        frames = self._extract_frames(str(path))
        if len(frames) < 3:
            raise ValueError(
                f"Video '{path.name}' produced only {len(frames)} frames "
                f"at {self.TARGET_FPS} fps (minimum 3 required). "
                f"The video may be too short or unreadable."
            )
        log.info(f"Extracted {len(frames)} frames")

        # ── Step 2: MTCNN detection & crop extraction ─────────────────────────
        full_faces, eye_crops, nose_crops = self._detect_and_crop(frames)

        if len(full_faces) < self.MIN_FACES:
            raise ValueError(
                f"Only {len(full_faces)} frames had a detectable face "
                f"(confidence ≥ {self.CONFIDENCE_THRESHOLD}). "
                f"Ensure the video contains a clearly visible, unobstructed face."
            )
        log.info(f"Face detected in {len(full_faces)}/{len(frames)} frames")

        # ── Step 3: Temporal difference tensors (exact training replica) ──────
        # full_faces is a list of (3, H, W) float32 [0, 1] CHW arrays
        temporal_arrays = self._generate_temporal_tensors(np.array(full_faces))
        n_windows = min(
            len(full_faces),
            len(eye_crops),
            len(nose_crops),
            len(temporal_arrays),
        )

        if n_windows == 0:
            raise ValueError(
                f"No valid windows produced from '{path.name}'. "
                f"Video may be too short for temporal analysis."
            )

        # ── Step 4: Convert to normalised PyTorch tensors ─────────────────────
        full_face_t = torch.stack([
            self._spatial_to_tensor(full_faces[i]) for i in range(n_windows)
        ])   # (N, 3, 224, 224)

        eye_crop_t = torch.stack([
            self._spatial_to_tensor(eye_crops[i]) for i in range(n_windows)
        ])   # (N, 3, 96, 96)

        nose_crop_t = torch.stack([
            self._spatial_to_tensor(nose_crops[i]) for i in range(n_windows)
        ])   # (N, 3, 64, 64)

        temporal_t = torch.stack([
            self._temporal_to_tensor(temporal_arrays[i]) for i in range(n_windows)
        ])   # (N, 6, 112, 112)

        log.info(
            f"Tensors ready — {n_windows} windows | "
            f"ff={tuple(full_face_t.shape)}, tm={tuple(temporal_t.shape)}"
        )
        return (full_face_t, eye_crop_t, nose_crop_t, temporal_t), n_windows

    # ── Private: frame extraction ─────────────────────────────────────────────

    def _extract_frames(self, video_path: str) -> list:
        """
        Extract RGB frames from video at TARGET_FPS using OpenCV.
        Uses the same frame_interval rounding as the training pipeline.
        """
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps <= 0:
            cap.release()
            raise ValueError(
                f"Cannot read FPS from video '{video_path}'. "
                f"File may be corrupt or in an unsupported codec."
            )

        frame_interval = max(1, round(original_fps / self.TARGET_FPS))
        frames = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_count += 1

        cap.release()
        return frames

    # ── Private: MTCNN detection & crop extraction ────────────────────────────

    def _detect_and_crop(self, frames: list):
        """
        Run MTCNN on each frame and extract full-face, eye, and nose crops.

        Returns three lists of CHW float32 [0, 1] numpy arrays.
        Frames where MTCNN confidence < CONFIDENCE_THRESHOLD are skipped.
        """
        full_faces, eye_crops, nose_crops = [], [], []

        for frame_rgb in frames:
            boxes, probs, landmarks = self.mtcnn.detect(frame_rgb, landmarks=True)

            if boxes is None or probs[0] < self.CONFIDENCE_THRESHOLD:
                continue

            h, w = frame_rgb.shape[:2]
            box  = boxes[0]
            x1   = max(0, int(box[0]));  y1 = max(0, int(box[1]))
            x2   = min(w, int(box[2]));  y2 = min(h, int(box[3]))

            face_crop = frame_rgb[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            full_face = cv2.resize(face_crop, self.CROP_SIZES["full_face"])
            eye_crop, nose_crop = self._extract_specific_crops(frame_rgb, landmarks[0])

            # Store as CHW float32 [0, 1] — matches process_video() in training
            full_faces.append(np.transpose(full_face,  (2, 0, 1)) / 255.0)
            eye_crops.append( np.transpose(eye_crop,   (2, 0, 1)) / 255.0)
            nose_crops.append(np.transpose(nose_crop,  (2, 0, 1)) / 255.0)

        return full_faces, eye_crops, nose_crops

    def _extract_specific_crops(self, image: np.ndarray, landmarks) -> tuple:
        """
        Extract eye and nose crops using MTCNN 5-point landmarks.
        Landmark order: [left_eye, right_eye, nose, mouth_left, mouth_right].
        Crop offsets are identical to extract_specific_crops() in training.
        """
        h, w = image.shape[:2]
        left_eye, right_eye, nose = landmarks[0], landmarks[1], landmarks[2]

        # Eye crop — centred between both eyes (±48 px radius)
        eye_cx = int((left_eye[0] + right_eye[0]) / 2)
        eye_cy = int((left_eye[1] + right_eye[1]) / 2)
        ex1 = max(0, eye_cx - 48);  ex2 = min(w, eye_cx + 48)
        ey1 = max(0, eye_cy - 48);  ey2 = min(h, eye_cy + 48)
        eye_crop = cv2.resize(image[ey1:ey2, ex1:ex2], self.CROP_SIZES["eye"])

        # Nose crop — centred on nose landmark (±32 px radius)
        nx, ny = int(nose[0]), int(nose[1])
        nx1 = max(0, nx - 32);  nx2 = min(w, nx + 32)
        ny1 = max(0, ny - 32);  ny2 = min(h, ny + 32)
        nose_crop = cv2.resize(image[ny1:ny2, nx1:nx2], self.CROP_SIZES["nose"])

        return eye_crop, nose_crop

    # ── Private: temporal difference tensors ─────────────────────────────────

    def _generate_temporal_tensors(self, full_face_frames: np.ndarray) -> list:
        """
        Compute sliding-window temporal difference tensors.

        IMPORTANT: Replicates the training preprocessing *exactly*, including
        the axis=-1 concatenation on CHW arrays. This produces arrays of shape
        (2W, 3, H) rather than (6, H, W), and _temporal_to_tensor handles
        the reshape — matching how the model was trained.

        full_face_frames: (N, 3, H, W) float32 [0, 1]
        Returns: list of (N-2) arrays, each shape (2W, 3, H)
        """
        if len(full_face_frames) < 3:
            return []

        tensors = []
        for i in range(len(full_face_frames) - 2):
            # Training code divides by 255 again here (input is already [0,1])
            # Kept for consistency — model was trained with these values
            f_t  = full_face_frames[i].astype(np.float32)     / 255.0
            f_t1 = full_face_frames[i + 1].astype(np.float32) / 255.0
            f_t2 = full_face_frames[i + 2].astype(np.float32) / 255.0

            diff1 = np.abs(f_t1 - f_t)                         # (3, H, W)
            diff2 = np.abs(f_t2 - f_t1)                        # (3, H, W)

            # axis=-1 on CHW → concat along W → (3, H, 2W)
            # then transpose (2,0,1)           → (2W, 3, H)
            # Matches training generate_temporal_tensors() identically
            temporal_tensor = np.concatenate((diff1, diff2), axis=-1)
            temporal_tensor = np.transpose(temporal_tensor, (2, 0, 1))
            tensors.append(temporal_tensor)

        return tensors

    # ── Private: tensor converters (mirrors DeepfakeH5Dataset exactly) ────────

    @staticmethod
    def _spatial_to_tensor(arr: np.ndarray) -> torch.Tensor:
        """
        CHW float32 [0, 1] numpy → ImageNet-normalised CHW tensor.
        Mirrors DeepfakeH5Dataset._spatial_to_tensor exactly.
        """
        arr = np.asarray(arr, dtype=np.float32)
        if arr.max() > 1.5:          # guard against uint8 input
            arr = arr / 255.0
        t = torch.from_numpy(arr)
        if t.dim() == 3 and t.shape[-1] in (1, 3):   # HWC → CHW
            t = t.permute(2, 0, 1)
        t = (t - _IMAGENET_MEAN) / _IMAGENET_STD
        return t

    def _temporal_to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        """
        Converts (2W, 3, H) training-pipeline temporal array → (6, 112, 112).
        Mirrors DeepfakeH5Dataset._temporal_to_tensor exactly, handling the
        (2W, 3, H) artifact produced by _generate_temporal_tensors.
        """
        arr = np.asarray(arr, dtype=np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
        t = torch.from_numpy(arr.copy())

        if t.dim() == 3:
            c0, c1, c2 = t.shape

            if c0 == 6:
                # Already (6, H, W) — pass through
                pass

            elif c2 == 6:
                # (H, W, 6) HWC
                t = t.permute(2, 0, 1).contiguous()

            elif c0 % 2 == 0 and c2 == 3:
                # (2H, W, 3) vertically stacked, channels last
                H, W = c0 // 2, c1
                t = (t.reshape(2, H, W, 3).permute(0, 3, 1, 2)
                      .reshape(6, H, W).contiguous())

            elif c0 % 2 == 0 and c1 == 3:
                # (2W, 3, H) — produced by our _generate_temporal_tensors
                # Matches training _temporal_to_tensor case "c0%2==0 and c1==3"
                H = c0 // 2
                W = c2
                t = (t.reshape(2, H, 3, W)
                      .permute(0, 2, 1, 3)
                      .reshape(6, H, W)
                      .contiguous())

            else:
                raise ValueError(
                    f"Cannot interpret temporal shape {tuple(t.shape)}. "
                    f"Expected (6,H,W), (H,W,6), (2H,W,3), or (2W,3,H)."
                )

        elif t.dim() == 4:
            c0, c1, c2, c3 = t.shape
            if c0 == 2 and c3 == 3:
                t = t.permute(0, 3, 1, 2).reshape(6, c2, c3).contiguous()
            elif c0 == 2 and c1 == 3:
                t = t.reshape(6, c2, c3).contiguous()
            else:
                raise ValueError(f"Cannot interpret 4D temporal shape {tuple(t.shape)}")

        else:
            raise ValueError(f"Temporal tensor has {t.dim()} dims; expected 3 or 4.")

        # Resize to fixed (TEMPORAL_H, TEMPORAL_W) for batch collation
        if t.shape[1] != self.TEMPORAL_H or t.shape[2] != self.TEMPORAL_W:
            t = F.interpolate(
                t.unsqueeze(0),
                size=(self.TEMPORAL_H, self.TEMPORAL_W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return t   # (6, 112, 112)