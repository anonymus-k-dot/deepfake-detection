# VERITAS System Architecture

VERITAS is an end-to-end deepfake detection system designed to identify facial manipulation in video sequences by combining localized spatial artifact analysis with inter-frame temporal motion modeling.

---

## 1. High-Level Pipeline Flow

```mermaid
flowchart TD
    subgraph Client["Client / Upload Interface (templates/index.html)"]
        A["Input Video<br/><code>.mp4 / .mov / .webm</code>"]
        WebUI["Interactive Web UI & REST API<br/><code>POST /api/analyze</code>"]
    end

    subgraph Preprocessing["Preprocessing Pipeline (preprocess.py)"]
        B["OpenCV Frame Extraction<br/><b>10 FPS Sampling</b>"]
        C["MTCNN Face & Landmark Detection<br/><b>Conf ≥ 0.95 (5 Landmarks)</b>"]
        
        D1["Eye Region Crop<br/><code>96 × 96</code> (ImageNet Norm)"]
        D2["Nose Region Crop<br/><code>64 × 64</code> (ImageNet Norm)"]
        D3["Full Face Crop<br/><code>224 × 224</code> (ImageNet Norm)"]
        D4["Temporal Window<br/><code>(N-2) × 6 × 112 × 112</code> (Diffs)"]
    end

    subgraph Model["Multi-Stream Neural Network (phase3_model.py)"]
        subgraph S1["Stream 1: Eye"]
            E1["Fixed Laplacian Filter<br/>+ Mod ResNet-18 + SEBlock<br/><b>Output: 256-d</b>"]
        end
        subgraph S2["Stream 2: Nose"]
            E2["4 Conv Blocks (Dilated)<br/>+ Full CBAM Attention<br/><b>Output: 256-d</b>"]
        end
        subgraph S3["Stream 3: Full Face"]
            E3["ResNet-18 CNN (49 Patches)<br/>+ 4-Layer ViT Encoder (Pre-LN)<br/><b>Output: 512-d</b>"]
        end
        subgraph S4["Stream 4: Motion"]
            E4["3 Conv Blocks<br/>+ SEBlock Attention<br/><b>Output: 256-d</b>"]
        end
    end

    subgraph Fusion["Fusion Head & Decision Engine (app.py)"]
        F["Feature Concatenation<br/><b>1280-d Vector</b>"]
        G["Fusion Classification Head<br/><code>Linear(1280 → 512) → BN → ReLU</code><br/><code>Dropout(0.4) → Linear(512 → 1)</code>"]
        H["Sigmoid Activation & Aggregation<br/><b>P(fake) ∈ [0, 1] across N windows</b>"]
        I["Inference Verdict & Attribution<br/><code>REAL / FAKE</code> + Stream Breakdown"]
    end

    A --> WebUI
    WebUI -->|"Video Payload"| B
    B -->|"RGB Frames"| C
    C -->|"Midpoint (±48px)"| D1
    C -->|"Nose Point (±32px)"| D2
    C -->|"Face Bounding Box"| D3
    C -->|"Consecutive Triplets"| D4

    D1 --> E1
    D2 --> E2
    D3 --> E3
    D4 --> E4

    E1 -->|"256-d"| F
    E2 -->|"256-d"| F
    E3 -->|"512-d"| F
    E4 -->|"256-d"| F

    F --> G
    G -->|"Scalar Logit"| H
    H --> I
    I -->|"JSON Response"| WebUI
```

---

## 2. Pipeline Stages

### Stage 1: Frame Extraction (`preprocess.py`)
- The input video is opened with OpenCV (`cv2.VideoCapture`).
- Frames are sampled at a fixed target rate of **10 FPS** using a calculated interval: `frame_interval = round(original_fps / 10)`.
- Frames are converted from OpenCV's BGR to RGB color space.

### Stage 2: Face Detection & Landmark Extraction (`preprocess.py`)
- Multi-task Cascaded Convolutional Networks (MTCNN) detects facial bounding boxes and 5 key landmarks (`left_eye`, `right_eye`, `nose`, `mouth_left`, `mouth_right`).
- Detections with confidence below `0.95` are discarded.
- Videos with fewer than 3 detected faces are rejected.

### Stage 3: Multi-Region Spatial & Temporal Cropping (`preprocess.py`)
Four distinct representations are extracted per valid sliding window:
1. **Full Face**: Bounding box crop resized to **224×224**, normalized with ImageNet mean and standard deviation.
2. **Eye Region**: Crop centered between the left and right eyes (±48px radius) resized to **96×96**, normalized.
3. **Nose / Boundary Region**: Crop centered on the nose landmark (±32px radius) resized to **64×64**, normalized.
4. **Temporal Motion Difference**: Triplet difference across consecutive full-face frames $(t, t+1, t+2)$:
   $$\text{diff}_1 = |f_{t+1} - f_t|, \quad \text{diff}_2 = |f_{t+2} - f_{t+1}|$$
   Concatenated along channels into a 6-channel tensor and interpolated to **6×112×112**.

### Stage 4: Multi-Stream Neural Feature Extraction (`phase3_model.py`)
- **EyeStream (11.34M params)**: Isolates high-frequency blending and blinking artifacts via a frozen Laplacian kernel and modified ResNet-18 with Squeeze-and-Excitation (SE) attention. Output: **256-d**.
- **NoseStream (1.04M params)**: Analyzes boundary seam distortions using a 4-stage convolutional network augmented by Convolutional Block Attention Module (CBAM). Output: **256-d**.
- **FullFaceHybridStream (14.61M params)**: Extracts structural and semantic face coherence by combining a ResNet-18 CNN patch extractor with a 4-layer Vision Transformer Encoder. Output: **512-d**.
- **TemporalStream (0.46M params)**: Identifies inter-frame motion jitter and temporal inconsistencies using 3 convolutional blocks and SE attention. Output: **256-d**.

### Stage 5: Multi-Modal Fusion Head (`phase3_model.py`)
- The 4 stream outputs are concatenated into a **1280-dimensional** representation vector:
  $$\mathbf{z} = [\mathbf{z}_{\text{eye}} \, (256) \parallel \mathbf{z}_{\text{nose}} \, (256) \parallel \mathbf{z}_{\text{face}} \, (512) \parallel \mathbf{z}_{\text{temporal}} \, (256)]$$
- The vector is processed through:
  $$\text{FC}(1280 \to 512) \to \text{BatchNorm1d}(512) \to \text{ReLU} \to \text{Dropout}(p=0.4) \to \text{FC}(512 \to 1)$$
- Produces a single raw scalar logit.

### Stage 6: Probability Scoring & Stream Attribution (`app.py`)
- **Logit to Probability**: Each window logit is passed through $\sigma(x) = \frac{1}{1 + e^{-x}}$ to compute window-level $P(\text{fake})$.
- **Temporal Aggregation**: The video-level verdict is computed as the arithmetic mean of $P(\text{fake})$ across all $N$ valid windows:
  $$\bar{P}(\text{fake}) = \frac{1}{N} \sum_{i=1}^N P_i(\text{fake})$$
- **Stream Attribution**: Evaluates stream importance by computing the mean absolute activation of each stream projected through the first fusion weight matrix:
  $$\text{Score}_s = \frac{1}{B \cdot d_{\text{hidden}}} \sum \left| \mathbf{W}_{s} \mathbf{z}_s^\top \right|$$
  Normalized to sum to 100% to provide explainable per-stream contribution metrics.
