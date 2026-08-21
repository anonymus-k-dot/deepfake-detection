# VERITAS Model Architecture Specification

`DeepfakeMultiStreamModel` is a multi-stream deep neural network designed to capture heterogeneous spatial, frequency, and temporal artifacts characteristic of facial manipulation.

---

## 1. Parameter Summary

| Component | Class Name | Input Dimensions | Output Dimension | Total Parameters | Trainable Parameters |
|---|---|---|---|---|---|
| **Stream 1** | `EyeStream` | $(B, 3, 96, 96)$ | $(B, 256)$ | 11,335,003 | 11,334,976 |
| **Stream 2** | `NoseStream` | $(B, 3, 64, 64)$ | $(B, 256)$ | 1,035,682 | 1,035,682 |
| **Stream 3** | `FullFaceHybridStream` | $(B, 3, 224, 224)$ | $(B, 512)$ | 14,612,544 | 14,612,544 |
| **Stream 4** | `TemporalStream` | $(B, 6, 112, 112)$ | $(B, 256)$ | 462,336 | 462,336 |
| **Fusion Head**| `Sequential` | $(B, 1280)$ | $(B, 1)$ | 657,409 | 657,409 |
| **Total** | `DeepfakeMultiStreamModel` | *Multi-input* | $(B, 1)$ | **28,102,974** | **28,102,947** |

*Note: 27 parameters are non-trainable weights of the frozen 3×3 depthwise Laplacian high-pass filter in `EyeStream`.*

---

## 2. Attention Mechanisms

### Squeeze-and-Excitation (`SEBlock`)
Learns adaptive channel-wise feature recalibration:
1. **Global Descriptor**: Spatial average pooling $\mathbf{z} \in \mathbb{R}^C$.
2. **Bottleneck MLP**: $\mathbf{s} = \sigma\left(\mathbf{W}_2 \, \text{ReLU}(\mathbf{W}_1 \mathbf{z})\right)$ with reduction ratio $r = 16$.
3. **Channel Gating**: $\tilde{\mathbf{X}} = \mathbf{X} \odot \mathbf{s}$.
- Used in: `EyeStream` (after `layer2` and `layer4`) and `TemporalStream`.

### Convolutional Block Attention Module (`CBAMBlock`)
Combines sequential channel and spatial attention:
1. **Channel Attention**: Computes both average-pooled and max-pooled channel descriptors, passes both through a shared bottleneck MLP, sums them, and applies sigmoid activation.
2. **Spatial Attention**: Computes channel mean and channel max across the spatial grid, concatenates them into a 2-channel map, applies a $7\times 7$ convolution, and activates with sigmoid.
- Used in: `NoseStream`.

---

## 3. Individual Neural Streams

### Stream 1: EyeStream (`EyeStream`)
Focuses on pupil boundaries, unnatural reflections, and blending edges around the ocular region.
- **Input**: $(B, 3, 96, 96)$ eye crop.
- **Fixed Laplacian High-Pass Filter**: 3-channel depthwise conv ($3\times 3$, kernel $[[0, 1, 0], [1, -4, 1], [0, 1, 0]]$, frozen).
- **Modified ResNet-18 Backbone**:
  - `conv1`: $3\times 3$ kernel, stride 1, padding 1 (preserves spatial resolution).
  - `maxpool`: Replaced with `nn.Identity()`.
  - `layer1` ($64$-d), `layer2` ($128$-d) $\to$ `SEBlock(128)`.
  - `layer3` ($256$-d), `layer4` ($512$-d) $\to$ `SEBlock(512)`.
- **Global Pooling & Head**: `AdaptiveAvgPool2d(1)` $\to$ `Linear(512, 256)`. Output: **256-d**.

### Stream 2: NoseStream (`NoseStream`)
Detects mask boundary seams, blurring, and skin tone transitions across the central face.
- **Input**: $(B, 3, 64, 64)$ nose crop.
- **Convolutional Stages**:
  - Block 1: $\text{Conv}(3 \to 64, 3\times 3, s=1, p=1) \to \text{BN} \to \text{ReLU}$
  - Block 2: $\text{Conv}(64 \to 128, 3\times 3, s=2, p=1) \to \text{BN} \to \text{ReLU}$
  - Block 3: $\text{Conv}(128 \to 256, 3\times 3, s=2, p=1) \to \text{BN} \to \text{ReLU}$
  - Block 4: $\text{Conv}(256 \to 256, 3\times 3, s=1, p=2, \text{dilation}=2) \to \text{BN} \to \text{ReLU}$ (preserves $16\times 16$ grid)
- **CBAM Attention**: `CBAMBlock(256, reduction=16, spatial_kernel=7)`.
- **Global Pooling & Head**: `AdaptiveAvgPool2d(1)` $\to$ `Linear(256, 256)`. Output: **256-d**.

### Stream 3: FullFaceHybridStream (`FullFaceHybridStream`)
Combines CNN local feature extraction with Vision Transformer global context modeling.
- **Input**: $(B, 3, 224, 224)$ full face crop.
- **CNN Feature Extractor**: Pretrained ResNet-18 (layers 1 through 4) produces $(B, 512, 7, 7)$ feature map ($49$ spatial patches).
- **Patch Projection**: `Linear(512, 256)` + `LayerNorm(256)` projects 49 patches to $d_{\text{model}} = 256$.
- **Tokens & Positional Encoding**:
  - Learnable `[CLS]` token $\in \mathbb{R}^{1 \times 1 \times 256}$.
  - Learnable 1D positional embedding $\in \mathbb{R}^{1 \times 50 \times 256}$.
- **Transformer Encoder**:
  - 4 Transformer encoder layers.
  - Multi-Head Attention: 8 heads, $d_k = 32$.
  - Feedforward dimension: $1024$ ($4 \times d_{\text{model}}$).
  - Pre-LayerNorm (`norm_first=True`), GELU activation, dropout $0.1$.
- **Output Projection**: `[CLS]` token $\to$ `LayerNorm(256)` $\to$ `Linear(256, 512)`. Output: **512-d**.

### Stream 4: TemporalStream (`TemporalStream`)
Detects inter-frame motion jitter and blinking/expression discontinuities.
- **Input**: $(B, 6, 112, 112)$ frame difference tensor (trained from scratch).
- **Architecture**:
  - Block 1: $\text{Conv}(6 \to 64, 7\times 7, s=2, p=3) \to \text{BN} \to \text{ReLU}$
  - Block 2: $\text{Conv}(64 \to 128, 3\times 3, s=2, p=1) \to \text{BN} \to \text{ReLU}$
  - Block 3: $\text{Conv}(128 \to 256, 3\times 3, s=2, p=1) \to \text{BN} \to \text{ReLU}$
  - Channel Attention: `SEBlock(256, reduction=16)`.
- **Global Pooling & Head**: `AdaptiveAvgPool2d(1)` $\to$ `Linear(256, 256)`. Output: **256-d**.

---

## 4. Multi-Modal Fusion Head

The representation vectors are concatenated along feature dimensions:
$$\mathbf{z}_{\text{fused}} = [\mathbf{z}_{\text{eye}} \, (256) \parallel \mathbf{z}_{\text{nose}} \, (256) \parallel \mathbf{z}_{\text{face}} \, (512) \parallel \mathbf{z}_{\text{temporal}} \, (256)] \in \mathbb{R}^{B \times 1280}$$

The fusion classification network applies:
$$\mathbf{h} = \text{ReLU}\left(\text{BatchNorm1d}\left(\mathbf{W}_1 \mathbf{z}_{\text{fused}} + \mathbf{b}_1\right)\right), \quad \mathbf{W}_1 \in \mathbb{R}^{512 \times 1280}$$
$$\hat{y}_{\text{logit}} = \mathbf{W}_2 (\text{Dropout}_{0.4}(\mathbf{h})) + b_2, \quad \mathbf{W}_2 \in \mathbb{R}^{1 \times 512}$$
