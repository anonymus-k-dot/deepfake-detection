# VERITAS Training Pipeline & Strategy

This document details the training procedures, loss function, differential optimization, backbone scheduling, and checkpointing for `DeepfakeMultiStreamModel`.

---

## 1. Objective Function

Binary Classification is framed with labels:
- **`0.0`** = Real (Authentic)
- **`1.0`** = Fake (Deepfake / Manipulated)

The network outputs a single unnormalized logit $\hat{y}_{\text{logit}} \in \mathbb{R}$, optimized using Binary Cross-Entropy with Logits:
$$\mathcal{L}_{\text{BCE}} = -\frac{1}{B} \sum_{i=1}^B \left[ y_i \log \sigma(\hat{y}_i) + (1 - y_i) \log (1 - \sigma(\hat{y}_i)) \right]$$
where $\sigma(z) = \frac{1}{1 + e^{-z}}$.

---

## 2. Differential Parameter Optimization

To prevent catastrophic forgetting in pretrained backbones while allowing new layers to learn rapidly, the parameters are partitioned into three differential learning rate groups (`model.get_param_groups()`):

| Parameter Group | Base Learning Rate | Included Components |
|---|---|---|
| **`backbone`** | `1e-5` | Modified ResNet-18 CNN layers in `EyeStream`, ResNet-18 CNN layers in `FullFaceHybridStream` |
| **`attention`** | `1e-4` | `SEBlock` in `EyeStream` & `TemporalStream`, `CBAMBlock` in `NoseStream`, Transformer Encoder & patch projection in `FullFaceHybridStream` |
| **`head`** | `3e-4` | FC layers of all 4 streams, Conv blocks of `NoseStream` and `TemporalStream`, full `fusion_head` |

*Note: The depthwise Laplacian filter in `EyeStream` (`eye_stream.laplacian`) is excluded from optimizer parameter groups and remains permanently frozen (`requires_grad=False`).*

---

## 3. Two-Phase Training Schedule

### Phase A: Warm-Up (Backbones Frozen)
```python
model = DeepfakeMultiStreamModel()
model.freeze_backbones()  # Disables gradients for ResNet-18 backbones
```
- Trains newly initialized attention modules, Transformer encoder, and fusion classifier layers while keeping pretrained convolutional features intact.

### Phase B: Fine-Tuning (Full Unfreeze)
```python
model.unfreeze_backbones()  # Enables gradients for ResNet backbones (Laplacian stays frozen)
```
- Trains all streams end-to-end with the differential learning rates defined above.

---

## 4. Hyperparameters & Settings

| Hyperparameter | Value | Description |
|---|---|---|
| **Optimizer** | Adam / AdamW | $\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$, `weight_decay = 0.0001` |
| **Batch Size** | `8` | Window-level mini-batch size |
| **Data Workers** | `2` | DataLoader worker processes with `pin_memory=True` |
| **Training Epochs** | $40+$ | Best validation checkpoint achieved at epoch 42 |
| **Scheduler** | StepLR / LambdaLR | Decays learning rates progressively across epochs |
| **Best Model Selection** | Validation ROC-AUC | Checkpoint saved when validation AUC reaches new maximum |

---

## 5. Checkpoint Format

Saved checkpoints (`best_model.pt`) contain:
```python
{
    "epoch": 42,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "scheduler_state": scheduler.state_dict(),
    "val_auc": 0.880473,
    "val_f1": 0.780632,
    "val_acc": 0.783821
}
```

---

## 6. Pre-Flight Model Verification

Before initiating training, execute the built-in shape and gradient verification:
```python
from phase3_model import run_shape_validation
run_shape_validation(device="cuda" if torch.cuda.is_available() else "cpu")
```
This performs a synthetic forward-backward pass, verifying:
- Output dimensions for all 4 streams ($(B, 256)$, $(B, 256)$, $(B, 512)$, $(B, 256)$)
- Fusion logit shape $(B, 1)$
- Gradient flow to fusion weights
- Non-trainable status of the frozen Laplacian kernel
