
import os
# import h5py
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader


# ── H5 key names ─────────────────────────────────────────────────────────────
H5_FULL_FACE = "full_face"
H5_EYE_CROP  = "eye_crop"
H5_NOSE_CROP = "nose_crop"
H5_TEMPORAL  = "temporal"


# =============================================================================
# SECTION 1 — ATTENTION MODULES
# =============================================================================

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention.

    Learns per-channel importance weights by:
        1. Global average-pooling to a (B, C) descriptor.
        2. Two FC layers with a bottleneck (reduction ratio r).
        3. Sigmoid gate applied back to each channel.

    Used in: EyeStream (after layer2 and layer4) and TemporalStream.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.avg_pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class CBAMBlock(nn.Module):
    """
    Full Convolutional Block Attention Module (channel + spatial).

    Applies two sequential attention gates:
        1. Channel attention  — WHAT features matter
        2. Spatial attention  — WHERE on the feature map the anomaly is

    Used in: NoseStream.
    """
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        reduced = max(channels // reduction, 8)

        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
        )

        # Spatial attention
        pad = spatial_kernel // 2
        self.spatial_conv = nn.Conv2d(
            2, 1, kernel_size=spatial_kernel, padding=pad, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape

        # Channel attention
        avg_c = self.avg_pool(x).view(b, c)
        max_c = self.max_pool(x).view(b, c)
        w_c = torch.sigmoid(
            self.channel_mlp(avg_c) + self.channel_mlp(max_c)
        ).view(b, c, 1, 1)
        x = x * w_c

        # Spatial attention
        avg_s = x.mean(dim=1, keepdim=True)
        max_s, _ = x.max(dim=1, keepdim=True)
        w_s = torch.sigmoid(
            self.spatial_conv(torch.cat([avg_s, max_s], dim=1))
        )
        x = x * w_s

        return x


# =============================================================================
# SECTION 2 — NEURAL STREAMS
# =============================================================================

class EyeStream(nn.Module):
    """
    Stream 1 — Texture and blinking artifacts (96×96 eye crops).

    Fixed Laplacian → Modified ResNet-18 (stride-1 first conv, no MaxPool)
    → SE attention after layer2 and layer4 → Global avg pool → FC 256-d.
    """
    def __init__(self, output_dim: int = 256):
        super().__init__()

        # Fixed Laplacian high-pass filter (frozen, groups=3 depthwise)
        self.laplacian = nn.Conv2d(
            3, 3, kernel_size=3, padding=1, bias=False, groups=3
        )
        kernel_2d = torch.tensor([
            [0.,  1., 0.],
            [1., -4., 1.],
            [0.,  1., 0.],
        ])
        laplacian_weight = kernel_2d.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self.laplacian.weight = nn.Parameter(laplacian_weight, requires_grad=False)

        # Modified ResNet-18 backbone
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(resnet.conv1.weight, mode="fan_out", nonlinearity="relu")
        resnet.maxpool = nn.Identity()

        self.stem   = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.se2    = SEBlock(128, reduction=16)
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.se4    = SEBlock(512, reduction=16)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(512, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.laplacian(x)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.se2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.se4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)                           # (B, 256)


class NoseStream(nn.Module):
    """
    Stream 2 — Structural distortion at blending boundary (64×64 nose crops).

    4 conv blocks (dilation in block 4 preserves 16×16 resolution)
    → Full CBAM → Global avg pool → FC 256-d.
    """
    def __init__(self, output_dim: int = 256):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3,  64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )

        self.cbam = CBAMBlock(256, reduction=16, spatial_kernel=7)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(256, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.cbam(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)                           # (B, 256)


class FullFaceHybridStream(nn.Module):
    """
    Stream 3 — Global symmetry and structural inconsistency (224×224 full face).

    ResNet-18 (all 4 layers) → 49 patch tokens → Linear projection
    → CLS token + positional embeddings → 4-layer Transformer encoder
    → CLS output → LayerNorm → FC 512-d.
    """
    def __init__(self, embed_dim: int = 256, num_heads: int = 8,
                 num_layers: int = 4, output_dim: int = 512):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # CNN backbone: full ResNet-18 → (B, 512, 7, 7)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])

        # Patch preparation
        self.patch_proj = nn.Linear(512, embed_dim)
        self.patch_norm = nn.LayerNorm(embed_dim)
        num_patches = 7 * 7   # 49 patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder (Pre-LN for stable training)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cls_norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        feat    = self.cnn(x)                                   # (B, 512, 7, 7)
        patches = feat.flatten(2).transpose(1, 2)               # (B, 49, 512)
        patches = self.patch_norm(self.patch_proj(patches))     # (B, 49, embed_dim)
        cls     = self.cls_token.expand(b, -1, -1)
        tokens  = torch.cat([cls, patches], dim=1) + self.pos_embed  # (B, 50, embed_dim)
        tokens  = self.transformer(tokens)
        cls_out = self.cls_norm(tokens[:, 0, :])
        return self.fc(cls_out)                                 # (B, 512)


class TemporalStream(nn.Module):
    """
    Stream 4 — Inter-frame motion jitter (6-channel temporal difference tensor).

    3 conv blocks (stride-2 each) → SE channel attention
    → Global avg pool → FC 256-d.
    No pretrained weights (6-channel input, trained from scratch).
    """
    def __init__(self, output_dim: int = 256):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.se   = SEBlock(256, reduction=16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(256, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.se(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)                           # (B, 256)


# =============================================================================
# SECTION 3 — FUSION HEAD + FULL MODEL
# =============================================================================

class DeepfakeMultiStreamModel(nn.Module):
    """
    Full multi-stream deepfake detection model.

    Fusion:
        [eye: 256] + [nose: 256] + [full_face: 512] + [temporal: 256]
        → concat → 1280-d
        → FC(1280→512) → BN → ReLU → Dropout(0.4) → FC(512→1)  raw logit

    At inference: torch.sigmoid(logit) ∈ [0, 1], where 1 = Fake.
    Label convention: Real = 0, Fake = 1.
    """
    CONCAT_DIM = 256 + 256 + 512 + 256   # = 1280

    def __init__(self, fusion_hidden: int = 512, dropout: float = 0.4):
        super().__init__()
        self.eye_stream       = EyeStream(output_dim=256)
        self.nose_stream      = NoseStream(output_dim=256)
        self.full_face_stream = FullFaceHybridStream(output_dim=512)
        self.temporal_stream  = TemporalStream(output_dim=256)

        self.fusion_head = nn.Sequential(
            nn.Linear(self.CONCAT_DIM, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(
        self,
        full_face: torch.Tensor,   # (B, 3, 224, 224)
        eye_crop:  torch.Tensor,   # (B, 3,  96,  96)
        nose_crop: torch.Tensor,   # (B, 3,  64,  64)
        temporal:  torch.Tensor,   # (B, 6,   H,   W)
    ) -> torch.Tensor:             # (B, 1) raw logit

        feat_eye  = self.eye_stream(eye_crop)
        feat_nose = self.nose_stream(nose_crop)
        feat_face = self.full_face_stream(full_face)
        feat_temp = self.temporal_stream(temporal)

        fused = torch.cat([feat_eye, feat_nose, feat_face, feat_temp], dim=1)
        return self.fusion_head(fused)

    # ── Training utilities ────────────────────────────────────────────────────

    def get_param_groups(self, lr_backbone=1e-5, lr_attention=1e-4, lr_head=3e-4):
        backbone_params  = []
        attention_params = []
        head_params      = []

        for name, p in self.eye_stream.named_parameters():
            if "laplacian" in name:
                pass   # frozen — skip
            elif "se" in name or "fc" in name:
                attention_params.append(p)
            else:
                backbone_params.append(p)

        for name, p in self.nose_stream.named_parameters():
            if "cbam" in name:
                attention_params.append(p)
            else:
                head_params.append(p)

        for name, p in self.full_face_stream.named_parameters():
            if "cnn" in name:
                backbone_params.append(p)
            elif any(k in name for k in ("transformer", "patch", "cls", "pos")):
                attention_params.append(p)
            else:
                head_params.append(p)

        for name, p in self.temporal_stream.named_parameters():
            if "se" in name:
                attention_params.append(p)
            else:
                head_params.append(p)

        head_params += list(self.fusion_head.parameters())

        return [
            {"params": backbone_params,  "lr": lr_backbone,  "name": "backbone"},
            {"params": attention_params, "lr": lr_attention, "name": "attention"},
            {"params": head_params,      "lr": lr_head,      "name": "head"},
        ]

    def freeze_backbones(self):
        for name, p in self.eye_stream.named_parameters():
            if "stem" in name or "layer" in name:
                p.requires_grad_(False)
        for p in self.full_face_stream.cnn.parameters():
            p.requires_grad_(False)
        print("[EyeStream + FullFaceStream CNN backbones frozen]")

    def unfreeze_backbones(self):
        for p in self.eye_stream.parameters():
            p.requires_grad_(True)
        for p in self.full_face_stream.cnn.parameters():
            p.requires_grad_(True)
        for p in self.eye_stream.laplacian.parameters():
            p.requires_grad_(False)   # Laplacian stays frozen forever
        print("[EyeStream + FullFaceStream CNN backbones unfrozen]")


# =============================================================================
# SECTION 4 — DATASET (per-video .h5 files)
# =============================================================================

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class DeepfakeH5Dataset(Dataset):
    """
    Dataset for per-video .h5 preprocessed files.

    Directory layout expected:
        split_dir/Real/*.h5   → label 0
        split_dir/Fake/*.h5   → label 1

    Each .h5 file contains N windows across four keys:
        H5_FULL_FACE, H5_EYE_CROP, H5_NOSE_CROP, H5_TEMPORAL
    """

    def __init__(self, split_dir: str, augment: bool = False):
        self.augment = augment
        self.samples = []   # (h5_path, label, window_idx)

        for label, cls_name in [(0, "Real"), (1, "Fake")]:
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.isdir(cls_dir):
                raise FileNotFoundError(f"Expected folder not found: {cls_dir}")
            h5_files = sorted(f for f in os.listdir(cls_dir) if f.endswith(".h5"))
            if not h5_files:
                raise FileNotFoundError(f"No .h5 files in {cls_dir}")

            for fname in h5_files:
                fpath = os.path.join(cls_dir, fname)
                try:
                    with h5py.File(fpath, "r") as hf:
                        for key in [H5_FULL_FACE, H5_EYE_CROP, H5_NOSE_CROP, H5_TEMPORAL]:
                            if key not in hf:
                                raise KeyError(
                                    f"Key '{key}' missing in {fpath}. "
                                    f"Available: {list(hf.keys())}"
                                )
                        n_windows = min(
                            hf[H5_FULL_FACE].shape[0],
                            hf[H5_EYE_CROP].shape[0],
                            hf[H5_NOSE_CROP].shape[0],
                            hf[H5_TEMPORAL].shape[0],
                        )
                        if n_windows == 0:
                            raise ValueError(f"All streams empty in {fpath}")
                    for w in range(n_windows):
                        self.samples.append((fpath, label, w))
                except Exception as e:
                    print(f"[WARNING] Skipping {fpath}: {e}")

        if not self.samples:
            raise RuntimeError(f"No valid samples found under {split_dir}")

        split_name = split_dir.split(os.sep)[-1]
        print(f"[Dataset] {split_name} | {len(self.samples)} windows | augment={augment}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        fpath, label, widx = self.samples[idx]

        with h5py.File(fpath, "r") as hf:
            full_face = hf[H5_FULL_FACE][widx]
            eye_crop  = hf[H5_EYE_CROP][widx]
            nose_crop = hf[H5_NOSE_CROP][widx]
            temporal  = hf[H5_TEMPORAL][widx]

        ff_t = self._spatial_to_tensor(full_face)
        ec_t = self._spatial_to_tensor(eye_crop)
        nc_t = self._spatial_to_tensor(nose_crop)
        tm_t = self._temporal_to_tensor(temporal)

        if self.augment:
            seed = torch.randint(0, 2**31, (1,)).item()
            ff_t = self._augment(ff_t, seed)
            ec_t = self._augment(ec_t, seed)
            nc_t = self._augment(nc_t, seed)
            tm_t = self._augment_temporal(tm_t, seed)

        return ff_t, ec_t, nc_t, tm_t, torch.tensor(label, dtype=torch.float32)

    # ── Tensor converters ─────────────────────────────────────────────────────

    @staticmethod
    def _spatial_to_tensor(arr: np.ndarray) -> torch.Tensor:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
        t = torch.from_numpy(arr)
        if t.dim() == 3 and t.shape[-1] in (1, 3):
            t = t.permute(2, 0, 1)
        t = (t - _IMAGENET_MEAN) / _IMAGENET_STD
        return t

    TEMPORAL_H = 112
    TEMPORAL_W = 112

    @staticmethod
    def _temporal_to_tensor(arr: np.ndarray) -> torch.Tensor:
        """
        Converts any temporal array layout → (6, 112, 112).

        Handles the (2W, 3, H) shape produced by the preprocessing pipeline
        (result of axis=-1 concat on CHW arrays then transpose (2,0,1)).
        """
        arr = np.asarray(arr, dtype=np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
        t = torch.from_numpy(arr.copy())

        if t.dim() == 3:
            c0, c1, c2 = t.shape
            if c0 == 6:
                pass
            elif c2 == 6:
                t = t.permute(2, 0, 1).contiguous()
            elif c0 % 2 == 0 and c2 == 3:
                H, W = c0 // 2, c1
                t = (t.reshape(2, H, W, 3).permute(0, 3, 1, 2)
                      .reshape(6, H, W).contiguous())
            elif c0 % 2 == 0 and c1 == 3:
                # (2W, 3, H) — produced by training preprocessing
                H, W = c0 // 2, c2
                t = (t.reshape(2, H, 3, W).permute(0, 2, 1, 3)
                      .reshape(6, H, W).contiguous())
            else:
                raise ValueError(f"Cannot interpret temporal shape {tuple(t.shape)}")
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

        if (t.shape[1] != DeepfakeH5Dataset.TEMPORAL_H
                or t.shape[2] != DeepfakeH5Dataset.TEMPORAL_W):
            t = F.interpolate(
                t.unsqueeze(0),
                size=(DeepfakeH5Dataset.TEMPORAL_H, DeepfakeH5Dataset.TEMPORAL_W),
                mode="bilinear", align_corners=False,
            ).squeeze(0)

        return t   # (6, 112, 112)

    @staticmethod
    def _augment(t: torch.Tensor, seed: int) -> torch.Tensor:
        gen = torch.Generator()
        gen.manual_seed(seed)
        if torch.rand(1, generator=gen).item() > 0.5:
            t = t.flip(-1)
        k = torch.randint(0, 4, (1,), generator=gen).item()
        if k > 0:
            t = torch.rot90(t, k, dims=[-2, -1])
        return t

    @staticmethod
    def _augment_temporal(t: torch.Tensor, seed: int) -> torch.Tensor:
        return DeepfakeH5Dataset._augment(t, seed)


# =============================================================================
# SECTION 5 — DATA LOADERS
# =============================================================================

def build_dataloaders(
    preprocessed_root: str,
    batch_size: int = 8,
    num_workers: int = 2,
) -> dict:
    """
    Returns {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}.
    Do NOT use 'test' until final Phase 6 evaluation.
    """
    train_ds = DeepfakeH5Dataset(os.path.join(preprocessed_root, "Training"),   augment=True)
    val_ds   = DeepfakeH5Dataset(os.path.join(preprocessed_root, "Validation"), augment=False)
    test_ds  = DeepfakeH5Dataset(os.path.join(preprocessed_root, "Testing"),    augment=False)

    kwargs = dict(num_workers=num_workers, pin_memory=True)
    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True, **kwargs),
        "val":   DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kwargs),
        "test":  DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kwargs),
    }


# =============================================================================
# SECTION 6 — VALIDATION CHECKS
# =============================================================================

def run_shape_validation(device: str = "cpu"):
    """
    Forward pass with synthetic data — call before any training to verify wiring.
    """
    print("=" * 60)
    print("Shape + gradient validation")
    print("=" * 60)

    model = DeepfakeMultiStreamModel().to(device)
    model.train()

    B = 2
    full_face  = torch.randn(B, 3, 224, 224).to(device)
    eye_crop   = torch.randn(B, 3,  96,  96).to(device)
    nose_crop  = torch.randn(B, 3,  64,  64).to(device)
    temporal   = torch.randn(B, 6, 112, 112).to(device)
    labels     = torch.zeros(B, 1).to(device)

    with torch.no_grad():
        fe = model.eye_stream(eye_crop)
        print(f"  EyeStream output:       {tuple(fe.shape)}  (expected: ({B}, 256))")
        fn = model.nose_stream(nose_crop)
        print(f"  NoseStream output:      {tuple(fn.shape)}  (expected: ({B}, 256))")
        ff = model.full_face_stream(full_face)
        print(f"  FullFaceStream output:  {tuple(ff.shape)}  (expected: ({B}, 512))")
        ft = model.temporal_stream(temporal)
        print(f"  TemporalStream output:  {tuple(ft.shape)}  (expected: ({B}, 256))")

    logits = model(full_face, eye_crop, nose_crop, temporal)
    print(f"  Full model output:      {tuple(logits.shape)}  (expected: ({B}, 1))")

    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, labels)
    print(f"  Loss (random init):     {loss.item():.4f}  (expect ~0.693)")
    loss.backward()

    fusion_fc_grad = model.fusion_head[0].weight.grad
    laplacian_grad = model.eye_stream.laplacian.weight.grad
    print(f"  Fusion FC gradient:     {'OK' if fusion_fc_grad is not None and fusion_fc_grad.abs().sum() > 0 else 'FAIL'}")
    print(f"  Laplacian gradient:     {'OK (None — frozen)' if laplacian_grad is None else 'FAIL'}")

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters:   {total:,}")
    print("=" * 60)
    print("Validation complete.")