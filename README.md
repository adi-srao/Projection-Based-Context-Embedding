# Projection-Based Context Embedding (PBCE)
### Uncertainty-Aware 3D ALS Point Cloud Semantic Segmentation

An implementation of Projection-Based Context Embedding for large-scale Airborne LiDAR Scanning (ALS) point cloud segmentation, extending the architecture proposed by [Dai et al.](https://ieeexplore.ieee.org/document/10506699) with several key architectural and engineering departures. Evaluated on the AHN3 and Toronto-3D datasets.

**Best result: 0.7106 mIoU on AHN3** (Ground, Vegetation, Buildings)

---

## Overview

PBCE segments large-scale ALS point clouds by combining a 2D Bird's-Eye View (BEV) projection branch with a sparse 3D voxel-convolution branch. The two streams are fused through an Embedding Disentangling (ED) module that maps projected points back into 3D space while preserving their 2D contextual features.

This repository diverges from the original paper in several significant ways:

- **Generalised BEV projection**: Rather than Dai et al.'s fixed four-channel elevation-statistics image (max/min/mean/std height), the implementation projects the full per-point feature vector into the BEV grid and aggregates via mean across all channels. This supports arbitrary feature configurations without architectural changes.
- **ResNet-50 U-Net encoder**: The original ResNet-34 FCN-style 2D branch is replaced with a full ResNet-50 U-Net (`ResNet50UNet` in `ProjectionConvolution.py`) with symmetric skip connections, approximately doubling representational capacity and preserving fine-grained spatial detail. The stem convolution accepts a variable `in_point_feat` channel count while retaining ImageNet-pretrained weights throughout the rest of the network.
- **Local relative height**: Height above ground is computed relative to the minimum z-value of all points within the same BEV pixel, rather than globally across the entire context region.
- **Epistemic uncertainty as input**: Uncertainty estimates are incorporated as point-level features, improving training stability and class retention.

---

## Pipeline

```
ALS Tile
   │
   ▼
Sliding Window Sampling  ──►  P_local (points in window)
                          └──►  P_context (surrounding region)
                                     │
                                     ▼
                              BEV Mean-Feature Projection
                                     │
                         ┌───────────┴────────────┐
                         ▼                        ▼
                  2D Branch                  3D Branch
             ResNet-50 U-Net          Sparse Voxel Encoder
                         │                        │
                         └───────────┬────────────┘
                                     ▼
                         Embedding Disentangling (ED)
                                     │
                                     ▼
                            Per-Point Class Labels
```

---

## Features

**Point features (AHN3, uncertainty enabled, geometric descriptors disabled):**
`x, y, z, epistemic_uncertainty, height_above_ground, z_global` → 6 channels

**With Weinmann-style geometric descriptors (jakteristics):**
Adds omnivariance, eigenentropy, anisotropy, planarity, linearity, surface variation, sphericity, verticality → 14 channels total

**Loss function:** FocalDiceLoss — a combination of focal loss (down-weighting easy examples) and multi-class Dice loss (overlap-based), with optional label smoothing. Class imbalance is handled via inverse-frequency class weighting.

**Training infrastructure:**
- Shared memory preloading via `multiprocessing.shared_memory`
- Precomputed sliding-window indices at dataset init
- Mixed-precision training (`torch.amp.GradScaler`)
- Gradient clipping (`clip_grad_norm_`, max_norm=1.0)
- Persistent DataLoader workers with `pin_memory` and `prefetch_factor=6`
- Gradient accumulation for Toronto-3D (VRAM-constrained)

---

## Results

### AHN3 (3 classes: Ground, Vegetation, Buildings)

| Configuration | mIoU |
|---|---|
| baseline_22-4 (initial) | 0.3904 |
| baseline_27-4 (best pure baseline) | 0.6935 |
| **uncertainty_aware_28-4 (final)** | **0.7106** |

Key findings:
- Reducing from 5 ASPRS classes to 3 dominant classes (Ground, Vegetation, Buildings) produced the single largest improvement, nearly doubling mIoU by eliminating unstable gradients from severely imbalanced minority classes.
- Increasing spatial resolution (grid 1.5→0.5, voxel 0.75→0.5) alongside the ResNet-50 U-Net introduction significantly improved class separability.
- The model was consistently underfitting rather than overfitting — dropout and label smoothing were removed, and the Dice loss weight (α) was reduced from 0.5 to 0.25.
- Integrating epistemic uncertainty as a feature made training significantly more stable and reduced class forgetting. Height-above-ground and z_global helped differentiate vegetation from ground/buildings but ground–building confusion persists; verticality as a feature is a suggested next step.

### Toronto-3D

Best validation mIoU: **0.1096** (tuning_4). Results were constrained by VRAM-forced batch size reduction, severe class imbalance across 9 semantic categories, and potential weakening of ImageNet spatial priors from modified IMAGE_SIZE. The dataset remains underfit; suggested improvements are larger batches, reduced regularisation, extended training schedules, and stronger imbalance mitigation.

---

## Repository Structure

```
.
├── AHN3/                    # AHN3 training pipeline
│   ├── train.py
│   ├── ProjectionConvolution.py   # ResNet-50 U-Net + PBCE module
│   ├── dataset.py                 # PointCloudDataset with shared memory
│   ├── losses.py                  # FocalDiceLoss
│   └── config.yaml                # Central runtime config
├── Toronto3d/               # Toronto-3D training pipeline
│   ├── train_toronto.py
│   └── ...
├── preprocessing.ipynb      # Point cloud preprocessing & geometric descriptor computation
└── requirements.txt
```

---

## Setup

**Requirements:** Python 3.10, CUDA 12.4, Windows (paths in requirements.txt are Windows-local; adapt as needed for Linux)

```bash
pip install -r requirements.txt
```

Key dependencies: `torch==2.6.0+cu124`, `spconv-cu124==2.3.8`, `jakteristics==0.6.2`, `laspy==2.7.0`, `scikit-learn`, `numpy`, `matplotlib`

> **Note:** `cumm` is referenced as a local file path in `requirements.txt`. You will need to install it separately from [its repository](https://github.com/FindDefinition/cumm) or adjust the path.

### Data

- **AHN3**: Available from the [AHN portal](https://www.ahn.nl/). Preprocessing (geometric descriptor computation, normalisation) is handled in `preprocessing.ipynb`.
- **Toronto-3D**: Available from the [Toronto-3D repository](https://github.com/WeikaiTan/Toronto-3D).

### Training

Configure `config.yaml`, then:

```bash
# AHN3
python AHN3/train.py

# Toronto-3D
python Toronto3d/train_toronto.py
```

---

## Reference

This work extends:

> H. Dai et al., *"Large-Scale ALS Point Cloud Segmentation via Projection-Based Context Embedding"*

Geometric descriptors computed using [jakteristics](https://github.com/jakarto3d/jakteristics) (Weinmann-style features).
