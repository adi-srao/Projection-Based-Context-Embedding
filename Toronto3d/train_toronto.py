import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import csv
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from Toronto3d.PointCloudDataset import PointCloudDataset
from Toronto3d.PBCE import PCENet
from Toronto3d.utilsToronto import calculate_class_weights, train_one_epoch, validate, pce_collate, build_window_sampler, print_fold_summary, compute_minority_classes
from torch.amp import GradScaler
from sklearn.model_selection import KFold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Enabled processing of three orthogonal 2D views (XY, YZ, XZ) alongside the 3D grid, dynamically rastered based on tile ranges and padded to ensure that the 2D images 
are passed to the Projection Processing accurately while remaining agnostic of the model being used. All three view features and their relative depth/height/width coordinates 
during fusion to capture geometry regardless of MLS/ALS data.
"""

TRAIN_CSV_DIR = r"F:\Aditya\Tiles\Toronto Tiles\split_tiles"
SAVE_DIR      = r"F:\Aditya\Lidar Semantic Segmentation\PBCE\Toronto3d\current"
NUM_CLASSES   = 9
RESOLUTION    = 0.25
PRETRAINED_2D = True

LOCAL_SIZE    = 12.8
CONTEXT_SIZE  = 23.04
STRIDE_RATIO  = 0.25
MAX_LOCAL_PTS = 32768
MAX_CTX_PTS   = 65536
WORKERS       = 2
CLIP_MAX      = 20      
CLIP_MIN      = 0.1
USE_UNCERTAINTY = True
USE_GEOMETRY    = False
BASE_FEAT       = 12
REMOVE_MINORITY = False

if not USE_UNCERTAINTY:
    IN_POINT_FEAT = (BASE_FEAT - 1) + 2
else:
    IN_POINT_FEAT = BASE_FEAT + 2
if not USE_GEOMETRY:
    IN_POINT_FEAT -= 8

NUM_EPOCHS      = 50
BATCH_SIZE      = 2
INIT_LR         = 0.0001
WEIGHT_DECAY    = 0.01
LAMBDA_SCC      = 0.25
PATIENCE        = 7
DROPOUT_PROB    = 0.3
LABEL_SMOOTHING = 0.1
GAMMA           = 2.0  
ALPHA           = 0.50

MINORITY_BOOST   = 3.0   

def run_training(train_paths: list[str], val_paths: list[str], fold_name: str, minority_classes: set[int]) -> None:
    fold_dir = os.path.join(SAVE_DIR, fold_name)
    os.makedirs(fold_dir, exist_ok=True)

    train_ds = PointCloudDataset(
        train_paths,
        local_size=LOCAL_SIZE,
        context_size=CONTEXT_SIZE,
        max_local=MAX_LOCAL_PTS,
        max_context=MAX_CTX_PTS,
        stride_ratio=STRIDE_RATIO,
        augment=True,
        use_uncertainty=USE_UNCERTAINTY,
        remove_minority=False,
        use_geometric_features=USE_GEOMETRY,
        minority_classes=minority_classes
    )

    val_ds = PointCloudDataset(
        val_paths,
        local_size=LOCAL_SIZE,
        context_size=CONTEXT_SIZE,
        max_local=MAX_LOCAL_PTS,
        max_context=MAX_CTX_PTS,
        stride_ratio=STRIDE_RATIO,
        augment=False,
        use_uncertainty=USE_UNCERTAINTY,
        remove_minority=False,
        use_geometric_features=USE_GEOMETRY,
        minority_classes=minority_classes
    )

    
    sampler = build_window_sampler(train_ds, MINORITY_BOOST)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,          
        collate_fn=pce_collate,
        num_workers=WORKERS,
        pin_memory=True,
        persistent_workers=(WORKERS > 0),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=pce_collate,
        num_workers=WORKERS,
        pin_memory=True,
        persistent_workers=(WORKERS > 0),
    )

    class_weights = calculate_class_weights(train_ds, NUM_CLASSES, CLIP_MAX, CLIP_MIN).to(device)

    model = PCENet(
        num_classes=NUM_CLASSES,
        in_point_feat=IN_POINT_FEAT,
        resolution=RESOLUTION,
        lambda_scc=LAMBDA_SCC,
        weights=class_weights,
        dropout_prob=DROPOUT_PROB,
        label_smoothing=LABEL_SMOOTHING,
        gamma=GAMMA,
        alpha=ALPHA,
        local_size=LOCAL_SIZE,
        context_size=CONTEXT_SIZE,
        max_z_span=train_ds.tile_ranges[2] 
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=INIT_LR, weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=PATIENCE
    )

    scaler = GradScaler(device="cuda", enabled=True)
    best_val_miou     = float("-inf")
    epochs_no_improve = 0

    try:
        for epoch in range(1, NUM_EPOCHS + 1):
            tr = train_one_epoch(model, train_loader, optimizer, device, NUM_CLASSES, scaler, 0.5)
            vl = validate(model, val_loader, device, NUM_CLASSES)
            csv_path = os.path.join(fold_dir, "metrics.csv")

            tr_ious = {f"tr_iou_c{i}": tr["iou_per_cls"][i] for i in range(NUM_CLASSES)}
            vl_ious = {f"vl_iou_c{i}": vl["iou_per_cls"][i] for i in range(NUM_CLASSES)}

            header = (
                ["epoch", "tr_loss", "tr_miou", "tr_acc"]
                + list(tr_ious.keys())
                + ["vl_loss", "vl_miou", "vl_acc"]
                + list(vl_ious.keys())
            )

            row = (
                [epoch, tr["loss"], tr["miou"], tr["acc"]]
                + list(tr_ious.values())
                + [vl["loss"], vl["miou"], vl["acc"]]
                + list(vl_ious.values())
            )

            with open(csv_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                if epoch == 1:
                    writer.writerow(header)
                writer.writerow(row)
            vl_iou_str = " | ".join(f"c{i}={vl['iou_per_cls'][i]:.2f}" for i in range(NUM_CLASSES))

            print(f"[{fold_name}] Ep {epoch:>2} | Tr {tr['miou']:.3f} | Vl {vl['miou']:.3f} | Acc {vl['acc']:.3f} | {vl_iou_str}")
            scheduler.step(vl["loss"])

            if vl["miou"] > best_val_miou:
                best_val_miou     = vl["miou"]
                epochs_no_improve = 0
                torch.save(model.state_dict(), os.path.join(fold_dir, "best.pth"))
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}\n")
                break
    finally:
        train_ds.cleanup()
        val_ds.cleanup()


def main():
    filenames = [
        "features_L001_1.csv",
        "features_L001_2.csv",
        "features_L001_3.csv",
        "features_L001_4.csv",
        "features_L002_1.csv",
        "features_L002_2.csv",
        "features_L002_3.csv",
        "features_L002_4.csv",
    ]
    all_paths = []
    for fn in filenames:
        p = os.path.join(TRAIN_CSV_DIR, fn)
        if not os.path.exists(p):
            raise RuntimeError(f"Tile not found: {p}")
        all_paths.append(p)

    MINORITY_CLASSES = compute_minority_classes(all_paths, NUM_CLASSES, CLIP_MAX)

    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_paths), start=1):
        train_paths = [all_paths[i] for i in train_idx]
        val_paths   = [all_paths[i] for i in val_idx]
        fold_name   = f"fold_{fold_idx}"
        print_fold_summary(fold_idx, train_paths, val_paths, NUM_CLASSES, MINORITY_CLASSES,
                            CLIP_MIN, CLIP_MAX, MINORITY_BOOST, LOCAL_SIZE, CONTEXT_SIZE, MAX_LOCAL_PTS, 
                            MAX_CTX_PTS, STRIDE_RATIO, USE_UNCERTAINTY, USE_GEOMETRY)
        run_training(train_paths, val_paths, fold_name, MINORITY_CLASSES)
if __name__ == "__main__":
    main()