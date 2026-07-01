from typing import Dict, List
import numpy as np
import pandas as pd
import torch       
import os
from tqdm.auto import tqdm
from torch.amp import autocast
from Toronto3d.PointCloudDataset import PointCloudDataset

def calculate_class_weights(dataset, num_classes, clip_Max, clip_Min):
    counts = np.zeros(num_classes)
    
    num_files = len(dataset.csv_paths)
    
    for fi in range(num_files):
        pts, shm = dataset._get_tile(fi)
        try:
            labels = pts[:, -1].astype(int)
            unique, c = np.unique(labels, return_counts=True)
            for val, count in zip(unique, c):
                if 0 <= val < num_classes:
                    counts[val] += count
        finally:
            shm.close() 
    
    total_pts = np.sum(counts)
    weights = np.ones(num_classes, dtype=np.float32)
    for i in range(num_classes):
        if counts[i] > 0:
            weights[i] = total_pts / (num_classes * counts[i])
            
    weights = weights / np.min(weights)
    return torch.tensor(np.clip(weights, clip_Min, clip_Max), dtype=torch.float32)

from typing import List, Dict
import torch

def pce_collate(batch: List[Dict]) -> Dict:
    p_locals, p_ctxs, lbls = [], [], []
    bi_local_list, bi_ctx_list = [], []

    for i, sample in enumerate(batch):
        n = sample["p_local"].shape[0]
        m = sample["p_context"].shape[0]
        
        p_locals.append(sample["p_local"])
        p_ctxs.append(sample["p_context"])
        lbls.append(sample["labels"])
        
        bi_local_list.append(torch.full((n,), i, dtype=torch.long))
        bi_ctx_list.append(torch.full((m,), i, dtype=torch.long))

    window_centers = torch.stack([item["window_center"] for item in batch], dim=0)

    return {
        "p_local":       torch.cat(p_locals, dim=0),
        "p_context":     torch.cat(p_ctxs,   dim=0),
        "labels":        torch.cat(lbls,      dim=0),
        "bi_local":      torch.cat(bi_local_list, dim=0),
        "bi_context":    torch.cat(bi_ctx_list,   dim=0),
        "batch_size":    len(batch),
        "window_centers": window_centers,  
    }

def compute_iou(preds, labels, num_classes):
    """Returns per-class IoU as a list, ignoring index -1."""
    ious = []
    for c in range(num_classes):
        pred_c  = (preds == c)
        label_c = (labels == c)
        valid   = (labels != -1)
        inter   = (pred_c & label_c & valid).sum().item()
        union   = (( pred_c | label_c) & valid).sum().item()
        ious.append(inter / union if union > 0 else float("nan"))
    return ious

def train_one_epoch(model, loader, optimizer, device, num_classes, scaler, maxNorm):
    model.train()
    tot, seg, scc = 0.0, 0.0, 0.0
    all_preds, all_labels = [], []
    pbar = tqdm(loader, desc="Train", leave=True)
    
    accumulation_steps = 4  
    optimizer.zero_grad()

    for i, batch in enumerate(pbar):
        p_local        = batch["p_local"].to(device, non_blocking=True)
        p_context      = batch["p_context"].to(device, non_blocking=True)
        labels         = batch["labels"].to(device, non_blocking=True)
        bi_local       = batch["bi_local"].to(device)
        bi_ctx         = batch["bi_context"].to(device)
        window_centers = batch["window_centers"].to(device)
        B              = batch["batch_size"]

        with torch.autocast(device_type='cuda', enabled=True):
            out = model(
                p_local=p_local, 
                p_context=p_context, 
                bi_local=bi_local, 
                bi_context=bi_ctx,
                batch_size=B, 
                window_centers=window_centers, 
                labels=labels
            )
            loss = out["loss"] / accumulation_steps

        if torch.isnan(loss):
            print(f"\n[!] NaN detected at iteration {i}. Skipping batch.")
            continue

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=maxNorm)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        tot += out["loss"].item()
        seg += out["loss_seg"].item()
        scc += out["loss_scc"].item()

        preds = out["logits"].argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

        pbar.set_postfix(loss=f'{out["loss"].item():.3f}',
                         seg=f'{out["loss_seg"].item():.3f}')

    if len(loader) % accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=maxNorm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    n           = len(loader)
    all_preds   = torch.cat(all_preds)
    all_labels  = torch.cat(all_labels)
    iou_per_cls = compute_iou(all_preds, all_labels, num_classes)
    miou        = np.nanmean(iou_per_cls)
    acc         = (all_preds[all_labels != -1] == all_labels[all_labels != -1]).float().mean().item()

    return {
        "loss": tot/n, "loss_seg": seg/n, "loss_scc": scc/n,
        "miou": miou,  "acc": acc,   "iou_per_cls": iou_per_cls,
    }


@torch.no_grad()
def validate(model, loader, device, num_classes):
    model.eval()
    tot, seg, scc = 0.0, 0.0, 0.0
    all_preds, all_labels = [], []
    pbar = tqdm(loader, desc="Val  ", leave=False)

    for batch in pbar:
        p_local        = batch["p_local"].to(device)
        p_context      = batch["p_context"].to(device)
        bi_local       = batch["bi_local"].to(device)
        bi_ctx         = batch["bi_context"].to(device)
        window_centers = batch["window_centers"].to(device) 
        labels         = batch["labels"].to(device)
        B              = batch["batch_size"]

        with torch.autocast(device_type='cuda', enabled=True):
            out = model(
                p_local=p_local, 
                p_context=p_context, 
                bi_local=bi_local, 
                bi_context=bi_ctx,
                batch_size=B, 
                window_centers=window_centers,
                labels=labels
            )

        tot += out["loss"].item()
        seg += out["loss_seg"].item()
        scc += out["loss_scc"].item()

        preds = out["logits"].argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    n           = len(loader)
    all_preds   = torch.cat(all_preds)
    all_labels  = torch.cat(all_labels)
    iou_per_cls = compute_iou(all_preds, all_labels, num_classes)
    miou        = np.nanmean(iou_per_cls)
    acc         = (all_preds[all_labels != -1] == all_labels[all_labels != -1]).float().mean().item()

    return {
        "loss": tot/n, "loss_seg": seg/n, "loss_scc": scc/n, 
        "miou": miou,  "acc": acc,   "iou_per_cls": iou_per_cls,
    }


def build_window_sampler(dataset, minority_boost):
    weights = [float(minority_boost) if window[-1] else 1.0 for window in dataset._windows]        
    return torch.utils.data.WeightedRandomSampler(
        weights, 
        num_samples=len(weights), 
        replacement=True
    )

def count_frames_for_tile(csv_path: str, LOCAL_SIZE: float, CONTEXT_SIZE: float, MAX_LOCAL_PTS: int, 
                          MAX_CTX_PTS: int, STRIDE_RATIO: float, USE_UNCERTAINTY: bool, USE_GEOMETRY: bool, MINORITY_CLASSES: set[int]) -> int:
    ds = PointCloudDataset(
        [csv_path],
        local_size=LOCAL_SIZE,
        context_size=CONTEXT_SIZE,
        max_local=MAX_LOCAL_PTS,
        max_context=MAX_CTX_PTS,
        stride_ratio=STRIDE_RATIO,
        augment=False,
        use_uncertainty=USE_UNCERTAINTY,
        remove_minority=False,
        use_geometric_features=USE_GEOMETRY,
        minority_classes=MINORITY_CLASSES
    )
    n = len(ds)
    ds.cleanup()
    return n


def get_class_counts(csv_path: str, NUM_CLASSES: int) -> np.ndarray:
    df = pd.read_csv(csv_path)
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for cls, cnt in df["label"].value_counts().items():
        if 0 <= int(cls) < NUM_CLASSES:
            counts[int(cls)] += int(cnt)
    return counts


def compute_minority_classes(csv_paths: list[str], NUM_CLASSES: int, CLIP_MAX: float) -> set[int]:
    total_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for p in csv_paths:
        total_counts += get_class_counts(p, NUM_CLASSES)

    raw = total_counts.sum() / (NUM_CLASSES * np.maximum(total_counts, 1).astype(float))
    raw_norm = raw / raw.min()

    return {c for c in range(NUM_CLASSES) if raw_norm[c] >= CLIP_MAX}


def compute_display_weights(train_counts: np.ndarray, NUM_CLASSES: int, CLIP_MIN: float, CLIP_MAX: float) -> np.ndarray:
    total = train_counts.sum()
    w = np.ones(NUM_CLASSES, dtype=np.float32)
    for i in range(NUM_CLASSES):
        if train_counts[i] > 0:
            w[i] = total / (NUM_CLASSES * train_counts[i])
    w = w / w.min()
    return np.clip(w, CLIP_MIN, CLIP_MAX)


def print_fold_summary(
    fold_idx: int,
    train_paths: list[str],
    val_paths: list[str],
    NUM_CLASSES: int,
    MINORITY_CLASSES: set,
    CLIP_MIN: float,
    CLIP_MAX: float,
    MINORITY_BOOST: float,
    LOCAL_SIZE: float,
    CONTEXT_SIZE: float,
    MAX_LOCAL_PTS: int,
    MAX_CTX_PTS: int,
    STRIDE_RATIO: float,
    USE_UNCERTAINTY: bool,
    USE_GEOMETRY: bool,
) -> None:

    print(f"  FOLD {fold_idx}")

    print(f"  TRAINING TILES  ({len(train_paths)} tiles)")
    print(f"  {'Tile':<40} {'Frames':>8}")

    train_total_frames = 0
    train_total_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for p in train_paths:
        frames = count_frames_for_tile(p, LOCAL_SIZE, CONTEXT_SIZE, MAX_LOCAL_PTS, MAX_CTX_PTS, STRIDE_RATIO, USE_UNCERTAINTY, USE_GEOMETRY, MINORITY_CLASSES)
        counts = get_class_counts(p, NUM_CLASSES)
        train_total_frames += frames
        train_total_counts += counts
        print(f"  {os.path.basename(p):<40} {frames:>8,}")
    print(f"  {'TOTAL':<40} {train_total_frames:>8,}")

    print(f"\n  VALIDATION TILES  ({len(val_paths)} tiles)")
    print(f"  {'Tile':<40} {'Frames':>8}")

    val_total_frames = 0
    val_total_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for p in val_paths:
        frames = count_frames_for_tile(p, LOCAL_SIZE, CONTEXT_SIZE, MAX_LOCAL_PTS, MAX_CTX_PTS, STRIDE_RATIO, USE_UNCERTAINTY, USE_GEOMETRY, MINORITY_CLASSES)
        counts = get_class_counts(p, NUM_CLASSES)
        val_total_frames += frames
        val_total_counts += counts
        print(f"  {os.path.basename(p):<40} {frames:>8,}")
    print(f"  {'TOTAL':<40} {val_total_frames:>8,}")

    train_total_pts = train_total_counts.sum()
    val_total_pts   = val_total_counts.sum()

    print(f"\n  CLASS DISTRIBUTION")
    print(f"  {'#':<3} {'Train pts':>12} {'Train%':>7} "
          f"{'Val pts':>12} {'Val%':>7}  Minority?")

    for c in range(NUM_CLASSES):
        tr_pct = 100.0 * train_total_counts[c] / train_total_pts if train_total_pts > 0 else 0.0
        vl_pct = 100.0 * val_total_counts[c]   / val_total_pts   if val_total_pts   > 0 else 0.0
        flag   = " Yes" if c in MINORITY_CLASSES else ""
        print(f"  {c:<3}  "
              f"{train_total_counts[c]:>12,} {tr_pct:>6.2f}% "
              f"{val_total_counts[c]:>12,} {vl_pct:>6.2f}%"
              f"{flag}")

    weights = compute_display_weights(train_total_counts, NUM_CLASSES, CLIP_MIN, CLIP_MAX)

    print(f"\n  CLASS WEIGHTS  (CLIP_MAX={CLIP_MAX}, CLIP_MIN={CLIP_MIN})")
    print(f"  {'#':<3} {'Raw inv-freq':>13} {'Clipped wt':>11}  Capped?")
    print(f"  {'-'*3}  {'-'*13} {'-'*11}  {'-'*7}")

    raw = train_total_counts.sum() / (NUM_CLASSES * np.maximum(train_total_counts, 1).astype(float))
    raw_norm = raw / raw.min()
    for c in range(NUM_CLASSES):
        capped = "  Yes" if raw_norm[c] >= CLIP_MAX else ""
        print(f"  {c:<3}  "
              f"{raw_norm[c]:>13.2f}x {weights[c]:>11.4f}{capped}")

    minority_str = ", ".join(f"{c} " for c in sorted(MINORITY_CLASSES))
    print(f"\n  WINDOW SAMPLER  ({MINORITY_BOOST}x)")
    print(f"  Minority classes: {minority_str}")