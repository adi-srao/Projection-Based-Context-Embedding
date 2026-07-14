from typing import Dict, List
import numpy as np
import pandas as pd
import torch       
from tqdm.auto import tqdm
from torch.amp import autocast
from torch.utils import model_zoo

from AHN3.ProjectionConvolution import Res2NetUNet
from Res2Net.res2net import model_urls


def get_file_distribution(file_list):
    file_labels = []
    for path in file_list:
        df = pd.read_csv(path)
        counts = df['label'].value_counts().to_dict()
        dominant_class = max(counts, key=counts.get)
        file_labels.append(dominant_class)
    return np.array(file_labels)

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

def pce_collate(batch: List[Dict]) -> Dict:
    p_locals, p_ctxs, p_local_abs, p_ctx_abs, lbls = [], [], [], [], []
    bi_local_list, bi_ctx_list = [], []

    for i, sample in enumerate(batch):
        n = sample["p_local"].shape[0]
        m = sample["p_context"].shape[0]
        p_locals.append(sample["p_local"])
        p_ctxs.append(sample["p_context"])
        p_local_abs.append(sample["p_local_abs"])
        p_ctx_abs.append(sample["p_context_abs"])
        lbls.append(sample["labels"])
        bi_local_list.append(torch.full((n,), i, dtype=torch.long))
        bi_ctx_list.append(torch.full((m,), i, dtype=torch.long))

    return {
        "p_local":      torch.cat(p_locals,     dim=0),
        "p_context":    torch.cat(p_ctxs,       dim=0),
        "p_local_abs":  torch.cat(p_local_abs,  dim=0),
        "p_context_abs":torch.cat(p_ctx_abs,    dim=0),
        "labels":       torch.cat(lbls,         dim=0),
        "bi_local":     torch.cat(bi_local_list, dim=0),
        "bi_context":   torch.cat(bi_ctx_list,   dim=0),
        "proj_grids":   [sample["proj_grids"] for sample in batch],
        "batch_size":   len(batch),
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


def train_one_epoch(model, loader, optimizer, device, num_classes, scaler):
    model.train()
    tot, seg, scc = 0.0, 0.0, 0.0
    all_preds, all_labels = [], []
    pbar = tqdm(loader, desc="Train", leave=True)

    for batch in pbar:
        p_local   = batch["p_local"].to(device, non_blocking=True)
        p_context = batch["p_context"].to(device, non_blocking=True)
        p_local_abs = batch["p_local_abs"].to(device, non_blocking=True)
        p_context_abs = batch["p_context_abs"].to(device, non_blocking=True)
        labels    = batch["labels"].to(device, non_blocking=True)
        bi_local  = batch["bi_local"].to(device)
        bi_ctx    = batch["bi_context"].to(device)
        B         = batch["batch_size"]

        optimizer.zero_grad()

        with autocast(device_type='cuda', enabled=True):
            out = model(
                p_local, p_context,
                p_local_abs, p_context_abs,
                batch["proj_grids"],
                bi_local, bi_ctx,
                batch_size=B,
                labels=labels
            )
            loss = out["loss"]

        scaler.scale(loss).backward()
    
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()

        tot += out["loss"].item()
        seg += out["loss_seg"].item()
        scc += out["loss_scc"].item()

        preds = out["logits"].argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

        pbar.set_postfix(loss=f'{out["loss"].item():.3f}',
                         seg=f'{out["loss_seg"].item():.3f}')

    n           = len(loader)
    all_preds   = torch.cat(all_preds)
    all_labels  = torch.cat(all_labels)
    iou_per_cls = compute_iou(all_preds, all_labels, num_classes)
    miou        = np.nanmean(iou_per_cls)
    acc         = (all_preds[all_labels != -1] == all_labels[all_labels != -1]).float().mean().item()

    return {
        "loss": tot/n, "seg": seg/n, "scc": scc/n,
        "miou": miou,  "acc": acc,   "iou_per_cls": iou_per_cls,
    }


@torch.no_grad()
def validate(model, loader, device, num_classes):
    model.eval()
    tot, seg, scc = 0.0, 0.0, 0.0
    all_preds, all_labels = [], []
    pbar = tqdm(loader, desc="Val  ", leave=False)

    for batch in pbar:
        p_local   = batch["p_local"].to(device)
        p_context = batch["p_context"].to(device)
        p_local_abs = batch["p_local_abs"].to(device)
        p_context_abs = batch["p_context_abs"].to(device)
        bi_local  = batch["bi_local"].to(device)
        bi_ctx    = batch["bi_context"].to(device)
        labels    = batch["labels"].to(device)
        B         = batch["batch_size"]

        out = model(
            p_local, p_context,
            p_local_abs, p_context_abs,
            batch["proj_grids"],
            bi_local, bi_ctx,
            batch_size=B,
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
        "loss": tot/n, "seg": seg/n, "scc": scc/n,
        "miou": miou,  "acc": acc,   "iou_per_cls": iou_per_cls,
    }

@torch.no_grad()
def dump_class_masks(model, loader, device, num_classes, out_path):
    """
    Runs `model` over `loader` (expected shuffle=False, so batches walk the
    dataset's windows in order) and rasterizes, for every window, a boolean
    (num_classes, H, W) BEV mask of predicted classes using the same "xy"
    pixel projection the model already computes internally.

    All windows are packed into a single compressed .npz at `out_path`:
      - masks:      (N, num_classes, H, W) uint8, 1 where a predicted point
                     of that class fell into that pixel
      - window_ids: (N,) index into loader.dataset._windows
      - file_idx:   (N,) source csv/tile index for that window
      - cx, cy:     (N,) window center coordinates (for locating it later)
    """
    model.eval()
    dataset = loader.dataset

    all_masks, window_ids, file_idx_list, cx_list, cy_list = [], [], [], [], []
    global_idx = 0
    pbar = tqdm(loader, desc="Dump masks", leave=False)

    for batch in pbar:
        p_local       = batch["p_local"].to(device)
        p_context     = batch["p_context"].to(device)
        p_local_abs   = batch["p_local_abs"].to(device)
        p_context_abs = batch["p_context_abs"].to(device)
        bi_local      = batch["bi_local"].to(device)
        bi_ctx        = batch["bi_context"].to(device)
        B             = batch["batch_size"]

        out = model(
            p_local, p_context,
            p_local_abs, p_context_abs,
            batch["proj_grids"],
            bi_local, bi_ctx,
            batch_size=B,
            labels=None,
        )

        preds = out["logits"].argmax(dim=1)   # (num_local_points,)
        pix   = out["pix_local_xy"]           # (num_local_points, 2) -> (u, v)
        H, W  = out["H"], out["W"]
        bi_cpu = bi_local.cpu()

        for b in range(B):
            m = bi_cpu == b
            u = pix[m, 0].clamp(0, W - 1).cpu().numpy()
            v = pix[m, 1].clamp(0, H - 1).cpu().numpy()
            pc = preds[m].cpu().numpy()

            window_mask = np.zeros((num_classes, H, W), dtype=np.uint8)
            for c in range(num_classes):
                sel = pc == c
                if sel.any():
                    window_mask[c, v[sel], u[sel]] = 1

            all_masks.append(window_mask)

            fi, cx, cy = dataset._windows[global_idx]
            window_ids.append(global_idx)
            file_idx_list.append(fi)
            cx_list.append(cx)
            cy_list.append(cy)
            global_idx += 1

    if len(all_masks) == 0:
        print(f"No windows to dump for {out_path}, skipping.")
        return

    masks_arr = np.stack(all_masks, axis=0)  # (N, C, H, W)

    np.savez_compressed(
        out_path,
        masks=masks_arr,
        window_ids=np.array(window_ids, dtype=np.int64),
        file_idx=np.array(file_idx_list, dtype=np.int64),
        cx=np.array(cx_list, dtype=np.float32),
        cy=np.array(cy_list, dtype=np.float32),
    )
    print(f"Saved class-wise BEV masks {masks_arr.shape} -> {out_path}")


def get_pretrained_res2net50_unet(in_channels=15, out_channels=128):
    model = Res2NetUNet(in_channels=in_channels, out_channels=out_channels, baseWidth=26, scale=4)
    
    pretrained_state = model_zoo.load_url(model_urls['res2net50_26w_4s'])
    
    filtered_state = {
        k: v for k, v in pretrained_state.items() 
        if not k.startswith('conv1') and not k.startswith('bn1') and not k.startswith('fc')
    }
    
    model.load_state_dict(filtered_state, strict=False)    
    return model