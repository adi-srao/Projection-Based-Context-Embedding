import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import csv
import torch
from torch.utils.data import DataLoader
from PointCloudDataset import PointCloudDataset
import yaml
import glob
from AHN3.PBCE import PCENet
from AHN3.utils import get_file_distribution, calculate_class_weights, train_one_epoch, validate, pce_collate, dump_class_masks
from torch.amp import GradScaler
from sklearn.model_selection import StratifiedKFold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

TRAIN_CSV_DIR = cfg['paths']['train_csv_dir']
SAVE_DIR      = cfg['paths']['save_dir']
NUM_CLASSES   = cfg['model']['num_classes']
IMAGE_SIZE    = tuple(cfg['model']['image_size'])
RESOLUTION    = cfg['model']['resolution']
GRID_SIZE     = tuple(cfg['model']['grid_size'])
PRETRAINED_2D = cfg['model']['pretrained_2d']

LOCAL_SIZE     = cfg['dataset']['local_size']
CONTEXT_SIZE   = cfg['dataset']['context_size']
STRIDE_RATIO   = cfg['dataset']['stride_ratio']
MAX_LOCAL_PTS  = cfg['dataset']['max_local_pts']
MAX_CTX_PTS    = cfg['dataset']['max_ctx_pts']
WORKERS       = cfg['dataset']['workers']
CLIP_MAX       = cfg['dataset']['clip_max']
CLIP_MIN       = cfg['dataset']['clip_min']
USE_UNCERTAINTY = cfg['dataset']['use_uncertainty']
USE_GEOMETRY = cfg['dataset']['use_geometric_features']
BASE_FEAT  = cfg['dataset']['in_point_feat']
REMOVE_MINORITY = cfg['dataset']['remove_minority_classes']

if not USE_UNCERTAINTY:
    IN_POINT_FEAT = (BASE_FEAT - 1) + 2
else:
    IN_POINT_FEAT = BASE_FEAT + 2

if not USE_GEOMETRY:
    IN_POINT_FEAT -= 8  

if REMOVE_MINORITY:
    NUM_CLASSES -= 2

NUM_EPOCHS    = cfg['training']['num_epochs']
BATCH_SIZE    = cfg['training']['batch_size']
INIT_LR       = cfg['training']['init_lr']
WEIGHT_DECAY  = cfg['training']['weight_decay']
LAMBDA_SCC    = cfg['training']['lambda_scc']
NUM_FOLDS     = cfg['training']['num_folds']
PATIENCE      = cfg['training']['patience']
DROPOUT_PROB  = cfg['training']['dropout_prob']
LABEL_SMOOTHING = cfg['training']['label_smoothing']
GAMMA        = cfg['training']['gamma']
ALPHA        = cfg['training']['alpha']

def run_fold(fold_idx, train_files, val_files):
    fold_dir = os.path.join(SAVE_DIR, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)
    

    train_ds = PointCloudDataset(
        train_files, 
        local_size=LOCAL_SIZE, 
        context_size=CONTEXT_SIZE,
        max_local=MAX_LOCAL_PTS, 
        max_context=MAX_CTX_PTS,
        stride_ratio=STRIDE_RATIO, 
        augment=True,
        use_uncertainty=USE_UNCERTAINTY,
        remove_minority=REMOVE_MINORITY,
        use_geometric_features=USE_GEOMETRY,
    )
    
    val_ds   = PointCloudDataset(
        val_files, 
        local_size=LOCAL_SIZE, 
        context_size=CONTEXT_SIZE,
        max_local=MAX_LOCAL_PTS, 
        max_context=MAX_CTX_PTS,                        
        stride_ratio=STRIDE_RATIO, 
        augment=False,
        use_uncertainty=USE_UNCERTAINTY,
        remove_minority=REMOVE_MINORITY,
        use_geometric_features=USE_GEOMETRY,
    )


    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        collate_fn=pce_collate, 
        num_workers=WORKERS, 
        pin_memory=True,
        persistent_workers=(WORKERS > 0), 
        prefetch_factor=6 if WORKERS > 0 else None
    )
    
    val_loader   = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        collate_fn=pce_collate, 
        num_workers=WORKERS, 
        pin_memory=True,
        persistent_workers=(WORKERS > 0), 
        prefetch_factor=6 if WORKERS > 0 else None
    )

    class_weights = calculate_class_weights(train_ds, NUM_CLASSES, CLIP_MAX, CLIP_MIN).to(device)
    print(f"Fold {fold_idx} Weights: {class_weights.tolist()}")

    model = PCENet(
        num_classes=NUM_CLASSES, 
        in_point_feat=IN_POINT_FEAT, 
        image_size=IMAGE_SIZE,
        resolution=RESOLUTION, 
        grid_size=GRID_SIZE, 
        pretrained_2d=PRETRAINED_2D,
        lambda_scc=LAMBDA_SCC, 
        weights=class_weights,
        dropout_prob=DROPOUT_PROB,
        label_smoothing = LABEL_SMOOTHING,
        gamma = GAMMA,
        alpha = ALPHA
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=PATIENCE)
    scaler = GradScaler(device='cuda', enabled=True)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    try:
        for epoch in range(1, NUM_EPOCHS + 1):
            tr = train_one_epoch(model, train_loader, optimizer, device, NUM_CLASSES, scaler)
            vl = validate(model, val_loader, device, NUM_CLASSES)

            csv_path = os.path.join(fold_dir, "metrics.csv")
            tr_class_ious = {f"tr_iou_c{i}": tr['iou_per_cls'][i] for i in range(NUM_CLASSES)}
            vl_class_ious = {f"vl_iou_c{i}": vl['iou_per_cls'][i] for i in range(NUM_CLASSES)}
            
            header = ["epoch", "tr_loss", "tr_miou", "tr_acc"] + \
                     list(tr_class_ious.keys()) + \
                     ["vl_loss", "vl_miou", "vl_acc"] + \
                     list(vl_class_ious.keys())

            row = [epoch, tr['loss'], tr['miou'], tr['acc']] + \
                  list(tr_class_ious.values()) + \
                  [vl['loss'], vl['miou'], vl['acc']] + \
                  list(vl_class_ious.values())

            with open(csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                if epoch == 1:
                    writer.writerow(header)
                writer.writerow(row)

            tr_iou_str = ", ".join([f"{v:.3f}" for v in tr['iou_per_cls']])
            vl_iou_str = ", ".join([f"{v:.3f}" for v in vl['iou_per_cls']])
            
            print(f"Fold {fold_idx} | Ep {epoch:>2} | Val mIoU: {vl['miou']:.3f} | Val Acc: {vl['acc']:.3f}")
            print(f"   > Tr Class IoUs: [{tr_iou_str}]")
            print(f"   > Vl Class IoUs: [{vl_iou_str}]")

            scheduler.step(vl["loss"])
            
            if vl["loss"] < best_val_loss:
                best_val_loss = vl["loss"]
                epochs_no_improve = 0
                torch.save(model.state_dict(), os.path.join(fold_dir, "best.pth"))
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered for Fold {fold_idx}")
                break

        best_ckpt_path = os.path.join(fold_dir, "best.pth")
        if os.path.exists(best_ckpt_path):
            model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
            masks_path = os.path.join(fold_dir, "class_masks.npz")
            dump_class_masks(model, val_loader, device, NUM_CLASSES, masks_path)
        else:
            print(f"Fold {fold_idx}: no best.pth found, skipping class-mask dump.")
    finally:
        train_ds.cleanup()
        val_ds.cleanup()

def main():
    all_csv_files = sorted(glob.glob(os.path.join(TRAIN_CSV_DIR, "*.csv")))
    if not all_csv_files: 
        raise RuntimeError(f"No CSVs in {TRAIN_CSV_DIR}")

    file_representative_classes = get_file_distribution(all_csv_files)
    
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_csv_files, file_representative_classes)):
        print(f"\nFold {fold_idx + 1}/{NUM_FOLDS} | Stratified Split")
        print(f"Train: {len(train_idx)} files | Val: {len(val_idx)} files")
        
        train_files = [all_csv_files[i] for i in train_idx]
        val_files   = [all_csv_files[i] for i in val_idx]
        
        run_fold(fold_idx + 1, train_files, val_files)

if __name__ == "__main__":
    main()