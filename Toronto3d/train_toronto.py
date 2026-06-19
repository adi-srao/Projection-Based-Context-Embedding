import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import csv
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from PointCloudDataset import PointCloudDataset
import yaml
import glob
from PBCE import PCENet
from utilsToronto import calculate_class_weights, train_one_epoch, validate, pce_collate
from torch.amp import GradScaler
from sklearn.model_selection import train_test_split, KFold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
Reduced context and local size, assuming that the model was leading to context/information overload and causing it to lose track of the signal.
Context and local points doubled to maximise the amount of information available within the smaller context/local regions.  
'''

# Path configuration
TRAIN_CSV_DIR = r"F:\Aditya\Tiles\Toronto Tiles\split_tiles"
SAVE_DIR      = r"F:\Aditya\Lidar Semantic Segmentation\PBCE\Toronto3d\current"
NUM_CLASSES   = 9
IMAGE_SIZE    = tuple([256, 256])
RESOLUTION    = 0.25
GRID_SIZE     = tuple([20, 256, 256]) 
PRETRAINED_2D = True

LOCAL_SIZE     = 6.4
CONTEXT_SIZE   = 16.0               
STRIDE_RATIO   = 0.25               
MAX_LOCAL_PTS  = 32768              
MAX_CTX_PTS    = 65536              
WORKERS        = 4
CLIP_MAX        = 4
CLIP_MIN        = 0.1

USE_UNCERTAINTY = True
USE_GEOMETRY   = True
BASE_FEAT      = 12
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
PATIENCE        = 20
DROPOUT_PROB    = 0.3
LABEL_SMOOTHING = 0.1
GAMMA           = 1.5
ALPHA           = 0.50

def prepare_split_data(tile_path, remove_labels=False):
    """Loads a single tile, removes minority labels, then performs a per-class 80/20 split."""
    data = pd.read_csv(tile_path)
    if remove_labels is not False and len(remove_labels) > 0:
        data = data[~data['label'].isin(remove_labels)]
        print(f"  Data rows after removal: {len(data)}")
    
    train_df, val_df = train_test_split(
        data, 
        test_size=0.20, 
        random_state=42, 
        stratify=data['label']
    )
    
    temp_train = tile_path.replace(".csv", "_train_split.csv")
    temp_val = tile_path.replace(".csv", "_val_split.csv")
    
    train_df.to_csv(temp_train, index=False)
    val_df.to_csv(temp_val, index=False)
    
    return [temp_train], [temp_val]

def run_training(train_files, val_files, fold_name, labels_to_remove=False):
    fold_dir = os.path.join(SAVE_DIR, fold_name)
    os.makedirs(fold_dir, exist_ok=True)

    if REMOVE_MINORITY:
        print(f"Removing least frequent classes in dataset: {labels_to_remove}")
    
    train_ds = PointCloudDataset(
        train_files, 
        local_size=LOCAL_SIZE, 
        context_size=CONTEXT_SIZE,
        max_local=MAX_LOCAL_PTS, 
        max_context=MAX_CTX_PTS,
        stride_ratio=STRIDE_RATIO, 
        augment=True,
        use_uncertainty=USE_UNCERTAINTY,
        remove_minority=False,
        use_geometric_features=USE_GEOMETRY,
    )
    
    val_ds = PointCloudDataset(
        val_files, 
        local_size=LOCAL_SIZE, 
        context_size=CONTEXT_SIZE,
        max_local=MAX_LOCAL_PTS, 
        max_context=MAX_CTX_PTS,                        
        stride_ratio=STRIDE_RATIO, 
        augment=False,
        use_uncertainty=USE_UNCERTAINTY,
        remove_minority=False,
        use_geometric_features=USE_GEOMETRY,
    )

    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        collate_fn=pce_collate, 
        num_workers=WORKERS, 
        pin_memory=True,
        persistent_workers=(WORKERS > 0)
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        collate_fn=pce_collate, 
        num_workers=WORKERS, 
        pin_memory=True,
        persistent_workers=(WORKERS > 0)
    )

    class_weights = calculate_class_weights(train_ds, NUM_CLASSES, CLIP_MAX, CLIP_MIN).to(device)
    print(f"Weights: {class_weights.tolist()}")
    
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
        label_smoothing=LABEL_SMOOTHING,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=PATIENCE)
    scaler = GradScaler(device='cuda', enabled=True)

    best_val_miou = float("-inf")
    epochs_no_improve = 0

    try:
        for epoch in range(1, NUM_EPOCHS + 1):
            tr = train_one_epoch(model, train_loader, optimizer, device, NUM_CLASSES, scaler, 0.5)
            vl = validate(model, val_loader, device, NUM_CLASSES)

            csv_path = os.path.join(fold_dir, "metrics.csv")
            tr_class_ious = {f"tr_iou_c{i}": tr['iou_per_cls'][i] for i in range(NUM_CLASSES)}
            vl_class_ious = {f"vl_iou_c{i}": vl['iou_per_cls'][i] for i in range(NUM_CLASSES)}
            
            header = ["epoch", "tr_loss", "tr_miou", "tr_acc"] + list(tr_class_ious.keys()) + ["vl_loss", "vl_miou", "vl_acc"] + list(vl_class_ious.keys())
            row = [epoch, tr['loss'], tr['miou'], tr['acc']] + list(tr_class_ious.values()) + [vl['loss'], vl['miou'], vl['acc']] + list(vl_class_ious.values())

            with open(csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                if epoch == 1: writer.writerow(header)
                writer.writerow(row)

            print(f"Ep {epoch:>2} | Val mIoU: {vl['miou']:.3f} | Val Acc: {vl['acc']:.3f}")
            
            scheduler.step(vl["loss"])
            if vl["miou"] > best_val_miou:
                best_val_miou = vl["miou"]
                epochs_no_improve = 0
                torch.save(model.state_dict(), os.path.join(fold_dir, "best.pth"))
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= PATIENCE:
                print("Early stopping triggered\n")
                break
    finally:
        train_ds.cleanup()
        val_ds.cleanup()
        for f in train_files + val_files:
            if ("_train_split.csv" in f or "_val_split.csv" in f) and os.path.exists(f):
                os.remove(f)

def main():
    filenames = ["features_L001_1.csv", "features_L001_2.csv", "features_L001_3.csv", "features_L001_4.csv", "features_L002_1.csv", "features_L002_2.csv", "features_L002_3.csv", "features_L003_4.csv"]


    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(filenames), start=1):
        train_filenames = [filenames[i] for i in train_idx]
        val_filenames = [filenames[i] for i in val_idx]
        fold_name = f"fold_{fold_idx}"

        print(f"Fold {fold_idx}: Training on: {train_filenames}")
        print(f"Fold {fold_idx}: Validating on: {val_filenames}")

        train_files = []
        val_files = []

        # Prepare train splits
        for filename in train_filenames:
            target_path = os.path.join(TRAIN_CSV_DIR, filename)
            if not os.path.exists(target_path):
                raise RuntimeError(f"{filename} not found in {TRAIN_CSV_DIR}.")

            split_train, _ = prepare_split_data(target_path, remove_labels=False)
            train_files.extend(split_train)

        # Validation files are used as-is
        for filename in val_filenames:
            val_target_path = os.path.join(TRAIN_CSV_DIR, filename)
            if not os.path.exists(val_target_path):
                raise RuntimeError(f"{filename} not found in {TRAIN_CSV_DIR}.")
            val_files.append(val_target_path)

        run_training(train_files, val_files, fold_name, labels_to_remove=False)

def get_least_frequent_classes(file_list, num_classes, k=5):
    counts = np.zeros(num_classes, dtype=np.int64)
    for p in file_list:
        df = pd.read_csv(p)
        labels = df['label'].values
        unique, c = np.unique(labels, return_counts=True)
        for val, cnt in zip(unique, c):
            if 0 <= val < num_classes:
                counts[int(val)] += int(cnt)
    return np.argsort(counts)[:k].tolist()

if __name__ == "__main__":
    main()