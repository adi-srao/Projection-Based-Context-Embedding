import glob
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from multiprocessing.shared_memory import SharedMemory

class PointCloudDataset(Dataset):
    def __init__(
        self,
        csv_paths,
        local_size:   float = 25.6,
        context_size: float = 128.0,
        max_local:    int   = 8192,
        max_context:  int   = 32768,
        stride_ratio: float = 0.5,
        augment:      bool  = False,
        grid_res:     float = 15.0,
        use_uncertainty: bool = True,
        remove_minority: bool = False,
        use_geometric_features: bool = True,
        minority_classes: set[int] = {3, 4, 5, 6, 7, 8},
    ):
        super().__init__()

        if isinstance(csv_paths, (str, Path)):
            csv_paths = sorted(glob.glob(str(csv_paths)))
        self.csv_paths    = [str(p) for p in csv_paths]
        self.local_size   = local_size
        self.context_size = context_size
        self.max_local    = max_local
        self.max_context  = max_context
        self.stride_ratio = stride_ratio
        self.augment      = augment
        self.grid_res     = grid_res
        self.use_uncertainty = use_uncertainty
        self.remove_minority = remove_minority
        self.use_geometric_features = use_geometric_features
        self._rng         = np.random.default_rng(42)

        self._shms:      List[SharedMemory] = []
        self._shapes:    List[Tuple] = []
        self._dtypes:    List[np.dtype] = []
        self._shm_names: List[str] = []

        self._z_floors:  List[float] = []
        self._windows:   List[Tuple[int, float, float, bool]] = []
        self._grids:     List[Dict[Tuple[int, int], np.ndarray]] = []

        self.minority_classes = minority_classes
        
        # Dynamic calculation baseline for vertical metrics
        max_z_span = 0.0

        for fi, path in enumerate(self.csv_paths):
            df   = pd.read_csv(path)

            if(not self.use_uncertainty): 
                df = df.drop(columns=['epistemic'], errors='ignore')
                if(not self.use_geometric_features):
                    df = df[['x', 'y', 'z', 'label']]
            else:
                if(not self.use_geometric_features):
                    df = df[['x', 'y', 'z', 'epistemic', 'label']]
        
            if(self.remove_minority):
                df = df[~df['label'].isin([3, 4])]

            cols = [c for c in df.columns if c != 'label']
            for c in ['z', 'y', 'x']:
                if c in cols:
                    cols.remove(c)
                    cols.insert(0, c)

            features  = df[cols].values.astype(np.float32)
            labels    = df['label'].values.astype(np.int64)
            z_floor   = float(df['z'].min())
            
            # Dynamic calculation track for vertical metrics envelope
            tile_z_span = float(df['z'].max() - z_floor)
            if tile_z_span > max_z_span:
                max_z_span = tile_z_span

            tile_data = np.hstack([features, labels.reshape(-1, 1)]).astype(np.float32)

            shm = SharedMemory(create=True, size=tile_data.nbytes)
            buf = np.ndarray(tile_data.shape, dtype=tile_data.dtype, buffer=shm.buf)
            buf[:] = tile_data

            self._shms.append(shm)
            self._shapes.append(tile_data.shape)
            self._dtypes.append(tile_data.dtype)
            self._shm_names.append(shm.name)
            self._z_floors.append(z_floor)

            grid = self._build_grid(tile_data)
            self._grids.append(grid)

            tile_windows = self._compute_windows_raw(tile_data, fi)

            for win in tile_windows:
                _, cx, cy = win
                l_idx, _ = self._sample_window(fi, cx, cy, tile_data)
                window_labels = labels[l_idx]
                has_minority = any(lbl in minority_classes for lbl in window_labels) if len(window_labels) > 0 else False
                self._windows.append((fi, cx, cy, has_minority))

        # Expose uniform tile ranges for multi-view projections and 3D sparse volume alignments
        self.tile_ranges = (self.context_size, self.context_size, float(np.ceil(max_z_span)))

    def _get_tile(self, fi: int) -> np.ndarray:
        shm = SharedMemory(name=self._shm_names[fi], create=False)
        arr = np.ndarray(self._shapes[fi], dtype=self._dtypes[fi], buffer=shm.buf)
        return arr, shm

    def _build_grid(self, pts: np.ndarray) -> Dict[Tuple[int, int], np.ndarray]:
        grid = {}
        xs, ys = pts[:, 0], pts[:, 1]
        gx = (xs / self.grid_res).astype(np.int32)
        gy = (ys / self.grid_res).astype(np.int32)
        keys = np.stack([gx, gy], axis=1)
        unique_keys, inverse_indices = np.unique(keys, axis=0, return_inverse=True)
        for i, key in enumerate(unique_keys):
            grid[tuple(key)] = np.where(inverse_indices == i)[0]
        return grid

    def _get_indices_from_grid(self, grid, cx, cy, size):
        hs = size / 2.0
        x_range = range(int((cx - hs) / self.grid_res), int((cx + hs) / self.grid_res) + 1)
        y_range = range(int((cy - hs) / self.grid_res), int((cy + hs) / self.grid_res) + 1)
        indices = [grid[(x, y)] for x in x_range for y in y_range if (x, y) in grid]
        return np.concatenate(indices) if indices else np.array([], dtype=np.int64)

    def _compute_windows_raw(self, pts: np.ndarray, file_idx: int) -> List[Tuple[int, float, float]]:
        stride = self.local_size * self.stride_ratio
        xs, ys = pts[:, 0], pts[:, 1]
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())
        windows = []
        cx = x_min + self.local_size / 2.0
        while cx < x_max:
            cy = y_min + self.local_size / 2.0
            while cy < y_max:
                if len(self._get_indices_from_grid(self._grids[file_idx], cx, cy, self.local_size)) > 0:
                    windows.append((file_idx, cx, cy))
                cy += stride
            cx += stride
        return windows

    def _sample_window(self, fi: int, cx: float, cy: float, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        grid = self._grids[fi]
        xs, ys = pts[:, 0], pts[:, 1]
        hs, hc = self.local_size / 2.0, self.context_size / 2.0

        l_cand = self._get_indices_from_grid(grid, cx, cy, self.local_size)
        c_cand = self._get_indices_from_grid(grid, cx, cy, self.context_size)

        l_mask = (xs[l_cand] >= cx - hs) & (xs[l_cand] < cx + hs) & \
                 (ys[l_cand] >= cy - hs) & (ys[l_cand] < cy + hs)
        c_mask = (xs[c_cand] >= cx - hc) & (xs[c_cand] < cx + hc) & \
                 (ys[c_cand] >= cy - hc) & (ys[c_cand] < cy + hc)

        l_idx = self._sample(l_cand[l_mask], self.max_local, self._rng)
        c_idx = self._sample(c_cand[c_mask], self.max_context, self._rng)
        return l_idx, c_idx

    @staticmethod
    def _sample(idx: np.ndarray, n: int, rng) -> np.ndarray:
        if len(idx) == 0:
            return np.zeros(n, dtype=np.int64)
        if len(idx) >= n:
            return idx[rng.integers(0, len(idx), size=n)]
        return rng.choice(idx, n, replace=True)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fi, cx, cy, _ = self._windows[idx]
        pts, shm_tile = self._get_tile(fi)

        try:
            l_idx, c_idx = self._sample_window(fi, cx, cy, pts)

            p_local   = pts[l_idx, :-1].copy()
            p_context = pts[c_idx, :-1].copy()
            labels    = pts[l_idx, -1].copy()
            z_floor   = self._z_floors[fi]
        finally:
            shm_tile.close()

        ctx_min = p_context[:, :3].min(axis=0)
        p_local[:, :2]   -= ctx_min[:2]
        p_context[:, :2] -= ctx_min[:2]

        local_z_min = p_local[:, 2].min()          
        hag_local   = (p_local[:, 2]   - local_z_min).reshape(-1, 1)
        hag_context = (p_context[:, 2] - local_z_min).reshape(-1, 1) 

        z_global_local   = (p_local[:, 2]   - z_floor).reshape(-1, 1)
        z_global_context = (p_context[:, 2] - z_floor).reshape(-1, 1)

        p_local[:, 2]   -= ctx_min[2]
        p_context[:, 2] -= ctx_min[2]

        p_local   = np.hstack([p_local,   hag_local,   z_global_local])
        p_context = np.hstack([p_context, hag_context, z_global_context])

        if self.augment:
            p_local   = self._augment(p_local,   self._rng)
            p_context = self._augment(p_context, self._rng)

        return {
            "p_local":   torch.from_numpy(p_local).float(),
            "p_context": torch.from_numpy(p_context).float(),
            "labels":    torch.from_numpy(labels).long(),
        }
    
    @staticmethod
    def _augment(pts: np.ndarray, rng) -> np.ndarray:
        pts   = pts.copy()
        angle = rng.uniform(0, 2 * np.pi)
        c, s  = np.cos(angle), np.sin(angle)
        pts[:, :2] = pts[:, :2] @ np.array([[c, s], [-s, c]], dtype=np.float32)
        if rng.random() > 0.5: pts[:, 0] = -pts[:, 0]
        scale = rng.uniform(0.95, 1.05)
        pts[:, :3] *= scale
        jitter = rng.normal(0, 0.01, size=(pts.shape[0], 3))
        pts[:, :3] += jitter
        return pts

    def cleanup(self):
        for shm in self._shms:
            try:
                shm.close()
                shm.unlink()
            except Exception: pass
        self._shms.clear()

    def __len__(self) -> int:
        return len(self._windows)

    def __del__(self):
        try: self.cleanup()
        except Exception: pass