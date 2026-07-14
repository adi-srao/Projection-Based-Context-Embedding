from typing import Tuple
import torch
import torch.nn as nn

class ContextProjection(nn.Module):
    """
    Projects P_context (M, 13) into a 28-channel BEV image.
    Aggregates all features (x, y, z, r, g, b, geometric features, etc.) 
    by averaging them within each grid cell.
    """
    def __init__(self, image_size: Tuple[int, int], resolution: float = 0.5, in_channels: int = 28):
        super().__init__()
        self.image_size = image_size
        self.resolution = resolution
        self.in_channels = in_channels

    @staticmethod
    def _project_coords(points: torch.Tensor, view: str) -> torch.Tensor:
        if view == "xy":
            return points[:, [0, 1]]
        if view == "yz":
            return points[:, [1, 2]]
        if view == "xz":
            return points[:, [0, 2]]
        raise ValueError(f"Unknown view: {view}")

    @staticmethod
    def _relative_depth(points: torch.Tensor, view: str) -> torch.Tensor:
        if view == "xy":
            return points[:, 2]
        if view == "yz":
            return points[:, 0]
        if view == "xz":
            return points[:, 1]
        raise ValueError(f"Unknown view: {view}")

    def forward(self, p_context, batch_idx=None, batch_size=1, proj_grids=None, view="xy"):
        H, W = self.image_size
        r    = self.resolution
        C_in = self.in_channels
        
        if batch_idx is None:
            batch_idx = p_context.new_zeros(p_context.shape[0], dtype=torch.long)

        coords = self._project_coords(p_context, view)
        u_all = torch.zeros(p_context.shape[0], dtype=torch.long, device=p_context.device)
        v_all = torch.zeros(p_context.shape[0], dtype=torch.long, device=p_context.device)

        for b in range(batch_size):
            mask = batch_idx == b
            if not mask.any():
                continue
            xy = coords[mask]
            if proj_grids is not None:
                grid_min = torch.as_tensor(proj_grids[b][view]["min"], device=p_context.device, dtype=p_context.dtype)
                u_all[mask] = torch.floor((xy[:, 0] - grid_min[0]) / r).long().clamp(0, W - 1)
                v_all[mask] = torch.floor((xy[:, 1] - grid_min[1]) / r).long().clamp(0, H - 1)
            else:
                u_all[mask] = torch.floor((xy[:, 0] - xy[:, 0].min()) / r).long().clamp(0, W - 1)
                v_all[mask] = torch.floor((xy[:, 1] - xy[:, 1].min()) / r).long().clamp(0, H - 1)

        flat_idx  = batch_idx * (H * W) + v_all * W + u_all
        total_pix = batch_size * H * W

        # Aggregate ALL features into the BEV grid
        # Shape: (total_pix, C_in)
        sum_feats = p_context.new_zeros(total_pix, C_in).scatter_add_(
            0, flat_idx.unsqueeze(1).expand(-1, C_in), p_context
        )
        cnt = p_context.new_zeros(total_pix).scatter_add_(0, flat_idx, torch.ones(p_context.shape[0], device=p_context.device))
        
        valid = cnt > 0
        avg_feats = torch.where(valid.unsqueeze(1), sum_feats / cnt.clamp(1).unsqueeze(1), torch.zeros_like(sum_feats))
        
        # Reshape to Image format: (B, C, H, W)
        image = avg_feats.view(batch_size, H, W, C_in).permute(0, 3, 1, 2).contiguous()

        # Metadata for the fusion branch
        pixel_coords = torch.stack([u_all, v_all], dim=1)            
        
        depth_col = self._relative_depth(p_context, view)
        min_depth = p_context.new_full((total_pix,), 1e9)
        min_depth.index_reduce_(0, flat_idx, depth_col, reduce="amin", include_self=True)
        rel_depth = depth_col - min_depth[flat_idx]
        
        return image, pixel_coords, rel_depth