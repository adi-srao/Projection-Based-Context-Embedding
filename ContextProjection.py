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

    def forward(self, p_context, batch_idx=None, batch_size=1):
        H, W = self.image_size
        r    = self.resolution
        C_in = self.in_channels
        
        if batch_idx is None:
            batch_idx = p_context.new_zeros(p_context.shape[0], dtype=torch.long)

        # Spatial coordinates for binning (assumed to be indices 0 and 1)
        x, y = p_context[:, 0], p_context[:, 1]
        u_all = torch.zeros_like(x, dtype=torch.long)
        v_all = torch.zeros_like(y, dtype=torch.long)

        # Calculate pixel coordinates per batch
        for b in range(batch_size):
            mask = batch_idx == b
            if not mask.any(): continue
            xb, yb = x[mask], y[mask]
            u_all[mask] = torch.floor((xb - xb.min()) / r).long().clamp(0, W - 1)
            v_all[mask] = torch.floor((yb - yb.min()) / r).long().clamp(0, H - 1)

        flat_idx  = batch_idx * (H * W) + v_all * W + u_all
        total_pix = batch_size * H * W

        # Aggregate ALL features into the BEV grid
        # Shape: (total_pix, C_in)
        sum_feats = p_context.new_zeros(total_pix, C_in).scatter_add_(
            0, flat_idx.unsqueeze(1).expand(-1, C_in), p_context
        )
        cnt = p_context.new_zeros(total_pix).scatter_add_(0, flat_idx, torch.ones_like(x))
        
        valid = cnt > 0
        avg_feats = torch.where(valid.unsqueeze(1), sum_feats / cnt.clamp(1).unsqueeze(1), torch.zeros_like(sum_feats))
        
        # Reshape to Image format: (B, C, H, W)
        image = avg_feats.view(batch_size, H, W, C_in).permute(0, 3, 1, 2).contiguous()

        # Metadata for the fusion branch
        pixel_coords = torch.stack([u_all, v_all], dim=1)            
        
        # Extract relative height (z is index 2) for the Disentangling block
        z_col = p_context[:, 2]
        min_z = p_context.new_full((total_pix,), 1e9)
        min_z.index_reduce_(0, flat_idx, z_col, reduce="amin", include_self=True)
        rel_height = z_col - min_z[flat_idx]
        
        return image, pixel_coords, rel_height