from typing import Tuple
import torch
import torch.nn as nn

class ContextProjection(nn.Module):
    """
    Projects 3D points into three orthogonal views centered around the window coordinates.
    Ensures absolute alignment between local and context points.
    """
    def __init__(self, resolution: float = 0.25, in_channels: int = 14, window_size: float = 23.04, max_z_span: float = 50.0):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.window_size = window_size
        
        self.W_xy = int(round(window_size / resolution))
        self.H_xy = int(round(window_size / resolution))
        
        self.W_yz = int(round(window_size / resolution))
        self.H_yz = int(round(max_z_span / resolution))
        
        self.W_xz = int(round(window_size / resolution))
        self.H_xz = int(round(max_z_span / resolution))

    def forward(self, p_context, batch_idx, batch_size, window_centers):
        C_in = self.in_channels
        x = p_context[:, 0]
        y = p_context[:, 1]
        z = p_context[:, 2]

        u_xy = torch.zeros_like(x, dtype=torch.long)
        v_xy = torch.zeros_like(y, dtype=torch.long)
        u_yz = torch.zeros_like(y, dtype=torch.long)
        v_yz = torch.zeros_like(z, dtype=torch.long)
        u_xz = torch.zeros_like(x, dtype=torch.long)
        v_xz = torch.zeros_like(z, dtype=torch.long)

        half_w = self.window_size / 2.0

        for b in range(batch_size):
            mask = batch_idx == b
            if not mask.any(): continue
            
            cx, cy = window_centers[b, 0], window_centers[b, 1]
            
            u_xy[mask] = torch.floor(((x[mask] + cx) - (cx - half_w)) / self.resolution).long().clamp(0, self.W_xy - 1)
            v_xy[mask] = torch.floor(((y[mask] + cy) - (cy - half_w)) / self.resolution).long().clamp(0, self.H_xy - 1)

            u_yz[mask] = torch.floor(((y[mask] + cy) - (cy - half_w)) / self.resolution).long().clamp(0, self.W_yz - 1)
            v_yz[mask] = torch.floor((z[mask]) / self.resolution).long().clamp(0, self.H_yz - 1)

            u_xz[mask] = torch.floor(((x[mask] + cx) - (cx - half_w)) / self.resolution).long().clamp(0, self.W_xz - 1)
            v_xz[mask] = torch.floor((z[mask]) / self.resolution).long().clamp(0, self.H_xz - 1)

        # XY Projection
        flat_idx_xy = batch_idx * (self.H_xy * self.W_xy) + v_xy * self.W_xy + u_xy
        total_pix_xy = batch_size * self.H_xy * self.W_xy
        sum_feats_xy = p_context.new_zeros(total_pix_xy, C_in).scatter_add_(0, flat_idx_xy.unsqueeze(1).expand(-1, C_in), p_context)
        cnt_xy = p_context.new_zeros(total_pix_xy).scatter_add_(0, flat_idx_xy, torch.ones_like(x))
        avg_feats_xy = torch.where(cnt_xy.unsqueeze(1) > 0, sum_feats_xy / cnt_xy.clamp(1).unsqueeze(1), torch.zeros_like(sum_feats_xy))
        image_xy = avg_feats_xy.view(batch_size, self.H_xy, self.W_xy, C_in).permute(0, 3, 1, 2).contiguous()
        
        min_z_xy = p_context.new_full((total_pix_xy,), 1e9)
        min_z_xy.index_reduce_(0, flat_idx_xy, z, reduce="amin", include_self=True)
        rel_height = z - min_z_xy[flat_idx_xy]

        # YZ Projection 
        flat_idx_yz = batch_idx * (self.H_yz * self.W_yz) + v_yz * self.W_yz + u_yz
        total_pix_yz = batch_size * self.H_yz * self.W_yz
        sum_feats_yz = p_context.new_zeros(total_pix_yz, C_in).scatter_add_(0, flat_idx_yz.unsqueeze(1).expand(-1, C_in), p_context)
        cnt_yz = p_context.new_zeros(total_pix_yz).scatter_add_(0, flat_idx_yz, torch.ones_like(y))
        avg_feats_yz = torch.where(cnt_yz.unsqueeze(1) > 0, sum_feats_yz / cnt_yz.clamp(1).unsqueeze(1), torch.zeros_like(sum_feats_yz))
        image_yz = avg_feats_yz.view(batch_size, self.H_yz, self.W_yz, C_in).permute(0, 3, 1, 2).contiguous()
        
        min_x_yz = p_context.new_full((total_pix_yz,), 1e9)
        min_x_yz.index_reduce_(0, flat_idx_yz, x, reduce="amin", include_self=True)
        rel_depth_x = x - min_x_yz[flat_idx_yz]

        # XZ Projection
        flat_idx_xz = batch_idx * (self.H_xz * self.W_xz) + v_xz * self.W_xz + u_xz
        total_pix_xz = batch_size * self.H_xz * self.W_xz
        sum_feats_xz = p_context.new_zeros(total_pix_xz, C_in).scatter_add_(0, flat_idx_xz.unsqueeze(1).expand(-1, C_in), p_context)
        cnt_xz = p_context.new_zeros(total_pix_xz).scatter_add_(0, flat_idx_xz, torch.ones_like(x))
        avg_feats_xz = torch.where(cnt_xz.unsqueeze(1) > 0, sum_feats_xz / cnt_xz.clamp(1).unsqueeze(1), torch.zeros_like(sum_feats_xz))
        image_xz = avg_feats_xz.view(batch_size, self.H_xz, self.W_xz, C_in).permute(0, 3, 1, 2).contiguous()
        
        min_y_xz = p_context.new_full((total_pix_xz,), 1e9)
        min_y_xz.index_reduce_(0, flat_idx_xz, y, reduce="amin", include_self=True)
        rel_width_y = y - min_y_xz[flat_idx_xz]

        return [
            [image_xy, torch.stack([u_xy, v_xy], dim=1), rel_height],
            [image_yz, torch.stack([u_yz, v_yz], dim=1), rel_depth_x],
            [image_xz, torch.stack([u_xz, v_xz], dim=1), rel_width_y]
        ]