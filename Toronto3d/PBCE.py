import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from ContextProjection import ContextProjection
from ProjectionConvolution import ResNet50UNet
from SparseConvolution import _3DEncoder, _make_linear
from FocalDiceLoss import FocalDiceLoss

class EmbeddingDisentangling(nn.Module):
    """
    Multi-Perspective Embedding Disentangling Block.
    Fuses 3D sparse features with context features queried from all three 
    orthogonal 2D planes, guided by a 3-axis relative position coordinate vector.
    """
    def __init__(self, dim_3d, dim_2d, dim_out, dropout_prob=0.1):
        super().__init__()
        self.height_proj = _make_linear(3, dim_2d) 
        self.fuse_2d     = _make_linear(dim_2d * 2, dim_2d)
        self.fuse_3d     = _make_linear(dim_2d + dim_3d, dim_out)
        self.dropout     = nn.Dropout(dropout_prob) 
        self.attn_gate   = nn.Linear(dim_out, dim_out)

    def forward(self, f3d, e2d, rel_pos_3d):
        h_feat  = self.height_proj(rel_pos_3d)
        lifted  = self.fuse_2d(torch.cat([h_feat, e2d], dim=1))
        f_prime = self.fuse_3d(torch.cat([lifted, f3d], dim=1))
        f_prime = self.dropout(f_prime)
        return torch.sigmoid(self.attn_gate(f_prime)) * f_prime


class PCENet(nn.Module):
    def __init__(
        self,
        num_classes:   int,
        in_point_feat: int   = 13,
        resolution:    float = 1.0,
        grid_size:     Tuple[int, int, int] = (16, 16, 16),
        pretrained_2d: bool  = True,
        lambda_scc:    float = 0.5,
        weights:       torch.Tensor = None,
        dropout_prob:  float = 0.1,
        label_smoothing: float = 0.0,
        gamma:         float = 2.0,
        alpha:         float = 0.5,
        tile_ranges:   Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_scc  = lambda_scc
        self.dropout_prob = dropout_prob
        self.label_smoothing = label_smoothing

        self.context_proj = ContextProjection(resolution, in_channels=in_point_feat, tile_ranges=tile_ranges)
        self.encoder_2d   = ResNet50UNet(in_channels=in_point_feat, pretrained=pretrained_2d)
        self.encoder_3d   = _3DEncoder(in_point_feat, grid_size)

        dim_2d   = self.encoder_2d.out_channels  
        dims_3d  = _3DEncoder.DIMS               
        
        self.ed_blocks = nn.ModuleList([
            EmbeddingDisentangling(dims_3d[i], dim_2d * 3, dims_3d[i], dropout_prob=self.dropout_prob)
            for i in range(4)
        ])

        total_dim = sum(dims_3d)   # 480
        self.classifier = nn.Sequential(
            _make_linear(total_dim, 256),
            nn.Dropout(0.5),          
            _make_linear(256, 128),  
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )
        self.ssc_head   = nn.Linear(dim_2d, num_classes)
        
        if weights is not None:
            self.seg_loss = FocalDiceLoss(
                num_classes=num_classes,
                gamma=gamma,
                alpha=alpha,
                weight=weights,
                ignore_index=-1
            )
            
        self.scc_loss   = nn.MultiLabelSoftMarginLoss()

    def _soft_pixel_labels(self, labels, pix_coords, bi, B, H, W):
        C    = self.num_classes
        soft = torch.zeros(B*H*W, C, device=labels.device)
        u    = pix_coords[:, 0].clamp(0, W-1)
        v    = pix_coords[:, 1].clamp(0, H-1)
        flat = bi*(H*W) + v*W + u
        ok   = (labels >= 0) & (labels < C)
        for c in range(C):
            m = ok & (labels == c)
            if m.any():
                soft[flat[m].unique(), c] = 1.0
        return soft

    def _pad_and_encode(self, img):
        """Zero-pads image tensors down right/bottom edges to the nearest multiple of 32."""
        raw_h, raw_w = img.shape[-2], img.shape[-1]
        max_dim = max(raw_h, raw_w)
        pad_target = ((max_dim - 1) // 32 + 1) * 32
        
        pad_h = pad_target - raw_h
        pad_w = pad_target - raw_w
        
        padded_img = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0.0)
        return self.encoder_2d(padded_img)

    def forward(self, p_local, p_context, bi_local, bi_context, batch_size, labels=None):
        B = batch_size
        
        proj_context = self.context_proj(p_context, bi_context, B)
        proj_local   = self.context_proj(p_local,   bi_local,   B)
        
        img_xy, _, _ = proj_context[0]
        img_yz, _, _ = proj_context[1]
        img_xz, _, _ = proj_context[2]
        
        _, pix_local_xy, rel_height  = proj_local[0]
        _, pix_local_yz, rel_depth_x = proj_local[1]
        _, pix_local_xz, rel_width_y = proj_local[2]

        rel_pos_3d = torch.stack([rel_height, rel_depth_x, rel_width_y], dim=1)

        E_2d_xy = self._pad_and_encode(img_xy)
        E_2d_yz = self._pad_and_encode(img_yz)
        E_2d_xz = self._pad_and_encode(img_xz)

        q_2d_xy = ResNet50UNet.query_context_features(E_2d_xy, pix_local_xy, bi_local)
        q_2d_yz = ResNet50UNet.query_context_features(E_2d_yz, pix_local_yz, bi_local)
        q_2d_xz = ResNet50UNet.query_context_features(E_2d_xz, pix_local_xz, bi_local)
        
        F_3d = self.encoder_3d(p_local, p_local[:, :3], bi_local, B)

        fused = []
        for i in range(4):
            q_2d_combined = torch.cat([q_2d_xy[i], q_2d_yz[i], q_2d_xz[i]], dim=1)
            fused.append(self.ed_blocks[i](F_3d[i], q_2d_combined, rel_pos_3d))
            
        logits = self.classifier(torch.cat(fused, dim=1))
        out = {"logits": logits}

        if labels is not None:
            loss_seg = self.seg_loss(logits, labels)
            
            def compute_view_scc(E_2d, pix_local):
                H, W = E_2d[0].shape[-2], E_2d[0].shape[-1]
                e_flat = E_2d[0].permute(0, 2, 3, 1).reshape(-1, E_2d[0].shape[1])
                scc_lgt = self.ssc_head(e_flat)
                soft_lbl = self._soft_pixel_labels(labels, pix_local, bi_local, B, H, W)
                return self.scc_loss(scc_lgt, soft_lbl)

            loss_scc_xy = compute_view_scc(E_2d_xy, pix_local_xy)
            loss_scc_yz = compute_view_scc(E_2d_yz, pix_local_yz)
            loss_scc_xz = compute_view_scc(E_2d_xz, pix_local_xz)
            
            # Combine losses across all three perspectives
            loss_scc = loss_scc_xy + loss_scc_yz + loss_scc_xz
            
            out["loss"]     = loss_seg + self.lambda_scc * loss_scc
            out["loss_seg"] = loss_seg
            out["loss_scc"] = loss_scc
            
        return out