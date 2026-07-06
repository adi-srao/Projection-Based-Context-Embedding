import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from Toronto3d.ContextProjection import ContextProjection
from Toronto3d.ProjectionConvolution import get_pretrained_sam_encoder, get_sam_encoder
from Toronto3d.SparseConvolution import _3DEncoder, _make_linear
from Toronto3d.FocalDiceLoss import FocalDiceLoss

class EmbeddingDisentangling(nn.Module):
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
        resolution:    float = 0.25,
        lambda_scc:    float = 0.5,
        weights:       torch.Tensor = None,
        dropout_prob:  float = 0.1,
        label_smoothing: float = 0.0,
        gamma:         float = 2.0,
        alpha:         float = 0.5,
        local_size:    float = 12.8,
        context_size:  float = 23.04,
        max_z_span:    float = 50.0,
        use_pretrained: bool = True,
        sam_checkpoint_path: str = None,
        sam_variant:   str = "vit_b",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_scc  = lambda_scc
        self.dropout_prob = dropout_prob

        self.context_proj = ContextProjection(resolution, in_channels=in_point_feat, window_size=context_size, max_z_span=max_z_span)
        self.local_proj   = ContextProjection(resolution, in_channels=in_point_feat, window_size=local_size, max_z_span=max_z_span)
        
        if use_pretrained and sam_checkpoint_path is not None:
            self.encoder_2d = get_pretrained_sam_encoder(
                sam_checkpoint_path, in_channels=in_point_feat, out_channels=128, variant=sam_variant
            )
            
        self.encoder_3d   = _3DEncoder(in_ch=in_point_feat, resolution=resolution, tile_ranges=(local_size, local_size, max_z_span))
        
        dim_2d   = self.encoder_2d.out_channels  # Tracks dynamically at 128 channels
        dims_3d  = _3DEncoder.DIMS               
        
        self.ed_blocks = nn.ModuleList([
            EmbeddingDisentangling(dims_3d[i], dim_2d * 3, dims_3d[i], dropout_prob=self.dropout_prob)
            for i in range(4)
        ])

        self.classifier = nn.Sequential(
            _make_linear(sum(dims_3d), 256),
            nn.Dropout(0.5),          
            _make_linear(256, 128),  
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )
        self.ssc_head   = nn.Linear(dim_2d, num_classes)
        
        if weights is not None:
            # FIXED: Activated label_smoothing by passing it directly to FocalDiceLoss
            self.seg_loss = FocalDiceLoss(
                num_classes=num_classes, 
                gamma=gamma, 
                alpha=alpha, 
                weight=weights, 
                ignore_index=-1,
                label_smoothing=label_smoothing
            )
        self.scc_loss   = nn.MultiLabelSoftMarginLoss()

    def _pad_tensor(self, img, pad_target):
        raw_h, raw_w = img.shape[-2], img.shape[-1]
        return F.pad(img, (0, pad_target - raw_w, 0, pad_target - raw_h), mode='constant', value=0.0)

    def forward(self, p_local, p_context, bi_local, bi_context, batch_size, window_centers, labels=None):
        B = batch_size
        
        proj_context = self.context_proj(p_context, bi_context, B, window_centers)
        proj_local   = self.local_proj(p_local,   bi_local,   B, window_centers)
        
        img_xy, _, _ = proj_context[0]
        img_yz, _, _ = proj_context[1]
        img_xz, _, _ = proj_context[2]
        
        _, pix_local_xy, rel_height  = proj_local[0]
        _, pix_local_yz, rel_depth_x = proj_local[1]
        _, pix_local_xz, rel_width_y = proj_local[2]

        rel_pos_3d = torch.stack([rel_height, rel_depth_x, rel_width_y], dim=1)

        max_dim = max(img_xy.shape[-2], img_xy.shape[-1], img_yz.shape[-2], img_yz.shape[-1], img_xz.shape[-2], img_xz.shape[-1])
        # Padded target must be a multiple of the encoder's patch_size (16 for SAM's ViT patch_embed,
        # was 32 for Res2Net's stride) or PatchEmbed silently truncates leftover pixels.
        unit = getattr(self.encoder_2d, "patch_size", 32)
        pad_target = ((max_dim - 1) // unit + 1) * unit

        # Pass zero-padded projections into our new multi-scale backbone
        E_2d_xy = self.encoder_2d(self._pad_tensor(img_xy, pad_target))
        E_2d_yz = self.encoder_2d(self._pad_tensor(img_yz, pad_target))
        E_2d_xz = self.encoder_2d(self._pad_tensor(img_xz, pad_target))

        # Call the query matching hooks straight out of the Res2NetUNet wrapper module definitions
        q_2d_xy = self.encoder_2d.query_context_features(E_2d_xy, pix_local_xy, bi_local)
        q_2d_yz = self.encoder_2d.query_context_features(E_2d_yz, pix_local_yz, bi_local)
        q_2d_xz = self.encoder_2d.query_context_features(E_2d_xz, pix_local_xz, bi_local)
        
        F_3d = self.encoder_3d(p_local, p_local[:, :3], bi_local, B)

        fused = []
        for i in range(4):
            q_2d_combined = torch.cat([q_2d_xy[i], q_2d_yz[i], q_2d_xz[i]], dim=1)
            fused.append(self.ed_blocks[i](F_3d[i], q_2d_combined, rel_pos_3d))
            
        logits = self.classifier(torch.cat(fused, dim=1))
        out = {"logits": logits}

        if labels is not None:
            out["loss_seg"] = self.seg_loss(logits, labels)
            
            # Semantic Consistency Constraint extraction loop
            def compute_view_scc(E_2d, pix_coords, W_orig, H_orig):
                H_pad, W_pad = E_2d[0].shape[-2], E_2d[0].shape[-1]
                e_flat = E_2d[0].permute(0, 2, 3, 1).reshape(-1, E_2d[0].shape[1])
                scc_lgt = self.ssc_head(e_flat)
                
                C = self.num_classes
                soft = torch.zeros(B * H_pad * W_pad, C, device=labels.device)
                u = pix_coords[:, 0].clamp(0, W_orig - 1)
                v = pix_coords[:, 1].clamp(0, H_orig - 1)
                flat = bi_local * (H_pad * W_pad) + v * W_pad + u
                
                ok = (labels >= 0) & (labels < C)
                for c in range(C):
                    m = ok & (labels == c)
                    if m.any():
                        soft[flat[m].unique(), c] = 1.0
                return self.scc_loss(scc_lgt, soft)

            loss_scc_xy = compute_view_scc(E_2d_xy, pix_local_xy, self.local_proj.W_xy, self.local_proj.H_xy)
            loss_scc_yz = compute_view_scc(E_2d_yz, pix_local_yz, self.local_proj.W_yz, self.local_proj.H_yz)
            loss_scc_xz = compute_view_scc(E_2d_xz, pix_local_xz, self.local_proj.W_xz, self.local_proj.H_xz)
            
            out["loss_scc"] = loss_scc_xy + loss_scc_yz + loss_scc_xz
            out["loss"]     = out["loss_seg"] + self.lambda_scc * out["loss_scc"]
            
        return out