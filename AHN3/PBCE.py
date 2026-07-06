from typing import Tuple
import torch
import torch.nn as nn

from AHN3.ContextProjection import ContextProjection
from AHN3.utils import get_pretrained_res2net50_unet
from AHN3.ProjectionConvolution import Res2NetUNet
from AHN3.SparseConvolution import _3DEncoder, _make_linear
from AHN3.FocalDiceLoss import FocalDiceLoss

class EmbeddingDisentangling(nn.Module):
    def __init__(self, dim_3d, dim_2d, dim_out, dropout_prob=0.1):
        super().__init__()
        self.height_proj = _make_linear(1, dim_2d)
        self.fuse_2d     = _make_linear(dim_2d * 2, dim_2d)
        self.fuse_3d     = _make_linear(dim_2d + dim_3d, dim_out)
        self.dropout     = nn.Dropout(dropout_prob) 
        self.attn_gate   = nn.Linear(dim_out, dim_out)

    def forward(self, f3d, e2d, rel_h):
        h_feat  = self.height_proj(rel_h.unsqueeze(1))
        lifted  = self.fuse_2d(torch.cat([h_feat, e2d], dim=1))
        f_prime = self.fuse_3d(torch.cat([lifted, f3d], dim=1))
        f_prime = self.dropout(f_prime)
        return torch.sigmoid(self.attn_gate(f_prime)) * f_prime


class PCENet(nn.Module):
    def __init__(
        self,
        num_classes:   int,
        in_point_feat: int   = 13,
        image_size:    Tuple[int, int] = (128, 128),
        resolution:    float = 1.0,
        grid_size:     Tuple[int, int, int] = (16, 16, 16),
        pretrained_2d: bool  = True,
        lambda_scc:    float = 0.5,
        weights:       torch.Tensor = None,
        dropout_prob: float = 0.1,
        label_smoothing: float = 0.0,
        gamma: float = 2.0,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_scc  = lambda_scc
        self.dropout_prob = dropout_prob
        self.label_smoothing = label_smoothing

        self.context_proj = ContextProjection(image_size, resolution, in_channels=in_point_feat)
        
        if pretrained_2d:
            self.encoder_2d = get_pretrained_res2net50_unet(in_channels=in_point_feat, out_channels=128)
        else:
            self.encoder_2d = Res2NetUNet(in_channels=in_point_feat, out_channels=128)
            
        self.encoder_3d   = _3DEncoder(in_point_feat, grid_size)

        dim_2d   = self.encoder_2d.out_channels  
        dims_3d  = _3DEncoder.DIMS               
        self.ed_blocks = nn.ModuleList([
            EmbeddingDisentangling(dims_3d[i], dim_2d, dims_3d[i], dropout_prob=self.dropout_prob)
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

    def forward(self, p_local, p_context, bi_local, bi_context,
                batch_size, labels=None):
        B = batch_size
        img, pix_ctx,   _          = self.context_proj(p_context, bi_context, B)
        _,   pix_local, rel_height = self.context_proj(p_local,   bi_local,   B)

        E_2d  = self.encoder_2d(img)
        
        q_2d  = Res2NetUNet.query_context_features(E_2d, pix_local, bi_local)
        
        F_3d  = self.encoder_3d(p_local, p_local[:, :3], bi_local, B)

        fused = [self.ed_blocks[i](F_3d[i], q_2d[i], rel_height)
                 for i in range(4)]
        logits = self.classifier(torch.cat(fused, dim=1))

        out = {"logits": logits}

        if labels is not None:
            H, W = img.shape[-2], img.shape[-1]
            loss_seg  = self.seg_loss(logits, labels)
            e1_flat   = E_2d[0].permute(0,2,3,1).reshape(-1, E_2d[0].shape[1])
            scc_lgt   = self.ssc_head(e1_flat)
            soft_lbl  = self._soft_pixel_labels(labels, pix_local, bi_local, B, H, W)
            loss_scc  = self.scc_loss(scc_lgt, soft_lbl)
            
            out["loss"]     = loss_seg + self.lambda_scc * loss_scc
            out["loss_seg"] = loss_seg
            out["loss_scc"] = loss_scc
        return out