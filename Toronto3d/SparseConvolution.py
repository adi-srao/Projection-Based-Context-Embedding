from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv


def _make_linear(in_ch: int, out_ch: int, bn: bool = True, relu: bool = True) -> nn.Sequential:
    layers: List[nn.Module] = [nn.Linear(in_ch, out_ch, bias=not bn)]
    if bn:
        layers.append(nn.BatchNorm1d(out_ch))
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

class SPVConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, resolution: float = 0.5, tile_ranges: Tuple[float, float, float] = (25.6, 25.6, 40.0)):
        super().__init__()
        self.resolution = resolution
        
        self.D = int(round(tile_ranges[0] / resolution)) # Depth (X)
        self.H = int(round(tile_ranges[1] / resolution)) # Height (Y)
        self.W = int(round(tile_ranges[2] / resolution)) # Width (Z)

        self.vox_conv = spconv.SparseSequential(
            spconv.SubMConv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False, indice_key="subm0"),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.mlp_pre  = _make_linear(in_ch  * 2, out_ch)
        self.mlp_post = _make_linear(out_ch * 2, out_ch)

    def _voxelise(self, coords_int, feats, B):
        D, H, W = self.grid_size
        return spconv.SparseConvTensor(feats, coords_int, [D, H, W], B)

    def _interp(self, vox_dense, coords_norm, bi, B):
        C   = vox_dense.shape[1]
        out = vox_dense.new_zeros(coords_norm.shape[0], C, dtype=vox_dense.dtype)
        
        for b in range(B):
            m = bi == b
            if not m.any():
                continue
            g = coords_norm[m][:, [2, 1, 0]].view(1, m.sum(), 1, 1, 3).to(vox_dense.dtype)
            v = vox_dense[b].unsqueeze(0)
            
            o = F.grid_sample(v, g, mode="bilinear",
                              padding_mode="border", align_corners=True)
            
            out[m] = o.squeeze(0).squeeze(-1).squeeze(-1).T
        return out

    def forward(self, point_feat, point_xyz, bi, B):
        vc = torch.zeros_like(point_xyz)
        for b in range(B):
            mask = bi == b
            if not mask.any(): continue
            pts_b = point_xyz[mask]
            vc[mask] = torch.floor((pts_b - pts_b.min(0).values) / self.resolution)

        ci = vc[:, 0].long().clamp(0, self.D - 1)
        hi = vc[:, 1].long().clamp(0, self.H - 1)
        wi = vc[:, 2].long().clamp(0, self.W - 1)
        coords_int = torch.stack([bi, ci, hi, wi], dim=1).int()

        sp_in  = spconv.SparseConvTensor(point_feat, coords_int, [self.D, self.H, self.W], B)
        sp_out = self.vox_conv(sp_in)

        vox_in_dense  = sp_in.dense()
        vox_out_dense = sp_out.dense()

        nc = vc.float().clone()
        nc[:, 0] = 2 * nc[:, 0] / max(self.D - 1, 1) - 1
        nc[:, 1] = 2 * nc[:, 1] / max(self.H - 1, 1) - 1
        nc[:, 2] = 2 * nc[:, 2] / max(self.W - 1, 1) - 1

        pre_pts  = self._interp(vox_in_dense,  nc, bi, B)
        post_pts = self._interp(vox_out_dense, nc, bi, B)

        mlp_mid = self.mlp_pre(torch.cat([point_feat, pre_pts],  dim=1))
        return    self.mlp_post(torch.cat([mlp_mid,   post_pts], dim=1))
    
class _3DEncoder(nn.Module):
    DIMS = [32, 64, 128, 256]

    def __init__(self, in_ch, grid_size):
        super().__init__()
        d = self.DIMS
        self.b1 = SPVConvBlock(in_ch,  d[0], grid_size)
        self.b2 = SPVConvBlock(d[0],   d[1], grid_size)
        self.b3 = SPVConvBlock(d[1],   d[2], grid_size)
        self.b4 = SPVConvBlock(d[2],   d[3], grid_size)
        self.d1 = _make_linear(d[0], d[0])
        self.d2 = _make_linear(d[1], d[1])
        self.d3 = _make_linear(d[2], d[2])

    def forward(self, feat, xyz, bi, B):
        f1 = self.b1(feat,        xyz, bi, B)
        f2 = self.b2(self.d1(f1), xyz, bi, B)
        f3 = self.b3(self.d2(f2), xyz, bi, B)
        f4 = self.b4(self.d3(f3), xyz, bi, B)
        return [f1, f2, f3, f4]