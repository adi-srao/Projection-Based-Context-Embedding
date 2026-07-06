from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything.modeling.image_encoder import Block, PatchEmbed
from segment_anything.modeling.common import LayerNorm2d


class SAMProjectionEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 14,
        out_channels: int = 128,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        patch_size: int = 16,
        window_size: int = 14,
        global_attn_indexes: Tuple[int, ...] = (2, 5, 8, 11),
        pretrain_img_size: int = 1024,   # only sizes pos_embed / rel_pos params, NOT the runtime input
        use_abs_pos: bool = False,       # abs pos embed doesn't interpolate in stock SAM code -> off by default
    ):
        super().__init__()
        self.patch_size   = patch_size
        self.out_channels = out_channels
        self.depth        = depth

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_channels,
            embed_dim=embed_dim,
        )

        self.use_abs_pos = use_abs_pos
        self.pos_embed = None
        if use_abs_pos:
            g = pretrain_img_size // patch_size
            self.pos_embed = nn.Parameter(torch.zeros(1, g, g, embed_dim))

        grid = pretrain_img_size // patch_size
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True,
                use_rel_pos=True, rel_pos_zero_init=True,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(grid, grid),
            ) for i in range(depth)
        ])

        # 4 evenly spaced tap points through the depth (mirrors the 4 Res2Net stages)
        taps = sorted({max(0, (depth * (i + 1)) // 4 - 1) for i in range(4)})
        while len(taps) < 4:
            taps.append(depth - 1)
        self.tap_indices = sorted(set(taps))[:4]

        self.necks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, out_channels, kernel_size=1, bias=False),
                LayerNorm2d(out_channels),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                LayerNorm2d(out_channels),
            ) for _ in range(4)
        ])

    def forward(self, img):
        H, W = img.shape[-2], img.shape[-1]
        x = self.patch_embed(img)  # B, H/p, W/p, C

        if self.pos_embed is not None:
            if self.pos_embed.shape[1:3] != x.shape[1:3]:
                pe = self.pos_embed.permute(0, 3, 1, 2)
                pe = F.interpolate(pe, size=x.shape[1:3], mode="bicubic", align_corners=False)
                x = x + pe.permute(0, 2, 3, 1)
            else:
                x = x + self.pos_embed

        tap_set = set(self.tap_indices)
        taps = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in tap_set:
                taps.append(x)

        outs = []
        for feat, neck in zip(taps, self.necks):
            f = feat.permute(0, 3, 1, 2).contiguous()          # B, C, h, w  (h=H/patch_size)
            f = neck(f)
            f = F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)
            outs.append(f)
        return outs

    @staticmethod
    def query_context_features(feat_maps, pixel_coords, batch_idx):
        out = []
        for feat in feat_maps:
            _, _, H, W = feat.shape
            u = pixel_coords[:, 0].clamp(0, W - 1)
            v = pixel_coords[:, 1].clamp(0, H - 1)
            out.append(feat[batch_idx, :, v, u])
        return out


SAM_VIT_CONFIGS = {
    "vit_b": dict(embed_dim=768,  depth=12, num_heads=12, global_attn_indexes=(2, 5, 8, 11)),
    "vit_l": dict(embed_dim=1024, depth=24, num_heads=16, global_attn_indexes=(5, 11, 17, 23)),
    "vit_h": dict(embed_dim=1280, depth=32, num_heads=16, global_attn_indexes=(7, 15, 23, 31)),
}


def get_pretrained_sam_encoder(checkpoint_path: str, in_channels: int = 14, out_channels: int = 128,
                                variant: str = "vit_h"):
    """
    Loads an official SAM checkpoint (e.g. sam_vit_b_01ec64.pth from Meta's release)
    into SAMProjectionEncoder. patch_embed (channel count differs), the SAM neck,
    and pos_embed are intentionally NOT loaded -- they don't transfer to a
    different channel count / resolution. Transformer blocks (qkv/proj/mlp/rel_pos)
    do transfer and are loaded with strict=False.
    """
    cfg = SAM_VIT_CONFIGS[variant]
    model = SAMProjectionEncoder(in_channels=in_channels, out_channels=out_channels, **cfg)

    sd = torch.load(checkpoint_path, map_location="cpu")
    enc_sd = {k[len("image_encoder."):]: v for k, v in sd.items() if k.startswith("image_encoder.")}
    filtered = {
        k: v for k, v in enc_sd.items()
        if not k.startswith("patch_embed") and not k.startswith("neck") and k != "pos_embed"
    }
    model.load_state_dict(filtered, strict=False)
    return model