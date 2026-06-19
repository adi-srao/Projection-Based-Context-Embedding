import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class ResNet50UNet(nn.Module):
    def __init__(self, in_channels=15, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            backbone.bn1, backbone.relu
        )
        self.maxpool = backbone.maxpool

        if pretrained:
            with torch.no_grad():
                new_weight = backbone.conv1.weight.data.mean(1, keepdim=True).repeat(1, in_channels, 1, 1)
                self.stem[0].weight.data = new_weight / (in_channels / 3.0)

        self.layer1 = backbone.layer1 
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3 
        self.layer4 = backbone.layer4 

        self.up4 = DecoderBlock(2048, 1024, 512)
        self.up3 = DecoderBlock(512, 512, 256)
        self.up2 = DecoderBlock(256, 256, 128)
        self.up1 = DecoderBlock(128, 64, 64)

        OUT = 128
        self.proj1 = nn.Conv2d(64, OUT, 1)
        self.proj2 = nn.Conv2d(128, OUT, 1)
        self.proj3 = nn.Conv2d(256, OUT, 1)
        self.proj4 = nn.Conv2d(512, OUT, 1)
        self.out_channels = OUT

    def forward(self, img):
        H, W = img.shape[-2], img.shape[-1]
        
        # Encoder
        s0 = self.stem(img)       # 1/2 size, 64 ch
        s1 = self.layer1(self.maxpool(s0)) # 1/4 size, 256 ch
        s2 = self.layer2(s1)      # 1/8 size, 512 ch
        s3 = self.layer3(s2)      # 1/16 size, 1024 ch
        s4 = self.layer4(s3)      # 1/32 size, 2048 ch

        # Decoder with Skip Connections
        d4 = self.up4(s4, s3)     # 1/16 size, 512 ch
        d3 = self.up3(d4, s2)     # 1/8 size, 256 ch
        d2 = self.up2(d3, s1)     # 1/4 size, 128 ch
        d1 = self.up1(d2, s0)     # 1/2 size, 64 ch

        # Project and Upsample to original resolution for query
        up = lambda t, p: F.interpolate(p(t), (H, W), mode="bilinear", align_corners=False)
        
        return [
            up(d1, self.proj1), 
            up(d2, self.proj2),
            up(d3, self.proj3), 
            up(d4, self.proj4)
        ]

    @staticmethod
    def query_context_features(feat_maps, pixel_coords, batch_idx):
        """Remains unchanged for compatibility"""
        out = []
        for feat in feat_maps:
            _, _, H, W = feat.shape
            u = pixel_coords[:, 0].clamp(0, W - 1)
            v = pixel_coords[:, 1].clamp(0, H - 1)
            out.append(feat[batch_idx, :, v, u])
        return out