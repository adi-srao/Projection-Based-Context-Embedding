import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from Res2Net.res2net import Bottle2neck, model_urls

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


class Res2NetUNet(nn.Module):
    def __init__(self, in_channels=15, out_channels=128, baseWidth=26, scale=4):
        super().__init__()
        self.inplanes = 64
        self.baseWidth = baseWidth
        self.scale = scale
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(Bottle2neck, 64, 3)              
        self.layer2 = self._make_layer(Bottle2neck, 128, 4, stride=2)   
        self.layer3 = self._make_layer(Bottle2neck, 256, 6, stride=2)   
        self.layer4 = self._make_layer(Bottle2neck, 512, 3, stride=2)   

        self.up4 = DecoderBlock(2048, 1024, 512)
        self.up3 = DecoderBlock(512, 512, 256)
        self.up2 = DecoderBlock(256, 256, 128)
        self.up1 = DecoderBlock(128, 64, 64)

        self.proj1 = nn.Conv2d(64, out_channels, 1)
        self.proj2 = nn.Conv2d(128, out_channels, 1)
        self.proj3 = nn.Conv2d(256, out_channels, 1)
        self.proj4 = nn.Conv2d(512, out_channels, 1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample, 
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, img):
        H, W = img.shape[-2], img.shape[-1]

        s0 = self.relu(self.bn1(self.conv1(img)))   
        s1 = self.layer1(self.maxpool(s0))          
        s2 = self.layer2(s1)                        
        s3 = self.layer3(s2)                        
        s4 = self.layer4(s3)                        

        d4 = self.up4(s4, s3)                       
        d3 = self.up3(d4, s2)                       
        d2 = self.up2(d3, s1)                       
        d1 = self.up1(d2, s0)                       

        up = lambda t, p: F.interpolate(p(t), (H, W), mode="bilinear", align_corners=False)
        
        return [
            up(d1, self.proj1), 
            up(d2, self.proj2),
            up(d3, self.proj3), 
            up(d4, self.proj4)
        ]

    @staticmethod
    def query_context_features(feat_maps, pixel_coords, batch_idx):
        out = []
        for feat in feat_maps:
            _, _, H, W = feat.shape
            u = pixel_coords[:, 0].clamp(0, W - 1)
            v = pixel_coords[:, 1].clamp(0, H - 1)
            out.append(feat[batch_idx, :, v, u])
        return out


def get_pretrained_res2net50_unet(in_channels=15, out_channels=128):
    model = Res2NetUNet(in_channels=in_channels, out_channels=out_channels, baseWidth=26, scale=4)
    
    pretrained_state = model_zoo.load_url(model_urls['res2net50_26w_4s'])
    
    filtered_state = {
        k: v for k, v in pretrained_state.items() 
        if not k.startswith('conv1') and not k.startswith('bn1') and not k.startswith('fc')
    }
    
    model.load_state_dict(filtered_state, strict=False)    
    return model