from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 1, dilation = 1):
        super().__init__()
        self.feature = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation = dilation)
        self.gate = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation = dilation)
        self.norm = nn.InstanceNorm2d(out_channels, affine = True)

    def forward(self, x):
        feature = self.feature(x)
        feature = self.norm(feature)
        gate = torch.sigmoid(self.gate(x))
        return F.leaky_relu(feature, 0.2) * gate

#=======================================Generator=======================================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        #Encoder
        self.enc1 = GatedConv2d(4, 64, 5, 2, 2)
        self.enc2 = GatedConv2d(64, 128, 3, 2, 1)
        self.enc3 = GatedConv2d(128, 256, 3, 2, 1)
        self.enc4 = GatedConv2d(256, 512, 3, 2, 1)

        self.bottleneck = nn.Sequential(
            GatedConv2d(512, 512, 3, 2, 1),
            GatedConv2d(512, 512, 3, 1, 2, dilation = 2),
            GatedConv2d(512, 512, 3, 1, 4, dilation = 4),
            GatedConv2d(512, 512, 3, 1, 8, dilation = 8),
            GatedConv2d(512, 512, 3, 1, 16, dilation = 16),
        )

        #decoder
        self.dec4 = GatedConv2d(1024, 512, 3)
        self.dec3 = GatedConv2d(768, 256, 3)
        self.dec2 = GatedConv2d(384, 128, 3)
        self.dec1 = GatedConv2d(192, 64, 3)

        self.output = nn.Conv2d(64, 3, kernel_size = 3, stride = 1, padding = 1)
    
    def upsample(self, x):
        return F.interpolate(x, scale_factor = 2, mode = "bilinear", align_corners = False)

    def forward(self, x):
        #encoder
        e1 = self.enc1(x) # 256 * 256
        e2 = self.enc2(e1) # 128 * 128
        e3 = self.enc3(e2) # 64 * 64
        e4 = self.enc4(e3) # 32 * 32

        b = self.bottleneck(e4) # 16 * 16

        #decoder 
        d4 = self.upsample(b)
        d4 = self.dec4(torch.cat([d4, e4], dim = 1))

        d3 = self.upsample(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim = 1))

        d2 = self.upsample(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim = 1))

        d1 = self.upsample(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim = 1))

        #output
        output = self.upsample(d1)
        output = torch.tanh(self.output(output))

        return output