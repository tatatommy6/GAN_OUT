import torch
import torch.nn as nn

def spectral_conv(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1):
    return nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # 이미지 3채널 + 마스크 1채널 = 4채널 입력
        self.model = nn.Sequential(
            spectral_conv(4, 64),
            nn.LeakyReLU(0.2, inplace = True),

            spectral_conv(64, 128),
            nn.LeakyReLU(0.2, inplace = True),

            spectral_conv(128, 256),
            nn.LeakyReLU(0.2, inplace = True),

            spectral_conv(256, 512),
            nn.LeakyReLU(0.2, inplace = True),

            spectral_conv(512, 1, stride = 1)
        )

    def forward(self, image, mask):
        x = torch.cat([image, mask], dim = 1)
        return self.model(x)
