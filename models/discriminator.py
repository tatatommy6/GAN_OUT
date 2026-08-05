import torch
import torch.nn as nn
#discriminator 는 이미지 전체를 true/fake 하나의 값으로 판단하지 않고, 여러 작은 영역별로 판단하는 patchGAN 구조임

def spectral_conv(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1):
    #기본적인 Conv2d 에 spectral nomalization을 적용하여 return 하는 함수
    #spectral nomalization: Discriminator 의 bias가 지나치게 커지는것을 제한함.
    return nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # 이미지 3채널 + 마스크 1채널 = 4채널 입력
        self.model = nn.Sequential(
            # [B,   4, 512, 512]
            # [B,  64, 256, 256]
            # [B, 128, 128, 128]
            # [B, 256,  64,  64]
            # [B, 512,  32,  32]
            # [B,   1,  31,  31]

            spectral_conv(4, 64),
            nn.LeakyReLU(0.2, inplace = True), #leakyReLU는 음수 영역에도 작은 기울기를 남김. ReLU는 음수를 전부 0으로 만듦.

            spectral_conv(64, 128),
            nn.LeakyReLU(0.2, inplace = True),

            spectral_conv(128, 256),
            nn.LeakyReLU(0.2, inplace = True),

            spectral_conv(256, 512),
            nn.LeakyReLU(0.2, inplace = True),

            spectral_conv(512, 1, stride = 1)
        )

    def forward(self, image, mask):
        # image의 shape: [B, 3, 512, 512]
        # mask의 shape:  [B, 1, 512, 512]
        # 채널 방향인 dim = 1로 합침
        #[B, 3, 512, 512]
        #       +
        #[B, 1, 512, 512]
        #       =
        #[B, 4, 512, 512]
        x = torch.cat([image, mask], dim = 1)
        return self.model(x)