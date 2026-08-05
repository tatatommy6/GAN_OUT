from pathlib import Path
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image

#=======================================Settings=======================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data" / "processed"
CHECKPOINT_DIR = ROOT / "checkpoints"
SAMPLE_DIR = ROOT / "samples"

extensions = {".jpg", ".jpeg", ".png", ".webp"}


#=======================================Dataset=======================================
class CustomDataset(Dataset):
    def __init__(self,img_dir,img_size = 512):
        self.img_size = img_size

        self.img_paths = [path for path in img_dir.rglob("*")
                        if path.is_file() and path.suffix.lower() in extensions]
        
        if not self.img_paths:
            raise RuntimeError(f"이미지가 없습니다: {img_dir}")

        self.transform = transforms.Compose([
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness = 0.1,
                contrast = 0.1,
                saturation = 0.1,
                hue = 0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = (0.5,0.5,0.5),
                std = (0.5,0.5,0.5),
            ),
        ])
        print(f"train image count:{len(self.img_paths)}")

    def __len__(self):
        return len(self.img_paths)
    
    def create_outpainting_mask(self):
        # 이진 마스크 생성
        # 1: 원본 이미지가 보이는 영역
        # 0: generator 가 생성해야하는 영역
        size = self.img_size
        mask = torch.ones(1, size, size)

        min_extent = size // 8
        max_extent = size // 2

        directions = ["left", "right", "top", "bottom"]
        random.shuffle(directions)

        # 1~3개 방향 동시에 확장
        selected_directions = directions[:random.randint(1, 3)]

        for direction in selected_directions:
            extension = random.randint(min_extent, max_extent)

            if direction == "left":
                mask[:,:,:extension] = 0
            elif direction == "right":
                mask[:,:, size - extension:] = 0
            elif direction == "top":
                mask[:,:extension,:] = 0
            elif direction == "bottom":
                mask[:, size - extension:,:] = 0
        return mask

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        try:
            with Image.open(img_path) as pil_image:
                pil_image = pil_image.convert("RGB")
                image_tensor: torch.Tensor = self.transform(pil_image) # type: ignore
        except (OSError, ValueError):
            return self.__getitem__(random.randrange(len(self)))

        mask = self.create_outpainting_mask()

        masked_image = image_tensor * mask

        generator_input = torch.cat([masked_image, mask], dim = 0)

        return {
            "real": image_tensor,
            "masked": masked_image,
            "mask": mask,
            "generator_input": generator_input
        }
#=======================================Gated Convolution Layer=======================================
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
        output = self.output(d1)
        output = torch.tanh(self.output(output))

        return output

    
#=======================================Discriminator=======================================
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

#=======================================sub functions=======================================
