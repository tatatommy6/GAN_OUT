from pathlib import Path
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data" / "processed"
CHECKPOINT_DIR = ROOT / "checkpoints"
SAMPLE_DIR = ROOT / "samples"

extensions = {".jpg", ".jpeg", ".png", ".webp"}

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