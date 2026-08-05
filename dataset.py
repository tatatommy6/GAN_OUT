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

class CustomDataset(Dataset): # 저장된 완성 이미지에서 매번 무작위로 일부 영역을 가려서 -> 원래 완성된 이미지 학습 쌍을 만듦
    def __init__(self,img_dir,img_size = 512):
        self.img_size = img_size

        self.img_paths = [path for path in img_dir.rglob("*") #data/processed 아래를 재귀적으로 검색하고 이미지 경로 저장 :rglob()
                        if path.is_file() and path.suffix.lower() in extensions] #확장자명을 소문자로 바꿔서 extensions 안에 있는지 확인
        
        if not self.img_paths:
            raise RuntimeError(f"이미지가 없습니다: {img_dir}")

        self.transform = transforms.Compose([
            transforms.RandomCrop(img_size), # 이미지를 랜덤 위치에서 512 * 512 로 크롭
            transforms.RandomHorizontalFlip(), # 50%확률로 상하반전
            transforms.ColorJitter( # 밝기 채도 대비를 조금씩 변경
                brightness = 0.1,
                contrast = 0.1,
                saturation = 0.1,
                hue = 0.02,
            ),
            transforms.ToTensor(), #PIL 형식의 이미지를 tensor로 변환
            transforms.Normalize( # generator 마지막 출력이 tanh라서 출력 범위도 [-1, 1]이기 때문에 Normalize를 함
                mean = (0.5,0.5,0.5),
                std = (0.5,0.5,0.5),
            ),
        ])
        print(f"train image count:{len(self.img_paths)}")

    def __len__(self): #데이터셋에 들어있는 전체 이미지 수를 return
        return len(self.img_paths) #이 값을 이용해 한 epoch에 몇장을 학습할지 결정
    
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

        # 1~3개 방향 동시에 확장. 따라서 호출할 때마다 마스크 방향과 크기가 달라짐
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

        mask = self.create_outpainting_mask() #무작위 마스크 생성

        masked_image = image_tensor * mask # 이미지에 마스크를 곱하여 생성 영역을 검은색으로 가림
        # 마스크가 1인곳은 이미지 유지, 0인 곳은 픽셀이 0. 따라서 검은색이 됨

        # generator가 검은색 자체를 원본 내용으로 오해하지 않도록 마스크도 입력으로 전달.
        generator_input = torch.cat([masked_image, mask], dim = 0)

        return { # 필요한 값을 딕셔너리로 변환하여 return
            "real": image_tensor, # (3*512*512) 정답 이미지
            "masked": masked_image,# (3*512*512) 일부가 가려진 이미지
            "mask": mask, # (1*512*512) 보존, 생성 영역 구분
            "generator_input": generator_input # (4*512*512) generator에 실제로 넣는 입력
        }
    # 채널 구성:
        # masked_image : RGB 3채널
        # mask: 마스크 1채널
        # 결과: 총 4채널