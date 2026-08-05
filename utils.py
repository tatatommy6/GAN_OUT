from pathlib import Path
from torchvision.utils import save_image
import torch

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "samples"

def set_requires_grad(model, requires_grad):
    for parameter in model.parameters():
        parameter.requires_grad = requires_grad

def missing_region_L1(fake, real, mask):
    # 생성 영역(mask == 0)에 대해서만 L1 loss 계산
    missing = 1.0 - mask
    difference = torch.abs(fake - real) * missing

    denominator = missing.sum() * real.shape[1]
    denominator = denominator.clamp(min = 1.0)

    return difference.sum() / denominator

def save_preview(masked, generated, real, mask, epoch):
    # 순서: 가려진 입력 | 생성 결과 | 정답
    composite = real * mask + generated * (1.0 - mask)

    count = min(4, real.shape[0])

    preview = torch.cat([masked[:count], composite[:count], real[:count]], dim = 3)

    preview = (preview + 1.0) / 2.0
    preview = preview.clamp(0, 1)

    save_image(preview, SAMPLE_DIR / f"epoch_{epoch:03d}.jpg", nrow = 1)

def save_checkpoint(generator, discriminator, optimizer_G, optimizer_D, epoch):
    torch.save(
        {
            "epoch": epoch,
            "generator": generator,
            
        }
    )