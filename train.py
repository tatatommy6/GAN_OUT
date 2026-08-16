from pathlib import Path
from dataset import CustomDataset
from models.generator import Generator
from models.discriminator import Discriminator
from torch.utils.data import DataLoader, Subset
from torchmetrics.image import StructuralSimilarityIndexMeasure # 구조적 유사도 지수: 두 이미지의 화질이나 유사도를 느끼게 평가하는 지표
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity # 학습된 지각적 이미지 패치 유사도: 라고 부릅니다. 두 이미지 사이의 거리를 측정하여 사람이 느끼는 시각적 차이와 비슷하게 유사도를 평가하는 딥러닝 기반 지표
from utils import set_requires_grad, missing_region_L1, save_preview, save_checkpoint, weighted_patch_mean

import torch
import random
import torch.nn.functional as F

# please check the test number!
TEST_NUM = 2
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
CHECKPOINT_DIR = ROOT / "checkpoints"
SAMPLE_DIR = ROOT / "samples"

IMG_SIZE = 512
BATCH_SIZE = 16
EPOCHS = 80
VAL_RATIO = 0.1
SPLIT_SEED = 42
NUM_WORKERS = 4
LAMBDA_RECON = 10.0 # L1 loss 에 곱하는 가중치. 50이면 픽셀 복원 정화도를 비교적 강하게 강조
SAVE_INTERVAL = 5

LR_GEN = 1e-4
LR_DISC = 1e-4

@torch.no_grad()
def validation(generator, val_loader, device, ssim_metric, lpips_metric):
    generator.eval()

    ssim_metric.reset()
    lpips_metric.reset()
    
    total_L1 = 0.0
    total_images = 0.0

    for batch in val_loader:
        real = batch["real"].to(device)
        mask = batch["mask"].to(device)
        gen_input = batch["generator_input"].to(device)
        generated = generator(gen_input)

        # 보존 영역은 원본을 사용하고 생성 영역만 생성 결과를 사용
        composite = real * mask + generated * (1.0 - mask)

        batch_size = real.shape[0]

        # 생성 영역에 대해서만 계산되는 L1 값
        loss_L1 = missing_region_L1(generated, real, mask)

        # ssis는 입력 범위가 [0, 1]이므로 기존 [-1, 1]인 범위를 적절한 범위로 변경
        composite_01 = ((composite + 1.0) / 2.0).clamp(0, 1)
        real_01 = ((real + 1.0) / 2.0).clamp(0, 1)

        ssim_metric.update(composite_01, real_01)
        lpips_metric.update(composite, real)

        total_L1 += loss_L1.item() * batch_size
        total_images += batch_size

    result = {
        "l1": total_L1 / total_images,
        "ssim": ssim_metric.compute().item(),
        "lpips": lpips_metric.compute().item(),
        }

    # metric이 GPU tensor를 계속 보관하지 않도록 정리
    ssim_metric.reset()
    lpips_metric.reset()

    generator.train()
    return result


def train():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    train_base_dataset = CustomDataset(
        DATA_DIR,
        img_size=IMG_SIZE,
        fixed_preview=False,
    ) # 학습용: 랜덤 crop, flip, color jitter, 랜덤 마스크 사용

    val_base_dataset = CustomDataset(
        DATA_DIR,
        img_size=IMG_SIZE,
        fixed_preview=True,
    ) # preview용: 같은 이미지와 같은 마스크 사용

    dataset_size = len(train_base_dataset)

    # 실행할 때마다 같은 train/val 분할을 사용
    split_generator = torch.Generator().manual_seed(SPLIT_SEED) # 난수 생성기
    indices = torch.randperm(dataset_size, generator = split_generator).tolist() # randperm: 0부터 n-1까지의 정수를 무작위로 섞어서 1차원 텐서로 변환

    val_size = min(int(dataset_size * VAL_RATIO), 512)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_dataset = Subset(train_base_dataset, train_indices)
    val_dataset = Subset(val_base_dataset, val_indices)

    print(f"Train images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    dataloader = DataLoader(train_dataset, 
                            batch_size = BATCH_SIZE, # 이미지 8개를 한 묶음 
                            shuffle = True,
                            num_workers = NUM_WORKERS, # worker 프로세스 4개가 데이터를 병렬로 준비. 각 worker는 이미지 열기, crop, flip, 색상 변형, 마스크 생성 등을 담당.
                            pin_memory = device.type == "cuda",
                            persistent_workers = NUM_WORKERS > 0, 
                            # 일반적으로 각 에폭이 끝나면 worker 프로세스를 종료했다가 다음 에폭에 다시 만들 수 있는데 위 옵션을 사용하여 계속 유지하여 반복적인 프로세스 생성 비용을 줄일 수 있음
                            drop_last = True) #마지막 배치의 이미지수가 8보다 작으면 그 배치를 버림

    val_loader = DataLoader(val_dataset,
                            batch_size = BATCH_SIZE,
                            shuffle = False,
                            num_workers = NUM_WORKERS,
                            pin_memory = device.type == "cuda",
                            persistent_workers = NUM_WORKERS > 0,
                            drop_last = False)

    preview_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    fixed_preview = next(iter(preview_loader))
    fixed_preview = {
        key: value.to(device)
        for key, value in fixed_preview.items()
    }
    
    generator = Generator().to(device) # generator 객체를 만들고 mps로 옮김
    discriminator = Discriminator().to(device) # discriminator 객체를 만들고 mps로 옮김

    # generator 와 discriminator 는 서로 다른 목료를 가지므로 optimizer도 따로 만듦
    optimizer_G = torch.optim.Adam(generator.parameters(), lr = LR_GEN, betas = (0.0, 0.9))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = LR_DISC, betas = (0.0, 0.9))
    start_epoch = 1

    checkpoint_path = (CHECKPOINT_DIR / "test2_size512_batch16_epoch80_recon50_19767pics__epoch015.pt") #이어서 학습 할 때
    if checkpoint_path.exists():

        checkpoint = torch.load(checkpoint_path, map_location = device, weights_only = True)
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_G.load_state_dict(checkpoint["optimizer_G"])
        optimizer_D.load_state_dict(checkpoint["optimizer_D"])

        start_epoch = checkpoint["epoch"] + 1

        print(f"Checkpoint loaded: epoch {checkpoint['epoch']}")
        print(f"Resume training from epoch {start_epoch}")

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    ssim_metric = StructuralSimilarityIndexMeasure(
        data_range=1.0,
    ).to(device)

    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type="alex",
        normalize=False,
    ).to(device)

    best_val_lpips = float("inf")

    #============학습 반복문===========
    for epoch in range(start_epoch, EPOCHS + 1):
        # 두 모델 학습 모드로 변경
        generator.train() 
        discriminator.train()

        for batch_index, batch in enumerate(dataloader, start = 1): # 배치 반복문
            # 로드 된 이미지 / 배치 사이즈 = 한 에폭 당 반복 되는 횟수

            # DataLoader 가 만든 딕셔너리에서 각각의 텐서를 꺼내 mps로 옮김
            real = batch["real"].to(device) # shape : [8,3,256,256]
            mask = batch["mask"].to(device) # shape: [8, 1, 256, 256]
            masked = batch["masked"].to(device) # shpae: [8,3,256,256]
            generator_input = batch["generator_input"].to(device) # shpae: [8,4,256,256]
            #===========training dicriminator===========
            # D 값이 양수로 클수록 진짜라고 판단, 음수로 작을수록 가짜라고 판단
            set_requires_grad(discriminator, True) # 모든 파라미터에 requires_grad 옵션 True로 설정
            optimizer_D.zero_grad(set_to_none = True) # pytorch에서 backward로 계산한 grad는 누적되는데 배치마다 독립적으로 업데이트 할려고 gradient를 제거함

            # Autocasting
            with torch.autocast(device_type="cuda", dtype = torch.bfloat16):
                with torch.no_grad(): #generator로 가짜 이미지 생성
                    generated = generator(generator_input)

                fake_composite = (real * mask + generated * (1.0 - mask)) # generator 출력 전체를 그대로 discriminator에 넣지 않고 원본 영역과 생성 영역을 합친 최종 이미지를 만듦
                real_score = discriminator(real, mask)
                fake_score = discriminator(fake_composite.detach(), mask)

                #hinge GAN loss
                loss_d_real = weighted_patch_mean(F.relu(1.0 - real_score), mask)
                loss_d_fake = weighted_patch_mean(F.relu(1.0 + fake_score), mask)
                loss_d = loss_d_real + loss_d_fake # 실제 이미지 손실과 가짜 이미지 손실을 더한 판별자의 최종 손실

            loss_d.backward() # 역전파
            optimizer_D.step() # 파라미터 업데이트

            #===========training generator===========

            set_requires_grad(discriminator, False) # generator 학습하는 동안 discriminator 파라미터가 업데이트 되지 않도록 동결
            optimizer_G.zero_grad(set_to_none=True) # 이전 배치에서 생성자에 남아있는 gradient를 초기화함
            with torch.autocast(device_type="cuda", dtype = torch.bfloat16): # 내부 연산 일부를 bfloat16으로 실행하여 GPU 메모리 사용량을 줄이고 연산 속도를 높임
                generated = generator(generator_input)

                fake_composite = (real * mask + generated * (1.0 - mask)) # 원본 이미지의 보존 영역과 생성자가 복원한 영역을 합쳐 최종 학습 이미지를 만듦

                fake_score = discriminator(fake_composite, mask) # generator가 만든 합성 이미지를 discriminator에 넣음. generator는 fake_score를 높이는 방향으로 학습됨

                loss_g_adv = -weighted_patch_mean(fake_score, mask)
                loss_g_recon = missing_region_L1(generated, real, mask) # 생성 결과와 실제 이미지 사이의 L1 복원 손실을 계산
                loss_g = (loss_g_adv + LAMBDA_RECON * loss_g_recon) # generator의 최종 손실

            loss_g.backward() # 역전파
            optimizer_G.step() # 파라미터 업데이트

            if batch_index % 50 == 0: # 1 배치마다 출력
                print(f"Epoch [{epoch}/{EPOCHS}] "
                    f"Batch [{batch_index}/{len(dataloader)}] "
                    f"D: {loss_d.item():.4f} "
                    f"G: {loss_g.item():.4f} "
                    f"Adv: {loss_g_adv.item():.4f} "
                    f"L1: {loss_g_recon.item():.4f}")

        val_metrics = validation(generator, val_loader, device, ssim_metric, lpips_metric)
        print(
            f"Validation Epoch [{epoch}/{EPOCHS}] "
            f"L1: {val_metrics['l1']:.4f} "
            f"SSIM: {val_metrics['ssim']:.4f} "
            f"LPIPS: {val_metrics['lpips']:.4f}"
            )

        if val_metrics["lpips"] < best_val_lpips:
            best_val_lpips = val_metrics["lpips"]
            best_checkpoint_path = (CHECKPOINT_DIR / f"test{TEST_NUM}_best_lpips.pt")

            torch.save(
                    {
                        "epoch": epoch,
                        "generator": generator.state_dict(),
                        "discriminator": discriminator.state_dict(),
                        "optimizer_G": optimizer_G.state_dict(),
                        "optimizer_D": optimizer_D.state_dict(),
                        "val_metrics": val_metrics,
                        "config": {
                            "img_size": IMG_SIZE,
                            "batch_size": BATCH_SIZE,
                            "lambda_recon": LAMBDA_RECON,
                            "lr_generator": LR_GEN,
                            "lr_discriminator": LR_DISC,
                            "split_seed": SPLIT_SEED,
                            "val_ratio": VAL_RATIO,
                        },
                    },
                    best_checkpoint_path,
                )
            print(
                    f"Best checkpoint saved: "
                    f"epoch={epoch}, "
                    f"LPIPS={best_val_lpips:.4f}"
                )
            
        if epoch % SAVE_INTERVAL == 0:
            generator.eval()

            with torch.no_grad():
                preview_generated = generator(
                    fixed_preview["generator_input"]
                )

            generator.train()

            save_preview(
                fixed_preview["masked"],
                preview_generated,
                fixed_preview["real"],
                fixed_preview["mask"],
                EPOCHS,
                epoch,
                TEST_NUM,
                IMG_SIZE,
                BATCH_SIZE,
                LAMBDA_RECON,
                len(train_base_dataset),
            )

            save_checkpoint(
                generator,
                discriminator,
                optimizer_G,
                optimizer_D,
                EPOCHS,
                epoch,
                TEST_NUM,
                IMG_SIZE,
                BATCH_SIZE,
                LAMBDA_RECON,
                len(train_base_dataset)
            )

            print(f"{epoch} epoch 저장 완료")

if __name__ == "__main__":
    train()
