from pathlib import Path
from torchvision.utils import save_image
import torch

#폴더 지정 (자동 생성 X)
ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "samples" 
CHECKPOINT_DIR = ROOT / "checkpoints"

def set_requires_grad(model, requires_grad): 
# 모델(discriminator, generator)의 모든 파라미터를 학습할지 여부를 결정 
    for parameter in model.parameters(): 
        parameter.requires_grad = requires_grad
        # requires_grad가 true 면 역전파시 gradient를 계산하여 파라미터를 학습
        # false 면 gradient를 계산하지 않고 파라미터 고정
        # = 한 모델을 학습할때 다른 모델이 업데이트 되지 않게 함.
        # generator를 업데이트 하지 않을때는 fake.detach()로 역전파 차단
        # discriminator를 업데이트 하지 않을때는 set_requires_grad로 역전파 차단

def missing_region_L1(fake, real, mask):
    # 생성 영역(mask == 0)에 대해서만 L1 loss 계산
    # fake: generator가 생성한 이미지 / real: 실제 정답 이미지 / mask: 유지되는 영역은 1, 생성해야하는 영역은 0

    missing = 1.0 - mask # 생성할 영역 성택
    difference = torch.abs(fake - real) * missing #생성 이미지와 정답 이미지의 픽셀별 절댓값 차이 계산
    denominator = missing.sum() * real.shape[1] # 비교 대상 원소 수 계산 missing.sum()은 가려진 픽셀 개수, real.shape[1]은 채널 수. RGB이미지니까 3
    denominator = denominator.clamp(min = 1.0) # 가려진 영역이 하나도 없으면 분모가 0이 되니까 최소값을 1로 제한하여 오류 방지

    return difference.sum() / denominator # avg L1 loss return 

def save_preview(masked, generated, real, mask, totalepoch, epoch, test, img_size, batch_size, recon, pic_cnt):
    # 이미지 저장 순서: 가려진 입력 | 생성 결과 | 정답
    composite = real * mask + generated * (1.0 - mask) #보이는 부분은 원본 유지, 가려졌던 부분만 generator의 출력으로 채움

    count = min(4, real.shape[0]) #이미지 최대 4개만 선택

    preview = torch.cat([masked[:count], composite[:count], real[:count]], dim = 3) # 이미지 텐서는 [배치, 채널, 높이, 넓이] 형태니까 dim = 3 은 너비니까 가로로 연결됨

    preview = (preview + 1.0) / 2.0 # -1 ~ 1범위로 정규화 된걸
    preview = preview.clamp(0, 1) # 0 ~ 1 범위로 되돌림

    # 이미지 저장
    save_image(preview, SAMPLE_DIR / f"test{test}_size{img_size}_batch{batch_size}_epoch{totalepoch:02d}_recon{recon:g}_{pic_cnt}pics_{epoch:03d}.jpg", nrow = 1)

def save_checkpoint(generator, discriminator, optimizer_G, optimizer_D, totalepoch, epoch, test, img_size, batch_size, recon, pic_cnt):
    checkpoint_path = CHECKPOINT_DIR / (
            f"test{test}_size{img_size}_batch{batch_size}_"
            f"epoch{totalepoch:02d}_recon{recon:g}_"
            f"{pic_cnt}pics__epoch{epoch:03d}.pt")
    
    torch.save( #.pt 파일에 이런 내용이 저장됨. 
        {
            "epoch": epoch,
            "generator": generator.state_dict(), # state_dict()를 사용하여 학습 가능한 매개변수가 들어있는 딕셔너리 형태로 저장함.
            "discriminator": discriminator.state_dict(),
            "optimizer_G": optimizer_G.state_dict(),
            "optimizer_D": optimizer_D.state_dict(),
        },
        checkpoint_path
    )

    # 현재 테스트의 체크포인트만 찾기
    pattern = (
        f"test{test}_size{img_size}_batch{batch_size}_"
        f"epoch{totalepoch:02d}_recon{recon:g}_"
        f"{pic_cnt}pics__epoch*.pt")
    checkpoints = sorted(CHECKPOINT_DIR.glob(pattern))

    # 최근 두개의 체크포인트만 남기기
    while len(checkpoints) > 2:
        old_checkpoint = checkpoints.pop(0)
        old_checkpoint.unlink()
        print(f"Deleted old checkpoint: {old_checkpoint.name}")