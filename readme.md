# GAN_OUT

이미지의 가장자리 일부를 생성하는 아웃페인팅 GAN 프로젝트. Gated Convolution 기반 Generator와 PatchGAN Discriminator를 사용하며, 학습할 때마다 1~3개 방향을 무작위로 가려 복원하도록 학습합니다.

## 실행 방법

Python 3 환경에서 PyTorch, torchvision, Pillow 필요.

```bash
pip install torch torchvision pillow
mkdir -p data/processed checkpoints samples
python preprocess.py --input <원본_이미지_폴더> --output data/processed
python train.py
```

학습 설정은 `train.py` 상단에서 변경할 수 있습니다. 기본값은 이미지 크기 256, 배치 크기 8, 5 epoch이며 최대 5,000장의 이미지를 사용합니다.

학습 중 생성된 미리보기는 `samples/`, 모델 체크포인트는 `checkpoints/`에 저장됩니다.
