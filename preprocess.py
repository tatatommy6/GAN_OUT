"""
이 파일이 하는 일:
    EXIF 방향 보정
    RGB 3채널로 통일
    짧은 변 512px 미만 이미지 제외
    긴 변이 2048px를 넘으면 비율을 유지해 축소
    손상된 이미지 제외
    JPEG 품질 95로 통일
"""

from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError

DEFAULT_INPUT = "../data/openimages20k"
DEFAULT_OUTPUT = "../data/processed"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def output_name(path):
    digest = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:10]
    return f"{path.stem}_{digest}.jpg"

def preprocess_image(source, destination, min_side, max_side, quality):
    try:
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            width, height = image.size

            # 원본이 너무 작으면 억지로 확대하지 않고 제외
            if min(width, height) < min_side:
                return False

            # 비율을 유지하면서 긴 변의 메모리 사용량만 제한
            if max(width, height) > max_side:
                scale = max_side / max(width, height)
                new_size = (round(width * scale), round(height * scale))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            destination.parent.mkdir(parents = True, exist_ok = True)
            image.save(destination, "JPEG", quality = quality, optimize = True)
        return True
    except (OSError, UnidentifiedImageError, ValueError):
        return False

def parse_args():
    parser = argparse.ArgumentParser(description = "GAN 학습 이미지 전처리")
    parser.add_argument("--input", type = Path, default = DEFAULT_INPUT)
    parser.add_argument("--output", type = Path, default = DEFAULT_OUTPUT)
    parser.add_argument("--min-side", type = int, default = 512)
    parser.add_argument("--max-side", type = int, default = 2048)
    parser.add_argument("--quality", type = int, default = 95)
    return parser.parse_args()

def main():
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"입력 폴더를 찾을 수 없습니다: {args.input}")

    files = sorted(path for path in args.input.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise RuntimeError(f"이미지가 없습니다: {args.input}")

    saved = 0
    skipped = 0
    for index, source in enumerate(files, start=1):
        destination = args.output / output_name(source)
        if preprocess_image(source, destination, args.min_side, args.max_side, args.quality):
            saved += 1
        else:
            skipped += 1

        if index % 500 == 0 or index == len(files):
            print(f"[{index}/{len(files)}] 저장 {saved}, 제외 {skipped}")

    print(f"완료: {args.output} (저장 {saved}, 제외 {skipped})")

if __name__ == "__main__":
    main()