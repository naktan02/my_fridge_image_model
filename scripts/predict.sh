#!/usr/bin/env bash
set -euo pipefail

# ❗ 모델 경로를 'runs/detect/train2'로 직접 지정합니다.
BEST_MODEL="runs/detect/train2/weights/best.pt"

# 테스트할 이미지 소스 (폴더 또는 파일)를 첫 번째 인자로 받습니다.
# 만약 인자가 없으면 기본값으로 'my_test_images' 폴더를 사용합니다.
IMAGE_SOURCE=${1:-"my_test_images"}

echo "🔍 지정된 모델로 예측을 시작합니다:"
echo "   - 모델: $BEST_MODEL"
echo "   - 대상: $IMAGE_SOURCE"
echo ""

# yolo predict 명령어 실행
yolo predict model="$BEST_MODEL" source="$IMAGE_SOURCE"