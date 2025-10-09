#!/usr/bin/env bash
set -euo pipefail

# runs/detect/ 폴더에서 가장 마지막에 학습된 결과 폴더를 찾습니다.
LATEST_RUN=$(ls -td runs/detect/train* | head -1)

# 해당 폴더에 best.pt 파일이 있는지 확인합니다.
if [ -f "$LATEST_RUN/weights/best.pt" ]; then
    BEST_MODEL="$LATEST_RUN/weights/best.pt"
    echo "✅ 가장 최신 모델을 찾았습니다. 추가 학습을 시작합니다."
    echo "   - 모델 경로: $BEST_MODEL"
    echo ""
    # 찾은 모델을 기반으로 기존 학습 스크립트를 실행합니다.
    ./scripts/train.sh model="$BEST_MODEL" "$@"
else
    echo "⚠️ 이어서 학습할 모델(best.pt)을 찾을 수 없습니다."
    echo "   - 첫 학습을 먼저 진행해 주세요: ./scripts/train.sh"
    exit 1
fi