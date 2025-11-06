#!/usr/bin/env bash
set -euo pipefail

echo "🚀 프로젝트 초기 설정을 시작합니다..."

# 1. uv 가상환경 생성 및 활성화 안내
echo "1. Python 가상환경을 생성합니다."
# 기존 가상환경이 있다면 덮어쓰지 않고 재사용하도록 수정
uv venv --seed

# 2. uv.lock 파일 기준으로 의존성 동기화 (올바른 명령어)
echo "2. uv.lock 파일 기준으로 의존성을 설치합니다."
uv sync uv.lock

# 3. 셸 스크립트에 실행 권한 부여
echo "3. 모든 셸 스크립트에 실행 권한을 부여합니다."
chmod +x ./scripts/*.sh

# 4. .env 파일 준비
if [ ! -f ".env" ]; then
    echo "4. .env.example을 복사하여 .env 파일을 생성합니다."
    cp .env.example .env
    echo "   ✅ .env 파일에 Roboflow API 키를 입력해주세요."
else
    echo "4. .env 파일이 이미 존재합니다."
fi

# 5. 프로젝트를 편집 가능 모드로 설치 (No module named 'src' 오류 방지)
echo "5. 프로젝트를 편집 가능 모드로 설치합니다."
uv pip install -e .

echo ""
echo "✅ 모든 설정이 완료되었습니다!"
echo "가상환경을 활성화하고 사용하세요: source .venv/bin/activate"