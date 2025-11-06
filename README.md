# my_fridge_image_model


# 1. setup.sh에 실행 권한 부여 (이건 딱 한 번만 하면 됩니다)
chmod +x setup.sh

# 2. 초기 설정 스크립트 실행
./setup.sh


source .venv/bin/activate

python download_data.py

./scripts/train.sh

./scripts/predict.sh

./scripts/resume_train.sh

./scripts/export.sh