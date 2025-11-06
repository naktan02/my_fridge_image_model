import sys
import yaml
from pathlib import Path
from roboflow import Roboflow
from dotenv import load_dotenv
import os

PROJECT_VERSION = 1
BASE = Path(__file__).resolve().parent
DATA_YAML_PATH = BASE / "configs/data.yaml"

load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    print("❌ .env에 ROBOFLOW_API_KEY 없음")
    sys.exit(1)

try:
    rf = Roboflow(api_key=api_key)
    # 워크스페이스는 꼭 명시 (여기만 네가 사용하는 슬러그로 고정)
    project = rf.workspace("myfridge-2ey6e").project("my_fridge-4s8uk")
    version = project.version(PROJECT_VERSION)

    print(f"Downloading v{PROJECT_VERSION} ...")
    dataset = version.download("yolov11")
    print(f"✅ downloaded: {dataset.location}")

    dataset_folder_name = Path(str(dataset.location)).name

    with open(DATA_YAML_PATH, 'r', encoding='utf-8') as f:
        data_yaml = yaml.safe_load(f) or {}
        if not isinstance(data_yaml, dict):
            raise ValueError("configs/data.yaml 형식이 dict가 아님")

    data_yaml['path'] = f'./{dataset_folder_name}'

    with open(DATA_YAML_PATH, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, indent=2, allow_unicode=True)

    print(f"✅ path updated -> ./{dataset_folder_name}")

except Exception as e:
    print(f"❌ 오류: {e}")
    print(f"   '{DATA_YAML_PATH}'를 수동 확인 바랍니다.")
    sys.exit(1)
