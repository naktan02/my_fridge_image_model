import os
import shutil
from pathlib import Path
from roboflow import Roboflow
from dotenv import load_dotenv

def organize_files(source_dir: Path, dest_dir: Path):
    """
    Roboflow에서 다운로드한 데이터셋을 프로젝트 구조에 맞게 정리합니다.
    - 소스: source_dir / (train|valid|test) / (images|labels)
    - 목적지: dest_dir / (images|labels) / (train|val|test)
    """
    print(f"\n📂 '{source_dir}' → '{dest_dir}' 구조로 정리 중...")

    # 목적지 폴더 생성 (없으면 자동 생성)
    (dest_dir / "images").mkdir(parents=True, exist_ok=True)
    (dest_dir / "labels").mkdir(parents=True, exist_ok=True)

    for split in ["train", "valid", "test"]:
        dest_split_name = "val" if split == "valid" else split

        src_split = source_dir / split
        if not src_split.exists():
            print(f"⚠️ '{split}' 폴더 없음 → 건너뜀")
            continue

        # 이미지
        src_images = src_split / "images"
        dest_images = dest_dir / "images" / dest_split_name
        dest_images.mkdir(parents=True, exist_ok=True)

        for f in src_images.glob("*"):
            target = dest_images / f.name
            if target.exists():
                print(f"⚠️ 덮어쓰기 방지: {target.name} 이미 존재 → 건너뜀")
                continue
            shutil.move(str(f), str(dest_images))
        print(f"✅ '{dest_split_name}' 이미지 이동 완료 ({len(list(src_images.glob('*')))}개)")

        # 라벨
        src_labels = src_split / "labels"
        dest_labels = dest_dir / "labels" / dest_split_name
        dest_labels.mkdir(parents=True, exist_ok=True)

        for f in src_labels.glob("*.txt"):
            target = dest_labels / f.name
            if target.exists():
                print(f"⚠️ 덮어쓰기 방지: {target.name} 이미 존재 → 건너뜀")
                continue
            shutil.move(str(f), str(dest_labels))
        print(f"✅ '{dest_split_name}' 라벨 이동 완료 ({len(list(src_labels.glob('*.txt')))}개)")

    print("\n🎯 데이터셋 정리 완료! 삭제 작업은 수행하지 않았습니다.")


def main():
    try:
        # --- 사용자 설정 ---
        WORKSPACE_ID = "myfridge-2ey6e"
        PROJECT_ID = "my_fridge-4s8uk"
        VERSION_NUMBER = 1

        # --- API 키 불러오기 ---
        api_key = os.environ["ROBOFLOW_API_KEY"]

        rf = Roboflow(api_key=api_key)
        project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)
        version = project.version(VERSION_NUMBER)

        # --- 다운로드 위치 ---
        download_location = Path(f"./_temp_{PROJECT_ID}_{VERSION_NUMBER}")
        download_location.mkdir(parents=True, exist_ok=True)

        dataset = version.download(model_format="yolov8", location=str(download_location))
        print(f"\n✅ Roboflow 데이터셋 다운로드 완료 → {dataset.location}")

        # --- 정리 ---
        project_root = Path(__file__).parent.resolve()
        source_directory = Path(dataset.location)
        destination_directory = project_root / "data"

        organize_files(source_directory, destination_directory)

        # --- 안전한 임시폴더 정리 ---
        if "_temp_" in str(download_location):
            shutil.rmtree(download_location, ignore_errors=True)
            print(f"🧹 안전하게 임시폴더 삭제 완료: {download_location}")
        else:
            print(f"⚠️ 예상치 못한 경로: {download_location}, 삭제 생략.")

        print("\n🎉 모든 데이터 준비 완료! 이제 'bash scripts/train.sh' 실행 가능.")

    except KeyError:
        print("❌ 환경 변수 'ROBOFLOW_API_KEY'가 없습니다. 터미널에서:")
        print("   export ROBOFLOW_API_KEY='YOUR_KEY_HERE'")
    except Exception as e:
        print(f"🚨 오류 발생: {e}")


if __name__ == "__main__":
    main()
