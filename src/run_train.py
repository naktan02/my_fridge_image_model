# run_train.py
from pathlib import Path
import hydra
from omegaconf import DictConfig
from ultralytics import YOLO

def _assert_detection_dataset(data_yaml: Path) -> None:
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")
    # labels/*.txt 존재 여부로 빠르게 검증
    labels_root = (data_yaml.parent / "labels")
    txt_cnt = len(list(labels_root.rglob("*.txt")))
    if txt_cnt == 0:
        raise RuntimeError(
            "labels/*.txt 가 없습니다. 분류 데이터로 보입니다.\n"
            "→ 탐지 포맷(YOLO txt)으로 변환하거나 conf/data.yaml 경로/구조를 확인하세요."
        )

def _assert_detection_model(model_name: str) -> None:
    name = model_name.lower()
    bad = ("-cls.pt", "-seg.pt", "-pose.pt")
    if any(name.endswith(suf) for suf in bad):
        raise RuntimeError(
            f"탐지 전용이 아닌 모델 가중치 지정: {model_name}\n"
            "→ 예: yolo11n.pt, yolov8n.pt 같은 det 전용을 사용하세요."
        )

@hydra.main(version_base=None, config_path="./conf", config_name="train")
def main(cfg: DictConfig) -> None:
    data_cfg = Path(cfg.model.data_cfg).resolve()
    _assert_detection_dataset(data_cfg)
    _assert_detection_model(cfg.model.name)

    model = YOLO(cfg.model.name)

    hsv = bool(cfg.train.hsv)
    hsv_kwargs = dict(hsv_h=0.015 if hsv else 0.0,
                      hsv_s=0.7 if hsv else 0.0,
                      hsv_v=0.4 if hsv else 0.0)

    model.train(
        data=str(data_cfg),
        imgsz=int(cfg.train.imgsz),
        epochs=int(cfg.train.epochs),
        batch=int(cfg.train.batch),
        device=str(cfg.train.device),
        lr0=float(cfg.train.lr0),
        mosaic=float(cfg.train.mosaic),
        close_mosaic=int(cfg.train.close_mosaic_epoch),
        seed=int(cfg.train.seed),
        **hsv_kwargs,
    )

    model.val(data=str(data_cfg), imgsz=int(cfg.train.imgsz), conf=0.001)

if __name__ == "__main__":
    main()
