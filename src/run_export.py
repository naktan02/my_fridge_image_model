# src/run_train.py
from pathlib import Path
import yaml
import hydra
from omegaconf import DictConfig
from ultralytics import YOLO

def _resolve_dataset_dirs(data_yaml: Path):
    """data.yaml을 읽어 train/val/test labels 디렉터리들을 절대경로로 반환."""
    with open(data_yaml, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)

    # base path
    base = d.get("path", None)
    base = Path(base).resolve() if base else None

    def _abs_img_dir(key):
        v = d.get(key, None)
        if v is None:
            return None
        p = Path(v)
        if not p.is_absolute():
            if base is None:
                # base가 없으면 data.yaml 기준 상대경로
                p = (data_yaml.parent / p).resolve()
            else:
                p = (base / p).resolve()
        return p

    train_img = _abs_img_dir("train")
    val_img   = _abs_img_dir("val")
    test_img  = _abs_img_dir("test")  # 없으면 None 가능

    def _labels_dir(img_dir: Path | None):
        if img_dir is None:
            return None
        # .../images → .../labels
        if img_dir.name == "images":
            return img_dir.parent / "labels"
        # 사용자 커스텀일 경우에도 상응 디렉터리 추정 시도
        cand = img_dir.parent / "labels"
        return cand

    return {
        "train_labels": _labels_dir(train_img),
        "val_labels":   _labels_dir(val_img),
        "test_labels":  _labels_dir(test_img),
        "names": d.get("names", None),
        "nc": d.get("nc", None),
    }

def _assert_detection_dataset(data_yaml: Path) -> None:
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    info = _resolve_dataset_dirs(data_yaml)
    counts = {}
    for split in ("train_labels", "val_labels", "test_labels"):
        p = info[split]
        if p is None:
            counts[split] = 0
            continue
        counts[split] = len(list(Path(p).glob("*.txt")))

    total = sum(counts.values())
    if counts["train_labels"] == 0 or total == 0:
        raise RuntimeError(
            "라벨 .txt를 찾지 못했습니다.\n"
            f"- train labels: {info['train_labels']} ({counts['train_labels']})\n"
            f"- val   labels: {info['val_labels']} ({counts['val_labels']})\n"
            f"- test  labels: {info['test_labels']} ({counts['test_labels']})\n"
            "→ data.yaml의 path/train/val/test 경로와 라벨 폴더를 확인하세요."
        )

    # names/nc 간단 일관성 체크(있으면)
    names = info["names"]
    nc = info["nc"]
    if names is not None and nc is not None and isinstance(names, list) and isinstance(nc, int):
        if len(names) != nc:
            print(f"⚠️  경고: nc({nc}) != len(names)({len(names)}). data.yaml을 확인하세요.")

def _assert_detection_model(model_name: str) -> None:
    name = model_name.lower()
    bad = ("-cls.pt", "-seg.pt", "-pose.pt")
    if any(name.endswith(suf) for suf in bad):
        raise RuntimeError(
            f"탐지 전용이 아닌 모델 가중치 지정: {model_name}\n"
            "→ 예: yolo11n.pt, yolov8n.pt 같은 det 전용을 사용하세요."
        )

@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg: DictConfig) -> None:
    data_cfg = Path(cfg.model.data_cfg).resolve()
    _assert_detection_dataset(data_cfg)
    _assert_detection_model(cfg.model.name)

    model = YOLO(cfg.model.name)

    hsv = bool(cfg.train.hsv)
    hsv_kwargs = dict(
        hsv_h=0.015 if hsv else 0.0,
        hsv_s=0.7 if hsv else 0.0,
        hsv_v=0.4 if hsv else 0.0,
    )

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
