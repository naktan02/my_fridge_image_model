# src/run_train.py
from pathlib import Path
import yaml
import hydra
from omegaconf import DictConfig
from ultralytics import YOLO

def _resolve_path(base_dir: Path | None, data_yaml_dir: Path, sub: str | None) -> Path | None:
    if not sub:
        return None
    p = Path(sub)
    if p.is_absolute():
        return p
    # base(path)가 있으면 그 기준, 없으면 data.yaml 기준
    return ((base_dir or data_yaml_dir) / p).resolve()

def _labels_dir_from_images(img_dir: Path | None) -> Path | None:
    if img_dir is None:
        return None
    # .../images -> .../labels
    if img_dir.name == "images":
        return (img_dir.parent / "labels").resolve()
    # 혹시 커스텀이라면 images 옆의 labels 가정
    return (img_dir.parent / "labels").resolve()

def _resolve_dataset_dirs(data_yaml: Path):
    with open(data_yaml, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}

    base = d.get("path", None)
    base_dir = Path(base).resolve() if base else None
    dy_dir = data_yaml.parent.resolve()

    # Roboflow는 폴더가 'valid'지만, 키는 보통 'val'을 씀 → 둘 다 지원
    train_key = "train"
    val_key = "val" if "val" in d else ("valid" if "valid" in d else None)
    test_key = "test" if "test" in d else None

    train_img = _resolve_path(base_dir, dy_dir, d.get(train_key))
    val_img   = _resolve_path(base_dir, dy_dir, d.get(val_key)) if val_key else None
    test_img  = _resolve_path(base_dir, dy_dir, d.get(test_key)) if test_key else None

    return {
        "train_img": train_img,
        "val_img": val_img,
        "test_img": test_img,
        "train_labels": _labels_dir_from_images(train_img),
        "val_labels": _labels_dir_from_images(val_img),
        "test_labels": _labels_dir_from_images(test_img),
        "names": d.get("names"),
        "nc": d.get("nc"),
    }

def _assert_detection_dataset(data_yaml: Path) -> None:
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    info = _resolve_dataset_dirs(data_yaml)

    def _count_txt(p: Path | None) -> int:
        if p is None or not p.exists():
            return 0
        return len(list(p.glob("*.txt")))

    t_cnt = _count_txt(info["train_labels"])
    v_cnt = _count_txt(info["val_labels"])
    s_cnt = _count_txt(info["test_labels"])
    total = t_cnt + v_cnt + s_cnt

    print("=== DATA RESOLVE ===")
    print(f"data.yaml           : {data_yaml}")
    print(f"train images        : {info['train_img']}")
    print(f"val   images        : {info['val_img']}")
    print(f"test  images        : {info['test_img']}")
    print(f"train labels dir    : {info['train_labels']}  (txt={t_cnt})")
    print(f"val   labels dir    : {info['val_labels']}    (txt={v_cnt})")
    print(f"test  labels dir    : {info['test_labels']}   (txt={s_cnt})")
    print("====================")

    if t_cnt == 0 or total == 0:
        raise RuntimeError(
            "라벨 .txt를 찾지 못했습니다.\n"
            f"- train labels: {info['train_labels']} (txt={t_cnt})\n"
            f"- val   labels: {info['val_labels']} (txt={v_cnt})\n"
            f"- test  labels: {info['test_labels']} (txt={s_cnt})\n"
            "→ data.yaml의 path/train/val(valid)/test 경로와 라벨 폴더를 확인하세요."
        )

    names, nc = info["names"], info["nc"]
    if isinstance(names, list) and isinstance(nc, int) and len(names) != nc:
        print(f"⚠️ 경고: nc({nc}) != len(names)({len(names)}). data.yaml을 확인하세요.")

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
