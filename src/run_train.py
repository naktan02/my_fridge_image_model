"""Hydra entry point for YOLOv11 training with deterministic defaults."""
from pathlib import Path

import hydra
from omegaconf import DictConfig
from ultralytics import YOLO


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg: DictConfig) -> None:
    data_cfg = Path(cfg.model.data_cfg).resolve()
    model = YOLO(cfg.model.name)

    hsv_enabled = bool(cfg.train.hsv)
    hsv_kwargs = {
        "hsv_h": 0.015 if hsv_enabled else 0.0,
        "hsv_s": 0.7 if hsv_enabled else 0.0,
        "hsv_v": 0.4 if hsv_enabled else 0.0,
    }

    model.train(
        data=str(data_cfg),
        imgsz=cfg.train.imgsz,
        epochs=cfg.train.epochs,
        batch=cfg.train.batch,
        device=cfg.train.device,
        lr0=cfg.train.lr0,
        mosaic=cfg.train.mosaic,
        close_mosaic=cfg.train.close_mosaic_epoch,
        seed=cfg.train.seed,
        **hsv_kwargs,
    )

    model.val(data=str(data_cfg), imgsz=cfg.train.imgsz, conf=0.001)


if __name__ == "__main__":
    main()
