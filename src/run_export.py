"""Export best YOLO checkpoint to INT8 TFLite and author release metadata."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import hydra
from omegaconf import DictConfig
from ultralytics import YOLO


def _latest_best(weights_root: Path) -> Path:
    candidates = list(weights_root.rglob("best.pt"))
    if not candidates:
        raise FileNotFoundError("No best.pt checkpoint found under runs/detect")
    return max(candidates, key=lambda path: path.stat().st_mtime)


@hydra.main(version_base=None, config_path="../conf", config_name="export")
def main(cfg: DictConfig) -> None:
    runs_root = Path("../runs/detect").resolve()
    best_ckpt = _latest_best(runs_root)

    model = YOLO(str(best_ckpt))
    export_path = Path(
        model.export(
            format=cfg.export.format,
            imgsz=cfg.export.imgsz,
            int8=cfg.export.int8,
            nms=cfg.export.nms,
            data=cfg.export.data,
            fraction=cfg.export.fraction,
            device=cfg.export.device,
        )
    ).resolve()

    release_dir = Path("../release").resolve()
    release_dir.mkdir(parents=True, exist_ok=True)

    target_tflite = release_dir / "model.tflite"
    target_tflite.write_bytes(export_path.read_bytes())

    checksum = hashlib.sha256(target_tflite.read_bytes()).hexdigest()

    labels_path = release_dir / "labels.json"
    labels_bytes = labels_path.read_bytes() if labels_path.exists() else b""
    labels_checksum = hashlib.sha256(labels_bytes).hexdigest() if labels_bytes else None

    manifest = {
        "model_name": "fridge_yolo11n_int8",
        "version": "0.3.0",
        "checksum_sha256": checksum,
        "task": "detect",
        "input": {
            "size": cfg.export.imgsz,
            "layout": "NHWC",
            "color": "RGB",
            "norm": "0..1",
            "letterbox": True,
        },
        "postprocess": dict(cfg.manifest.postprocess),
        "labels_file": "labels.json",
        "data_version": cfg.manifest.get("data_version", "unknown"),
        "labels_sha256": labels_checksum,
        "notes": "trained on v0.3; mosaic=0.8; hsv=on; close_mosaic_epoch=20",
    }

    (release_dir / "model_manifest.json").write_text(json.dumps(manifest, indent=2))

    checksums_lines = [f"model.tflite  {checksum}"]
    if labels_checksum:
        checksums_lines.append(f"labels.json  {labels_checksum}")
    (release_dir / "CHECKSUMS.txt").write_text("\n".join(checksums_lines) + "\n")


if __name__ == "__main__":
    main()
