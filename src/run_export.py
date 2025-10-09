#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Export YOLO checkpoint → pick only best_full_integer_quant.tflite → copy to release/model.tflite.
   - 모든 파라미터는 외부 Hydra YAML(export.yaml)에서만 읽는다.
   - 내장 기본값/데이터클래스 없음.
"""

from __future__ import annotations

from pathlib import Path
import shutil

import hydra
from omegaconf import DictConfig
from ultralytics import YOLO


def _latest_best(weights_root: Path) -> Path:
    """runs/detect 아래에서 가장 최근 best.pt 선택."""
    candidates = list(weights_root.rglob("best.pt"))
    if not candidates:
        raise FileNotFoundError("No best.pt checkpoint found under runs/detect")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _nearby(export_path: Path, name: str) -> Path:
    """Ultralytics export 반환 경로 기준으로 같은 위치의 산출물을 가리킴."""
    return export_path.with_name(name) if export_path.suffix == ".tflite" else (export_path / name)


def _pick_full_integer_quant(export_path: Path) -> Path:
    """반드시 full integer quant 산출물만 선택."""
    p = _nearby(export_path, "best_full_integer_quant.tflite")
    if p.exists():
        return p
    raise FileNotFoundError(
        f"full INT8 artifact not found: {p}\n"
        f"- Check calibration data path(cfg.export.data) and export logs."
    )


@hydra.main(version_base=None, config_path="../conf", config_name="export")
def main(cfg: DictConfig) -> None:
    # 1) 체크포인트 선택(기본: runs/detect에서 최신 best.pt)
    runs_root = Path("runs/detect").resolve()
    best_ckpt = _latest_best(runs_root)

    # 2) 내보내기 (모든 파라미터는 export.yaml의 cfg.export.* 사용)
    model = YOLO(str(best_ckpt))
    export_path = Path(
        model.export(
            format=cfg.export.format,     # "tflite"
            imgsz=cfg.export.imgsz,       # 예: 640
            int8=cfg.export.int8,         # True → INT8 경로(여러 변형 생성 가능)
            nms=cfg.export.nms,           # True/False
            data=cfg.export.data,         # calibration/data.yaml
            fraction=cfg.export.fraction, # 0~1
            device=cfg.export.device,     # "cpu"/"cuda"
        )
    ).resolve()

    # 3) full_integer_quant만 집어서 릴리즈
    picked = _pick_full_integer_quant(export_path)

    release_dir = Path("release").resolve()
    release_dir.mkdir(parents=True, exist_ok=True)

    target_tflite = release_dir / "model.tflite"
    shutil.copy2(picked, target_tflite)

    print("\n✅ Export done.")
    print(f" - checkpoint : {best_ckpt}")
    print(f" - export ret : {export_path}")
    print(f" - selected   : {picked.name}")
    print(f" - release    : {target_tflite}")


if __name__ == "__main__":
    main()
