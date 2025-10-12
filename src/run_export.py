#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export YOLO checkpoint → (A) SavedModel → TFLite(INT8 weights, UINT8 I/O)
                     or (B) Direct TFLite export with NMS
→ release/model.tflite

- 기본은 A 경로(네 기존 흐름) 유지.
- MediaPipe ObjectDetector 호환(출력 2 or 4) 보장을 위해,
  필요 시 B 경로(ultralytics 직접 TFLite export + nms=True)를 토글로 제공.
  config: export.use_direct_tflite=true 로 켜면 됨.

Hydra config (../conf/export.yaml) 예:
  export:
    imgsz: 640
    nms: true
    device: "cpu"            # or "cuda"
    rep_dir: "data/calib"    # 대표 데이터 디렉터리(이미지)
    rep_limit: 200           # 최대 샘플 수(선택)
    use_direct_tflite: false # true면 경로 B 사용(권장: MediaPipe 쓸 때)
    int8: true               # direct tflite 경로에서 int8 양자화 여부
"""

from __future__ import annotations

from pathlib import Path
import shutil
import random
from typing import Iterator, List, Optional

import hydra
from omegaconf import DictConfig
from ultralytics import YOLO

import tensorflow as tf


# -------------------------------
# Utilities
# -------------------------------

def _latest_best(weights_root: Path) -> Path:
    """runs/detect 아래에서 가장 최근 best.pt 선택."""
    candidates = list(weights_root.rglob("best.pt"))
    if not candidates:
        raise FileNotFoundError("No best.pt checkpoint found under runs/detect")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _pick_images(rep_dir: Path, limit: Optional[int]) -> List[Path]:
    """대표 데이터 이미지 목록 수집 (jpg/png/webp/bmp). 없으면 빈 리스트."""
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    imgs = [p for p in rep_dir.rglob("*") if p.suffix.lower() in exts]
    if limit and len(imgs) > limit:
        random.shuffle(imgs)
        imgs = imgs[:limit]
    return imgs


def _load_image_for_rep(path: Path, size: int) -> tf.Tensor:
    """
    SavedModel은 일반적으로 float32 [0,1] 입력.
    대표 데이터도 float32 [0,1]로 맞춰서 넣는다.
    최종 I/O UINT8은 converter에서 지정.
    """
    raw = tf.io.read_file(str(path))
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    if size and size > 0:
        img = tf.image.resize(img, (size, size), method=tf.image.ResizeMethod.BILINEAR)
    img = tf.expand_dims(img, 0)  # [1,H,W,3]
    return img


def _rep_dataset(rep_dir: Optional[Path], size: int, limit: Optional[int]):
    """
    TFLiteConverter representative_dataset generator.
    rep_dir 없으면 경고 출력 후 랜덤 텐서 몇 개로 폴백(품질 저하).
    """
    if rep_dir and rep_dir.exists():
        imgs = _pick_images(rep_dir, limit)
        if imgs:
            print(f"ℹ️  Using {len(imgs)} representative images from: {rep_dir}")
            for p in imgs:
                yield [_load_image_for_rep(p, size)]
            return

    print("⚠️  No representative images found; using random tensors (calibration quality will be poor).")
    for _ in range(8):
        rnd = tf.random.uniform(shape=(1, size, size, 3), minval=0.0, maxval=1.0, dtype=tf.float32)
        yield [rnd]


def _ensure_release_dir() -> Path:
    out = Path("release").resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


# -------------------------------
# Path A: SavedModel → TFLite (INT8 weights, UINT8 I/O)
# -------------------------------

def export_via_savedmodel_to_tflite(
    ckpt: Path,
    imgsz: int,
    nms: bool,
    device: str,
    rep_dir: Optional[Path],
    rep_limit: Optional[int],
) -> Path:
    """
    A 경로: SavedModel로 내보낸 다음, TFLiteConverter로
           INT8(가중치/연산) + UINT8(I/O) 변환.
    """
    model = YOLO(str(ckpt))
    saved_model_dir = Path(
        model.export(
            format="saved_model",
            imgsz=imgsz,
            nms=nms,
            device=device,
        )
    ).resolve()

    # 버전 별 반환 경로 보정
    if saved_model_dir.suffix:
        saved_model_dir = saved_model_dir.with_suffix("") / "saved_model"
    if not saved_model_dir.exists():
        alt = saved_model_dir.parent / "saved_model"
        if alt.exists():
            saved_model_dir = alt
        else:
            raise FileNotFoundError(f"SavedModel not found near: {saved_model_dir}")

    # Converter 설정
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: _rep_dataset(rep_dir, imgsz, rep_limit)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_bytes = converter.convert()

    release_dir = _ensure_release_dir()
    out_path = release_dir / "model.tflite"
    out_path.write_bytes(tflite_bytes)

    print("\n✅ Export via SavedModel → TFLite done (UINT8 I/O).")
    print(f" - checkpoint : {ckpt}")
    print(f" - saved_model: {saved_model_dir}")
    print(f" - release    : {out_path}")
    if rep_dir:
        print(f" - rep_dir    : {rep_dir} (limit={rep_limit})")
    else:
        print(" - rep_dir    : <NONE> (used random tensors)")

    return out_path


# -------------------------------
# Path B: Direct TFLite export (nms=True) via Ultralytics
# -------------------------------

def export_direct_tflite_with_nms(
    ckpt: Path,
    imgsz: int,
    nms: bool,
    device: str,
    int8: bool,
) -> Path:
    """
    B 경로: Ultralytics가 직접 TFLite로 export.
           nms=True로 내보내면 MediaPipe ObjectDetector 호환(2 or 4 outputs) 가능성이 높음.
           int8=True면 정수 양자화(ultralytics 내부 경로 사용).
    """
    model = YOLO(str(ckpt))
    tflite_path = Path(
        model.export(
            format="tflite",
            imgsz=imgsz,
            nms=nms,
            device=device,
            int8=int8,   # int8 양자화(버전 호환 필요)
        )
    ).resolve()

    # 산출물 통일: release/model.tflite 로 복사
    release_dir = _ensure_release_dir()
    out_path = release_dir / "model.tflite"
    shutil.copy2(tflite_path, out_path)

    print("\n✅ Export direct TFLite done.")
    print(f" - checkpoint : {ckpt}")
    print(f" - tflite_src : {tflite_path}")
    print(f" - release    : {out_path}")
    print(f" - nms        : {nms}")
    print(f" - int8       : {int8}")

    return out_path


# -------------------------------
# Main
# -------------------------------

@hydra.main(version_base=None, config_path="../conf", config_name="export")
def main(cfg: DictConfig) -> None:
    # 1) 최신 체크포인트
    runs_root = Path("runs/detect").resolve()
    best_ckpt = _latest_best(runs_root)

    imgsz = int(cfg.export.imgsz)
    nms = bool(cfg.export.nms)
    device = str(cfg.export.device)

    use_direct = bool(cfg.export.get("use_direct_tflite", False))
    int8_direct = bool(cfg.export.get("int8", True))

    # 대표 데이터 옵션(SavedModel→Converter 경로에서 사용)
    rep_dir = None
    rep_limit = None
    if "rep_dir" in cfg.export and cfg.export.rep_dir:
        rep_dir = Path(str(cfg.export.rep_dir)).resolve()
    if "rep_limit" in cfg.export and cfg.export.rep_limit:
        rep_limit = int(cfg.export.rep_limit)

    if use_direct:
        # 경로 B: 직접 TFLite + NMS (MediaPipe 호환에 더 안전)
        export_direct_tflite_with_nms(
            ckpt=best_ckpt,
            imgsz=imgsz,
            nms=nms,
            device=device,
            int8=int8_direct,
        )
    else:
        # 경로 A: SavedModel → TFLiteConverter (네 기존 흐름)
        export_via_savedmodel_to_tflite(
            ckpt=best_ckpt,
            imgsz=imgsz,
            nms=nms,
            device=device,
            rep_dir=rep_dir,
            rep_limit=rep_limit,
        )

    print("\nℹ️  Next step (optional): run your metadata injector (labels) and drop the final tflite into app/assets.")
    print("    e.g., python add_metadata.py  →  app/src/main/assets/model_with_metadata.tflite")


if __name__ == "__main__":
    main()
