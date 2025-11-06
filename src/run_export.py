# src/run_export.py (교체)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
import sys, random, shutil, warnings
from typing import Optional, Iterator, List
import hydra
from omegaconf import DictConfig
from ultralytics import YOLO
import tensorflow as tf

def _latest_best(weights_root: Path) -> Path:
    cands = list(weights_root.rglob("best.pt"))
    if not cands:
        raise FileNotFoundError("No best.pt under runs/detect")
    return max(cands, key=lambda p: p.stat().st_mtime)

def _ensure_release_dir() -> Path:
    out = Path("release").resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out

def _inspect_outputs(tflite_path: Path) -> int:
    itp = tf.lite.Interpreter(model_path=str(tflite_path))
    itp.allocate_tensors()
    outs = itp.get_output_details()
    ins = itp.get_input_details()
    print(f"INPUT : {ins}")
    print(f"OUTPUT: {len(outs)}")
    for i, od in enumerate(outs):
        print(f"{i} {od['name']} {od['shape']} {od['dtype']}")
    return len(outs)

def _maybe_enforce_outputs(tflite_path: Path, require_2_or_4: bool):
    n = _inspect_outputs(tflite_path)
    if require_2_or_4 and n not in (2, 4):
        raise RuntimeError(f"Expected 2 or 4 outputs, found {n}: {tflite_path}")
    if not require_2_or_4 and n not in (2, 4):
        print(f"⚠️  NOTE: outputs={n} (YOLO raw). Use plain TFLite+NMS on Android (no MediaPipe ObjectDetector).")

def _copy_to_release(src: Path) -> Path:
    dst = _ensure_release_dir() / "model.tflite"
    shutil.copy2(src, dst)
    print(f"✅ release: {dst}")
    return dst

def _rep_dataset(rep_dir: Optional[Path], size: int, limit: Optional[int]) -> Iterator[list]:
    if rep_dir and rep_dir.exists():
        exts = {".jpg",".jpeg",".png",".webp",".bmp"}
        imgs: List[Path] = [p for p in rep_dir.rglob("*") if p.suffix.lower() in exts]
        if limit and len(imgs) > limit:
            random.shuffle(imgs); imgs = imgs[:limit]
        if imgs:
            print(f"ℹ️  Using {len(imgs)} representative images from: {rep_dir}")
            for p in imgs:
                raw = tf.io.read_file(str(p))
                img = tf.io.decode_image(raw, channels=3, expand_animations=False)
                img = tf.image.convert_image_dtype(img, tf.float32)
                img = tf.image.resize(img, (size, size))
                yield [tf.expand_dims(img, 0)]
            return
    warnings.warn("No representative images; using random tensors.")
    for _ in range(8):
        yield [tf.random.uniform((1, size, size, 3), 0.0, 1.0, tf.float32)]

def export_direct_tflite(
    ckpt: Path, imgsz: int, nms: bool, device: str, int8: bool,
    project: Path, name: str, require_2_or_4: bool
) -> Path:
    model = YOLO(str(ckpt))
    out = model.export(format="tflite", imgsz=imgsz, nms=nms, device=device,
                       int8=int8, project=str(project), name=name)
    tfl = Path(out) if isinstance(out, (str, Path)) else None
    if not tfl or not tfl.exists():
        cands = list((project / name).glob("*.tflite"))
        if not cands:
            raise FileNotFoundError(f"No *.tflite in {(project/name)}")
        tfl = cands[0]
    _maybe_enforce_outputs(tfl, require_2_or_4)
    return _copy_to_release(tfl)

def export_via_savedmodel_uint8_io(
    ckpt: Path, imgsz: int, nms: bool, device: str,
    rep_dir: Optional[Path], rep_limit: Optional[int],
    require_2_or_4: bool
) -> Path:
    model = YOLO(str(ckpt))
    saved = Path(model.export(format="saved_model", imgsz=imgsz, nms=nms, device=device)).resolve()
    if saved.is_file(): saved = saved.with_suffix("") / "saved_model"
    if not saved.exists():
        alt = saved.parent / "saved_model"
        saved = alt if alt.exists() else saved
    if not saved.exists():
        raise FileNotFoundError(f"SavedModel not found: {saved}")

    conv = tf.lite.TFLiteConverter.from_saved_model(str(saved))
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: _rep_dataset(rep_dir, imgsz, rep_limit)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.uint8
    conv.inference_output_type = tf.uint8

    tfl_bytes = conv.convert()
    tmp = Path("exports") / "tmp_uint8_io.tflite"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(tfl_bytes)
    _maybe_enforce_outputs(tmp, require_2_or_4)
    return _copy_to_release(tmp)

@hydra.main(version_base=None, config_path="../conf", config_name="export")
def main(cfg: DictConfig) -> None:
    imgsz   = int(cfg.export.imgsz)
    nms     = bool(cfg.export.nms)
    device  = str(cfg.export.device)
    project = Path(str(cfg.export.get("project", "./exports"))).resolve()
    name    = str(cfg.export.get("name", "tflite_fp32"))

    use_direct     = bool(cfg.export.get("use_direct_tflite", True))
    int8_direct    = bool(cfg.export.get("int8", False))
    io_uint8       = bool(cfg.export.get("io_uint8", False))
    require_mpipe  = bool(cfg.export.get("require_mediapipe_outputs", False))  # ← 새 옵션

    if cfg.export.get("ckpt"):
        ckpt = Path(str(cfg.export.ckpt)).resolve()
    else:
        ckpt = _latest_best(Path("runs/detect").resolve())
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    project.mkdir(parents=True, exist_ok=True)

    if use_direct and not io_uint8:
        export_direct_tflite(ckpt, imgsz, nms, device, int8_direct, project, name, require_mpipe)
    elif io_uint8:
        rep_dir = Path(str(cfg.export.rep_dir)).resolve() if cfg.export.get("rep_dir") else None
        rep_limit = int(cfg.export.rep_limit) if cfg.export.get("rep_limit") else None
        export_via_savedmodel_uint8_io(ckpt, imgsz, nms, device, rep_dir, rep_limit, require_mpipe)
    else:
        export_direct_tflite(ckpt, imgsz, nms, device, False, project, name, require_mpipe)

    print("\nNext:")
    print(" 1) python add_metadata.py  → model_with_metadata.tflite")
    print(" 2) (선택) check_outputs.py model_with_metadata.tflite")
    print(" 3) Android: TFLite Interpreter + 커스텀 NMS로 박스/라벨 렌더")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[EXPORT FAILED] {e}", file=sys.stderr)
        sys.exit(1)
