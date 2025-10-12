# run_export.py
from __future__ import annotations
from pathlib import Path
import shutil
import sys

import hydra
from omegaconf import DictConfig
from ultralytics import YOLO

def _load_interpreter(tfl_path: Path):
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter(model_path=str(tfl_path))
    except Exception:
        import tensorflow as tf
        return tf.lite.Interpreter(model_path=str(tfl_path))

def _assert_mediapipe_compatible(tfl_path: Path) -> None:
    interp = _load_interpreter(tfl_path)
    interp.allocate_tensors()
    outs = interp.get_output_details()
    n = len(outs)
    if n not in (2, 4):
        raise RuntimeError(f"❌ Bad outputs={n}. Expected 2 or 4. file={tfl_path}")
    ins = interp.get_input_details()[0]
    print(f"ℹ️  INPUT {ins['dtype']} {ins['shape']} | OUTPUTS={n}")

def _latest_best(weights_root: Path) -> Path:
    cands = list(weights_root.rglob("best.pt"))
    if not cands:
        raise FileNotFoundError("runs/detect/*/weights/best.pt 가 없습니다.")
    return max(cands, key=lambda p: p.stat().st_mtime)

@hydra.main(version_base=None, config_path="./conf", config_name="export")
def main(cfg: DictConfig) -> None:
    # 1) 최신 best.pt (탐지 학습 산출물)
    best = _latest_best(Path("runs/detect").resolve())
    print(f"▶ exporting from: {best}")

    # 2) 직출 + NMS 내장 (호환성 최우선)
    if not bool(cfg.export.use_direct_tflite):
        print("⚠️  use_direct_tflite=false → 강제로 true로 전환(호환성 우선).")
    use_direct = True
    imgsz = int(cfg.export.imgsz)
    nms = bool(cfg.export.nms)
    int8 = bool(cfg.export.int8)
    device = str(cfg.export.device)
    project = Path(str(cfg.export.get("project", "./exports"))).resolve()
    name = str(cfg.export.get("name", "tflite_export"))

    model = YOLO(str(best))
    out = model.export(
        format="tflite",
        imgsz=imgsz,
        nms=nms,
        int8=int8,
        device=device,
        project=str(project),
        name=name,
    )
    tflite_path = Path(out).resolve() if isinstance(out, str) else None
    if tflite_path is None or not tflite_path.exists():
        # 일부 버전은 반환값이 디렉터리일 수 있음 → best.tflite 탐색
        tdir = project / name
        cands = list(tdir.glob("*.tflite"))
        if not cands:
            raise FileNotFoundError(f"TFLite 산출물을 찾지 못함: {tdir}")
        tflite_path = cands[0]

    print(f"▶ tflite: {tflite_path}")
    _assert_mediapipe_compatible(tflite_path)

    # 3) release/model.tflite로 복사 (앱 자산 교체용 표준 경로)
    release_dir = Path("release").resolve()
    release_dir.mkdir(parents=True, exist_ok=True)
    out_path = release_dir / "model.tflite"
    shutil.copy2(tflite_path, out_path)
    print(f"✅ release: {out_path}")

    print("\n다음 순서로 마무리:")
    print(" 1) python add_metadata.py  → model_with_metadata.tflite 생성")
    print(" 2) python check_outputs.py model_with_metadata.tflite  (2/4 확인)")
    print(" 3) app/src/main/assets/ 로 파일 교체 → Clean/Rebuild → 기기 재설치")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[EXPORT FAILED] {e}", file=sys.stderr)
        sys.exit(1)
