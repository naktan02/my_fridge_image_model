#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
import sys

# === 설정 ===
RELEASE_DIR = Path("release")                # 소스/출력 모두 release 폴더
SRC_JSON_NAME = "labels_source.json"         # 기본 파일명 (없으면 자동 탐지)
OUT_TXT_NAME  = "labels.txt"
OUT_META_NAME = "labels_meta.json"
# ============

def find_source_json(release_dir: Path, default_name: str) -> Path:
    """release 폴더에서 소스 JSON을 찾는다.
       1) default_name이 있으면 그걸 사용
       2) 없으면 *.json을 스캔해 '단 하나'일 때 그걸 사용
    """
    cand = release_dir / default_name
    if cand.exists():
        return cand

    json_files = sorted(p for p in release_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"[오류] release 폴더에 JSON 파일이 없습니다: {release_dir.resolve()}")
    if len(json_files) > 1:
        raise FileNotFoundError(
            f"[오류] release 폴더에 JSON이 여러 개입니다. {default_name} 이름으로 정해 주세요.\n"
            f" - 감지된 파일들: {[p.name for p in json_files]}"
        )
    return json_files[0]

def main():
    # 0) 폴더 보장
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 소스 JSON 결정
    src_json_path = find_source_json(RELEASE_DIR, SRC_JSON_NAME)
    out_txt_path  = RELEASE_DIR / OUT_TXT_NAME
    out_meta_path = RELEASE_DIR / OUT_META_NAME

    print(f"[INFO] Source JSON : {src_json_path.resolve()}")
    print(f"[INFO] Output TXT  : {out_txt_path.resolve()}")
    print(f"[INFO] Output META : {out_meta_path.resolve()}")

    # 2) 로드
    with src_json_path.open("r", encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError("[오류] JSON 최상위 구조가 list가 아닙니다.")

    # 3) id 기준 정렬 + 연속성 검증
    try:
        items.sort(key=lambda x: x["id"])
    except KeyError:
        raise KeyError("[오류] 모든 항목에 'id' 키가 있어야 합니다.")
    ids = [it["id"] for it in items]
    if ids != list(range(len(items))):
        raise AssertionError(f"[오류] id가 0..N-1 연속이 아닙니다: {ids}")

    # 4) labels.txt (name만, 줄 단위)
    with out_txt_path.open("w", encoding="utf-8", newline="\n") as f:
        for it in items:
            name = str(it.get("name", "")).strip()
            if not name:
                raise ValueError(f"[오류] name 누락/공백 항목 존재: id={it.get('id')}")
            f.write(name + "\n")

    # 5) labels_meta.json (server_key, synonyms만)
    meta = [
        {
            "server_key": it.get("server_key", ""),
            "synonyms": it.get("synonyms", []),
        }
        for it in items
    ]
    with out_meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] 생성 완료: {out_txt_path.name}, {out_meta_path.name} (classes={len(items)})")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
