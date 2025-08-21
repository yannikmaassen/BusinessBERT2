from __future__ import annotations
import json, os
from typing import List, Dict, Any
import yaml


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path: str) -> List[Dict]:
    data: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception:
                pass
    return data


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
