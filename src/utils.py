# Common utility functions used across the project

import json
from pathlib import Path
from typing import Any, Dict, Iterable


def read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON file and return it as a Python dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    """Write a dictionary to a JSON file (pretty formatted)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """
    Write an iterable of dictionaries to a JSONL file.
    JSONL = one JSON object per line.
    """
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    """Append a single dictionary as a new line in a JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path):
    """Read a JSONL file and yield one dictionary per line."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line.strip())