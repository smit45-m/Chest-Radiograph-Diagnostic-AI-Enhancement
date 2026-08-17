"""YAML config loader with dotted-key access."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config:
    def __init__(self, data: dict[str, Any], root: Path | None = None) -> None:
        self._data = data
        self.root = root or Path.cwd()

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        cfg = cls(data, root=path.parent.parent.resolve())
        return cfg

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def path(self, dotted: str) -> Path:
        value = Path(str(self.get(dotted)))
        if not value.is_absolute():
            value = self.root / value
        return value

    def __getitem__(self, key: str) -> Any:
        return self._data[key]
