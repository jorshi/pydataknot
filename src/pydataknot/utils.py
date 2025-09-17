import json
from typing import Any


def json_dump(obj: Any, indent: int = 4) -> str:
    """Pretty-print JSON with objects indented and lists kept on a single line."""

    def _dump(x: Any, level: int) -> str:
        pad = " " * (indent * level)
        nxt = " " * (indent * (level + 1))

        if isinstance(x, dict):
            if not x:
                return "{}"
            parts = []
            for i, (k, v) in enumerate(x.items()):
                key = json.dumps(k)
                val = _dump(v, level + 1)
                parts.append(f"{nxt}{key}: {val}")
            return "{\n" + ",\n".join(parts) + "\n" + pad + "}"

        elif isinstance(x, list):
            # Keep lists inline
            if not x:
                return "[]"
            items = [_dump(v, level + 1) for v in x]
            return "[" + ", ".join(items) + "]"

        # Primitives (str, int, float, bool, None) get normal JSON encoding
        else:
            return json.dumps(x)

    return _dump(obj, 0)
