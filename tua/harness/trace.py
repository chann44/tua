"""Per-run artifacts: `runs/<ts>-<slug>/trace.jsonl` plus `screenshots/`.

Everything the loop does is appended as one JSON line (prompts, tool calls,
results, token usage, latency), so runs are diffable across models and a later
`tua replay` can render an HTML timeline.
"""

from __future__ import annotations

import datetime
import json
import re
import time
from pathlib import Path

from tua.harness.messages import Image


class Trace:
    def __init__(self, runs_dir: Path, task: str):
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        slug = re.sub(r"[^a-z0-9]+", "-", task.lower()).strip("-")[:40] or "run"
        self.dir = runs_dir / f"{ts}-{slug}"
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "screenshots").mkdir(exist_ok=True)
        self._f = (self.dir / "trace.jsonl").open("a", encoding="utf-8")
        self._shots = 0

    def log(self, event: dict) -> None:
        self._f.write(json.dumps({"ts": round(time.time(), 3), **event}, ensure_ascii=False, default=str) + "\n")
        self._f.flush()

    def save_image(self, image: Image) -> str:
        self._shots += 1
        ext = "png" if "png" in image.mime else "jpg"
        path = self.dir / "screenshots" / f"shot-{self._shots:03d}.{ext}"
        path.write_bytes(image.data)
        return str(path.relative_to(self.dir))

    def close(self) -> None:
        self._f.close()
