#!/usr/bin/env python
"""Zip the repository into dist/<name>-<version>.zip for archiving (e.g. Zenodo)."""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.0.0"
OUT = ROOT / "dist" / f"provenance-audit-v{VERSION}.zip"
SKIP = {".git", "dist", "processed", "__pycache__", ".pytest_cache", ".ruff_cache"}


def main() -> int:
    OUT.parent.mkdir(exist_ok=True)
    files = []
    for p in ROOT.rglob("*"):
        if not p.is_file() or any(part in SKIP for part in p.parts):
            continue
        files.append(p)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(files):
            z.write(p, Path(f"provenance-audit-{VERSION}") / p.relative_to(ROOT))
    size = OUT.stat().st_size / 1e6
    print(f"wrote {OUT} ({len(files)} files, {size:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
