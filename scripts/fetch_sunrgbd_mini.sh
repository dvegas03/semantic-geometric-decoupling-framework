#!/usr/bin/env bash
# Stages a tiny synthetic SUN RGB-D-like slice under data/pilot/sunrgbd for H7 preview.
set -euo pipefail

ROOT="${1:-data/pilot/sunrgbd}"
PYTHON="${PYTHON:-}"
if [ -z "${PYTHON}" ]; then
  if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
  else
    PYTHON="python3"
  fi
fi
mkdir -p "${ROOT}/image" "${ROOT}/depth" "${ROOT}/meta"

"${PYTHON}" - <<'PY'
from pathlib import Path
import json
import numpy as np
from PIL import Image

root = Path("data/pilot/sunrgbd")
n = 8
rng = np.random.default_rng(0)
for i in range(n):
    stem = f"{i:05d}"
    h, w = 480, 640
    rgb = (rng.integers(0, 255, size=(h, w, 3))).astype(np.uint8)
    Image.fromarray(rgb).save(root / "image" / f"{stem}.png")
    depth_m = rng.uniform(0.5, 3.0, size=(h, w)).astype(np.float32)
    x0, y0, x1, y1 = 200, 150, 360, 320
    depth_m[y0:y1, x0:x1] = 1.5
    Image.fromarray((depth_m * 1000).astype(np.uint16)).save(root / "depth" / f"{stem}.png")
    fx = fy = 525.0
    cx, cy = w / 2, h / 2
    zs = [1.4, 1.6]
    ys = [(y0 - cy) / fy * z for z in zs for _ in (0, 1)]
    corners = []
    for z in zs:
        for y in (y0, y1):
            for x in (x0, x1):
                corners.append([(x - cx) / fx * z, (y - cy) / fy * z, z])
    meta = {
        "label": "chair",
        "category": "chair",
        "query_text": "the chair",
        "bbox_2d": [x0, y0, x1, y1],
        "aabb_3d": corners,
        "intrinsics": [fx, fy, cx, cy],
    }
    (root / "meta" / f"{stem}.json").write_text(json.dumps(meta, indent=2))
print(f"staged {n} synthetic SUN-RGB-D-like frames under {root}")
PY

echo "OK: ${ROOT}"
