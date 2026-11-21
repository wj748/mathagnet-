# -*- coding: utf-8 -*-
"""公网发布入口：绑定 0.0.0.0:$PORT（沙箱反向代理要求）。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("HOST", "0.0.0.0")
os.environ.setdefault("PORT", "8787")

from math_agent.api.server import main  # noqa: E402

if __name__ == "__main__":
    main()
