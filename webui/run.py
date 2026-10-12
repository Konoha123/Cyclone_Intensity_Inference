#!/usr/bin/env python
"""webui 启动入口（本地展示工具）。

用法（在 cyclone-intensity-inference-env 环境中，仓库根目录下执行）：
    python webui/run.py [--weights weights/model.pt] [--host 127.0.0.1] [--port 8000]

启动时经 src/cyclone_intensity 核心库校验权重 SHA-256 与类别表，失败即退出。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "webui"))

import uvicorn

from backend.main import DEFAULT_WEIGHTS, create_app


def main() -> int:
    parser = argparse.ArgumentParser(description="Cyclone Intensity WebUI（本地展示工具）")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="可信权重路径（默认 weights/model.pt）")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认仅本机）")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app = create_app(weights=args.weights)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
