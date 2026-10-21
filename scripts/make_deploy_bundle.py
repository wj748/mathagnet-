# -*- coding: utf-8 -*-
"""构建自有服务器部署包（tar.gz）。

内容 = 可直接 docker compose 上线的最小集合：
  src/ web/ prompts/ models/ data/          # 代码、前端、BGE 模型、题库+向量库
  config.yaml requirements*.txt Dockerfile docker-compose.yml Caddyfile
  scripts/public_deploy_check.py            # 服务器侧自检
  .env                                      # 已自动关闭 Redis（沙箱/服务器无本地 Redis 时兜底）

用法:  python scripts/make_deploy_bundle.py
输出:  dist/math-agent-deploy-<日期>.tar.gz
注意:  包内 .env 含 GLM API Key 与签名密钥——传输用 scp，勿放网盘/仓库。
"""
import io
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "dist"

COPY_DIRS = ["src", "web", "prompts", "models"]
DATA_DIRS = ["raw", "processed", "qdrant", "uploads"]
COPY_FILES = [
    "config.yaml", "requirements.lock.txt", "requirements.txt",
    "Dockerfile", "docker-compose.yml", "Caddyfile",
    Path("scripts") / "public_deploy_check.py",
]
DATA_DB = "math_agent.db"


def iter_files(base: Path):
    for p in sorted(base.rglob("*")):
        if p.is_file() and not any(
            part in {"__pycache__", ".pytest_cache", "node_modules"} for part in p.parts
        ):
            yield p


def main():
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"math-agent-deploy-{time.strftime('%Y%m%d')}.tar.gz"
    n = 0
    with tarfile.open(out, "w:gz") as tar:
        for d in COPY_DIRS:
            base = ROOT / d
            for f in iter_files(base):
                tar.add(f, arcname=f"{d}/{f.relative_to(base).as_posix()}")
                n += 1
        for d in DATA_DIRS:
            base = ROOT / "data" / d
            if base.exists():
                for f in iter_files(base):
                    tar.add(f, arcname=f"data/{d}/{f.relative_to(base).as_posix()}")
                    n += 1
        for name in [DATA_DB, DATA_DB + "-wal", DATA_DB + "-shm"]:
            f = ROOT / "data" / name
            if f.exists():
                tar.add(f, arcname=f"data/{name}")
                n += 1
        for name in COPY_FILES:
            f = ROOT / name
            if f.exists():
                tar.add(f, arcname=name.as_posix() if hasattr(name, "as_posix") else str(name))
                n += 1
        # .env：强制关闭 Redis（服务器走内存兜底；要用 Redis 再起容器并改回）
        env = (ROOT / ".env").read_text(encoding="utf-8")
        env = "\n".join(
            line if not line.startswith("MATH_AGENT_REDIS__ENABLED=")
            else "MATH_AGENT_REDIS__ENABLED=false"
            for line in env.splitlines()
        ) + "\n"
        info = tarfile.TarInfo(name=".env")
        payload = env.encode("utf-8")
        info.size = len(payload)
        info.mtime = int(time.time())
        tar.addfile(info, io.BytesIO(payload))
        n += 1
    print(f"打包完成: {out}")
    print(f"文件数: {n} | 大小: {out.stat().st_size/1024/1024:.1f} MB")
    print("上传:  scp <包> user@服务器:~/  &&  ssh 解压后见《公网部署清单》三步上线")


if __name__ == "__main__":
    sys.exit(main())
