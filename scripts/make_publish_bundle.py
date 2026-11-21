# -*- coding: utf-8 -*-
"""构建自包含的公网发布目录 publish/（供 sites 沙箱部署）。

沙箱适配要点（逐条对应沙箱限制）：
1. 必须监听 $PORT 且绑定 0.0.0.0        → serve.py 设置 HOST/PORT
2. 沙箱无 Redis / 无数据库服务          → redis.enabled=false，走内存兜底
3. 沙箱无 torch → 嵌入退化为 hashing    → 索引必须用 hashing 重建（见 build_publish_index.py）
4. 同源部署（前端由同一 FastAPI 托管）  → CORS 无需放开，保持收紧
5. 不带 BGE 模型（92MB，沙箱用不到）     → 缩小上传体积
"""
from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "publish"

if PUB.exists():
    shutil.rmtree(PUB)
PUB.mkdir(parents=True)

IGN = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache",
                             "node_modules", ".git", "*.db-wal", "*.db-shm")


def copy_tree(src: Path, dst_rel: str) -> None:
    dst = PUB / dst_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, ignore=IGN, dirs_exist_ok=True)


# ---------------------------------------------------------------- 代码与前端
copy_tree(ROOT / "src", "src")
copy_tree(ROOT / "web", "web")
copy_tree(ROOT / "prompts", "prompts")

# ---------------------------------------------------------------- 向量索引（hashing 重建版）
copy_tree(PUB.parent / "publish" / "data" / "qdrant", "data/qdrant") if False else None
src_qdrant = ROOT / "publish_qdrant_hashing"
if src_qdrant.exists():
    copy_tree(src_qdrant, "data/qdrant")

# ---------------------------------------------------------------- 业务库（一致性快照）
db_src = ROOT / "data" / "math_agent.db"
db_dst = PUB / "data" / "math_agent.db"
db_dst.parent.mkdir(parents=True, exist_ok=True)
s = sqlite3.connect(db_src)
d = sqlite3.connect(db_dst)
s.backup(d)
d.close()
s.close()

# ---------------------------------------------------------------- 配置
cfg = (ROOT / "config.yaml").read_text(encoding="utf-8")
cfg = cfg.replace("backend: auto             # auto | bge | hashing",
                  "backend: hashing         # 公网沙箱无 torch，强制哈希向量（索引同源构建）")
cfg = cfg.replace("  enabled: false\n  url: redis://localhost:6379/0",
                  "  enabled: false          # 公网沙箱无 Redis，走内存兜底\n  url: redis://localhost:6379/0")
(PUB / "config.yaml").write_text(cfg, encoding="utf-8")

# ---------------------------------------------------------------- 依赖（去掉 transformers/torch 相关）
(PUB / "requirements.txt").write_text("""# 公网沙箱精简依赖：不含 transformers/torch（沙箱用 hashing 嵌入）
langgraph>=0.2
langchain-core>=0.3
langchain-openai>=0.2
pydantic>=2.0
pyyaml>=6.0
qdrant-client>=1.9
sqlalchemy>=2.0
sympy>=1.12
fastapi>=0.110
uvicorn[standard]>=0.27
python-multipart>=0.0.9
""", encoding="utf-8")

# ---------------------------------------------------------------- 入口
(PUB / "serve.py").write_text('''# -*- coding: utf-8 -*-
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
''', encoding="utf-8")

# ---------------------------------------------------------------- 环境变量（密钥）
env_dst = PUB / ".env"
if env_dst.exists():
    env_dst.unlink()
src_env = (ROOT / ".env").read_text(encoding="utf-8")
lines = []
for line in src_env.splitlines():
    s_ = line.strip()
    if not s_ or s_.startswith("#"):
        continue
    k = s_.split("=", 1)[0]
    if k in ("MATH_AGENT_REDIS__ENABLED",):
        continue                      # 沙箱无 Redis
    lines.append(s_)
lines.append("MATH_AGENT_REDIS__ENABLED=false")
env_dst.write_text("\n".join(lines) + "\n", encoding="utf-8")

total = sum(p.stat().st_size for p in PUB.rglob("*") if p.is_file())
print(f"✔ publish/ 就绪  总计 {total/1024/1024:.1f}MB")
for p in sorted(PUB.rglob("*")):
    if p.is_file() and p.stat().st_size > 200 * 1024:
        print(f"  {p.relative_to(PUB)}  {p.stat().st_size/1024/1024:.1f}MB")
