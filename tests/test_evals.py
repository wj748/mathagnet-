"""把七套验收评测接成 pytest 真实门禁。

背景：`scripts/eval_*.py` 里的 check() 历史上只 print 不 assert，一旦被 pytest
收集就会出现"断言失败仍判 PASS"的假绿。现已给 check() 加上严格模式
（检测 PYTEST_CURRENT_TEST，失败即 raise），本文件再把各套 main() 的退出码
接成断言，形成可放进 CI 的门禁。

用法：
    RUN_EVALS=1 pytest tests/test_evals.py -q     # 跑全部七套（较慢，LLM 在线时更慢）
    pytest tests/ -q                              # 默认跳过，只跑冒烟

注意：eval 脚本会访问 Qdrant 本地库（单进程锁），跑之前请先停掉 API 服务。
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"

# (脚本名, 说明, 超时秒)
EVAL_SUITES = [
    ("eval_optimizations.py", "四项优化 + 判分准确性", 1800),
    ("eval_conversation.py", "意图路由 + 学习路由 + 行为约束", 900),
    ("eval_memory.py", "记忆优化：压缩/释放/state 有界", 900),
    ("eval_security.py", "安全回归", 600),
    ("eval_teaching.py", "知识点讲解 Agent", 900),
    ("eval_badcase.py", "历史踩坑回归", 1200),
]

_RUN = os.environ.get("RUN_EVALS") == "1"
_SKIP_REASON = "默认跳过（耗时）；设置 RUN_EVALS=1 启用，且请先停掉 API 服务（Qdrant 单进程锁）"


@pytest.mark.evals
@pytest.mark.skipif(not _RUN, reason=_SKIP_REASON)
@pytest.mark.parametrize("script,desc,timeout", EVAL_SUITES,
                         ids=[s for s, _, _ in EVAL_SUITES])
def test_eval_suite(script: str, desc: str, timeout: int) -> None:
    path = SCRIPTS_DIR / script
    assert path.exists(), f"评测脚本缺失：{path}"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(ROOT), env=env, timeout=timeout,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        errors="replace",
    )
    out = proc.stdout or ""
    assert proc.returncode == 0, (
        f"【{script}】{desc} 未通过（退出码 {proc.returncode}）\n"
        f"----- 输出尾部 -----\n{out[-3000:]}"
    )
