# 初中数学辅导 Agent —— 生产镜像
# 构建：docker build -t math-agent .
# 运行：docker run -p 8787:8787 --env-file .env -v $(pwd)/data:/app/data math-agent
#   （挂载 data/ 可持久化 SQLite 题库/答题记录/Qdrant 向量库；models/ 建议同样挂载以复用 embedding 模型）
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_ENDPOINT=https://hf-mirror.com

WORKDIR /app

# 先装依赖（利用层缓存；锁版本文件由 `pip freeze` 生成）
COPY requirements.lock.txt .
RUN pip install -r requirements.lock.txt

# torch 走 CPU 专用索引（官方 PyPI 源体积大且不含匹配的 CPU 轮子，与 requirements 注释一致）
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.0"

COPY src/ ./src/
COPY web/ ./web/
COPY config.yaml ./

# data/（题库、向量库、上传目录）与 models/（BGE embedding）由运行时挂载注入，
# 不打进镜像：题库重建执行 `python scripts/build_question_bank.py --with-vector`
EXPOSE 8787
ENV HOST=0.0.0.0 PORT=8787

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8787/api/health')" || exit 1

CMD ["python", "-m", "math_agent.api.server"]
