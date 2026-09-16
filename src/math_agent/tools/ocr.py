"""OCR / 文档解析工具（方案 3.3）。

多后端，按可用性自动选择
------------------------
1. ``paddleocr``  —— 本地识别图片 / PDF（需额外安装 paddleocr）。
2. ``pdf_text``   —— pypdf / PyMuPDF 抽取 PDF 文本层。
3. ``multimodal`` —— 多模态 LLM（GLM-4V 等）兜底，识别手写体。
4. ``plain``      —— 纯文本兜底（txt/md），保证链路不中断。

识别结果统一返回 ``{text, blocks, confidence, backend}``；
**置信度低于阈值时要求用户确认**（方案风险表：OCR 手写识别准确率低）。
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from .. import safety                                    # noqa: F401
from ..config import DATA_DIR
from .registry import register_tool

CONF_THRESHOLD = 0.60

# 安全白名单：只允许解析上传目录内的白名单扩展名文件（P0：防任意文件读取/密钥外泄）
ALLOWED_OCR_DIR = (DATA_DIR / "uploads").resolve()
ALLOWED_OCR_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".pdf", ".txt", ".md"}


def _safe_ocr_path(file_path: str) -> Path:
    p = Path(file_path)
    if not p.is_absolute():
        raise PermissionError("只允许上传目录内的文件，请先通过 /api/upload 上传")
    resolved = p.resolve()
    if ALLOWED_OCR_DIR != resolved and ALLOWED_OCR_DIR not in resolved.parents:
        raise PermissionError(f"拒绝访问：只允许解析 {ALLOWED_OCR_DIR} 目录内的文件")
    if resolved.suffix.lower() not in ALLOWED_OCR_EXT:
        raise PermissionError(f"不支持的文件类型：{resolved.suffix}（白名单：{sorted(ALLOWED_OCR_EXT)}）")
    return resolved


class OCRArgs(BaseModel):
    file_path: str = Field(..., description="本地文件绝对路径（图片 / PDF / 文本）")
    page: int = Field(0, ge=0, description="PDF 页码，0 表示全部页")


def _ocr_paddle(path: str) -> Dict[str, Any]:
    from paddleocr import PaddleOCR  # type: ignore

    ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
    result = ocr.ocr(path, cls=True)
    lines: List[str] = []
    confs: List[float] = []
    for page in result or []:
        for line in page or []:
            if not line:
                continue
            txt = line[1][0]
            conf = float(line[1][1])
            lines.append(txt)
            confs.append(conf)
    if not lines:
        return {"text": "", "blocks": [], "confidence": 0.0, "backend": "paddleocr"}
    return {"text": "\n".join(lines), "blocks": lines,
            "confidence": sum(confs) / len(confs), "backend": "paddleocr"}


def _ocr_pdf(path: str, page: int = 0) -> Dict[str, Any]:
    try:
        from pypdf import PdfReader
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except Exception as e:
            raise RuntimeError("缺少 pypdf，无法解析 PDF") from e
    reader = PdfReader(path)
    pages = [reader.pages[page]] if page > 0 else list(reader.pages)
    texts = [(p.extract_text() or "") for p in pages]
    text = "\n".join(t for t in texts if t.strip())
    if not text.strip():
        return {"text": "", "blocks": [], "confidence": 0.0, "backend": "pdf_text(空)"}
    return {"text": text, "blocks": texts, "confidence": 0.9, "backend": "pdf_text"}


def _ocr_multimodal(path: str) -> Dict[str, Any]:
    """多模态兜底：GLM-4V 视觉模型识别手写体（独立视觉 client，文本模型不支持图片）。"""
    import base64

    from ..llm import get_vision_client

    client = get_vision_client()
    if client is None:
        raise RuntimeError("视觉模型不可用（未配置 GLM_API_KEY / GLM_VISION_MODEL）")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(path)[1].lower().lstrip(".") or "png"
    if ext not in ("jpg", "jpeg", "png", "bmp", "webp"):
        raise RuntimeError(f"视觉模型只支持图片，不支持 .{ext}（PDF 请走文本层解析）")
    mime = {"jpg": "jpeg", "jpeg": "jpeg"}.get(ext, ext)
    from langchain_core.messages import HumanMessage

    msg = HumanMessage(content=[
        {"type": "text", "text": "请识别图片中的数学题目与学生作答，逐题输出：题号、题目、学生答案。保持原始数学符号。"},
        # ⚠ data URL 必须带 image/ 前缀（data:image/png），写成 data:png 会报 1210 图片解析错误
        {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
    ])
    last_err: Exception | None = None
    for attempt in range(3):                       # GLM-4V 偶发 1210 图片解析错误，重试即可恢复
        try:
            resp = client.invoke([msg])
            text = (resp.content or "").strip()
            if text:
                return {"text": text, "blocks": [text], "confidence": 0.75,
                        "backend": "multimodal"}
            last_err = RuntimeError("视觉模型返回为空")
        except Exception as e:
            last_err = e
    raise RuntimeError(f"多模态识别失败：{last_err}")


@register_tool(
    "ocr_document",
    "解析图片 / PDF / 文本中的数学题目与作答内容，返回纯文本。"
    "批改作业前必须先调用本工具获取文本，再调用 grade_assignment。",
    OCRArgs,
    timeout=90,   # 视觉模型多模态调用较慢，单独放宽超时（默认 30s）
)
def ocr_document(file_path: str, page: int = 0) -> Dict[str, Any]:
    safe_path = _safe_ocr_path(file_path)     # P0 硬约束：白名单目录 + 扩展名
    file_path = str(safe_path)
    if not safe_path.exists():
        raise FileNotFoundError(f"文件不存在：{file_path}")
    ext = safe_path.suffix.lower()

    result: Dict[str, Any] | None = None
    errors: List[str] = []

    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        for fn in (_ocr_paddle, _ocr_multimodal):
            try:
                result = fn(file_path)
                break
            except Exception as e:
                errors.append(f"{fn.__name__}: {type(e).__name__}")
        if result is None:
            raise RuntimeError(f"图片 OCR 失败，已尝试：{errors}；请安装 paddleocr 或配置多模态模型")
    elif ext == ".pdf":
        try:
            result = _ocr_pdf(file_path, page)
        except Exception as e:
            errors.append(f"pdf: {e}")
        if not result or not result.get("text", "").strip():
            raise RuntimeError(
                "PDF 无文本层（可能是扫描件）。请把页面截图为图片后重新上传，"
                f"或使用可提取文本的 PDF。已尝试：{errors}")
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        result = {"text": text, "blocks": [text], "confidence": 1.0, "backend": "plain"}

    result["need_confirm"] = float(result.get("confidence", 0.0)) < CONF_THRESHOLD
    result["file_path"] = file_path
    # OCR 提取的是学生作业内容，进会话/记忆/trace 前先做 PII 脱敏（数据最小化）
    result["text"] = safety.mask_pii(result.get("text", ""))
    result["blocks"] = [safety.mask_pii(b) for b in (result.get("blocks") or [])]
    return result


__all__ = ["ocr_document", "CONF_THRESHOLD"]
