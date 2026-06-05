"""
FastAPI 项目入口。

这个文件负责三件事：
1. 暴露前端页面和静态资源；
2. 提供聊天、文件上传、知识库重建等 HTTP 接口；
3. 把 Web 请求转换成 GraphAgent / RAG 服务可以理解的函数调用。

学习 Agent 项目时建议先看这里，因为它是「浏览器 -> 后端 -> Agent -> 工具/RAG」链路的起点。
"""

from pathlib import Path
import os
import shutil
import sys
from typing import List
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn


# RAG 目录中的历史代码大量使用了 `from utils.xxx import ...` 这种导入方式。
# 把 RAG 目录加入 sys.path 后，可以在不大规模改动原项目结构的情况下正常导入。
PROJECT_ROOT = Path(__file__).resolve().parent
RAG_ROOT = PROJECT_ROOT / "RAG"
if str(RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(RAG_ROOT))

from agent.GraphAgent import GraphAgent  # noqa: E402
from rag.vector_store import VectorStoreService  # noqa: E402


STATIC_DIR = PROJECT_ROOT / "static"
KNOWLEDGE_DIR = RAG_ROOT / "data"
ALLOWED_UPLOAD_SUFFIXES = {".txt", ".pdf", ".md", ".csv", ".docx"}


app = FastAPI(
    title="Agent 学习项目",
    description="一个包含 LangGraph Agent、工具调用、RAG 检索和前端聊天界面的学习项目。",
    version="1.1.0",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    """聊天接口入参。thread_id 用来区分不同会话的记忆。"""

    message: str = Field(..., min_length=1, description="用户输入的问题")
    thread_id: str = Field("web-demo", description="LangGraph 会话 ID")


class ChatResponse(BaseModel):
    """聊天接口出参。reply 是最终展示给用户的文本。"""

    reply: str

class DeleteKnowledgeFileRequest(BaseModel):
    """Request body for deleting a user-uploaded knowledge file."""

    path: str = Field(..., min_length=1, description="Path returned by /knowledge/files")


def safe_upload_name(filename: str) -> str:
    """
    生成安全文件名，避免用户上传 `../xxx` 这类路径穿越文件名。

    实际项目中还可以进一步接入 MIME 校验、病毒扫描、文件大小限制等安全能力。
    """

    original = Path(filename or "upload.txt").name
    suffix = Path(original).suffix.lower()
    stem = Path(original).stem[:60] or "document"

    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型：{suffix or '无后缀'}，仅支持 txt/pdf/md/csv/docx",
        )

    return f"{stem}_{uuid4().hex[:8]}{suffix}"


def list_knowledge_files() -> list[dict]:
    """列出 RAG/data 下已经上传或内置的知识文件，供前端展示知识库状态。"""

    if not KNOWLEDGE_DIR.exists():
        return []

    result = []
    for item in KNOWLEDGE_DIR.rglob("*"):
        if item.is_file() and item.suffix.lower() in ALLOWED_UPLOAD_SUFFIXES:
            result.append(
                {
                    "name": item.name,
                    "path": str(item.relative_to(KNOWLEDGE_DIR)),
                    "size": item.stat().st_size,
                }
            )
    return sorted(result, key=lambda file: file["path"])


def resolve_uploaded_knowledge_file(relative_path: str) -> Path:
    """Resolve and validate a deletable file path under RAG/data/uploads."""

    normalized_path = relative_path.replace("\\", "/").strip().lstrip("/")
    target_path = (KNOWLEDGE_DIR / normalized_path).resolve()
    upload_root = (KNOWLEDGE_DIR / "uploads").resolve()

    try:
        target_path.relative_to(upload_root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="only uploaded files can be deleted") from exc

    if not target_path.is_file():
        raise HTTPException(status_code=404, detail="uploaded file not found")

    if target_path.suffix.lower() not in ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(status_code=400, detail="unsupported file type")

    return target_path


@app.get("/")
async def root():
    """返回前端单页应用。"""

    return FileResponse(STATIC_DIR / "agent.html")


@app.get("/health")
async def health():
    """健康检查接口：前端可用它判断服务是否启动成功。"""

    return {
        "status": "ok",
        "knowledge_files": len(list_knowledge_files()),
        "allowed_upload_suffixes": sorted(ALLOWED_UPLOAD_SUFFIXES),
    }


@app.get("/examples")
async def examples():
    """给前端提供推荐问题，方便学习和演示 Agent 能力。"""

    return {
        "items": [
            "扫地机器人建图不完整、地图错乱怎么办？",
            "潮湿天气会影响扫拖一体机器人使用吗？",
            "帮我生成一份本月机器人使用报告",
            "小户型应该如何选择扫地机器人？",
            "机器人边刷缠绕头发应该怎么处理？",
        ]
    }


@app.get("/knowledge/files")
async def knowledge_files():
    """查看当前知识库文件列表。"""

    return {"files": list_knowledge_files()}


@app.delete("/knowledge/files")
async def delete_knowledge_file(data: DeleteKnowledgeFileRequest):
    """Delete one uploaded source file, then rebuild Milvus from remaining files."""

    target_path = resolve_uploaded_knowledge_file(data.path)
    target_path.unlink()

    try:
        VectorStoreService.clear_vector_db(clear_md5=True)
        VectorStoreService().load_document()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"file deleted, but rebuild failed: {exc}") from exc

    return {
        "message": "uploaded file deleted",
        "deleted": str(target_path.relative_to(KNOWLEDGE_DIR)),
        "files": list_knowledge_files(),
    }


@app.post("/knowledge/rebuild")
async def rebuild_knowledge():
    """
    重新把 RAG/data 中的文件切片并写入向量库。

    注意：当前 VectorStoreService 通过 md5 文件做去重，所以重复点击通常不会重复入库。
    """

    try:
        VectorStoreService().load_document()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"知识库重建失败：{exc}") from exc
    return {"message": "知识库重建完成", "files": list_knowledge_files()}


@app.post("/knowledge/clear")
async def clear_knowledge_db():
    """Clear the Milvus collection and md5 ingest cache without deleting source files."""

    try:
        result = VectorStoreService.clear_vector_db(clear_md5=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"clear milvus db failed: {exc}") from exc
    return {"message": "milvus db cleared", **result}


@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    """
    上传知识文件并保存到 RAG/data/uploads。

    保存后会自动触发一次知识库重建，这样用户上传资料后可以立刻在聊天中检索。
    """

    if not files:
        raise HTTPException(status_code=400, detail="请至少选择一个文件")

    upload_dir = KNOWLEDGE_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    for file in files:
        safe_name = safe_upload_name(file.filename)
        target_path = upload_dir / safe_name

        with target_path.open("wb") as target:
            shutil.copyfileobj(file.file, target)

        saved_files.append({"name": safe_name, "size": target_path.stat().st_size})

    try:
        VectorStoreService().load_document()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"文件已保存，但写入向量库失败：{exc}",
        ) from exc

    return {
        "message": f"成功上传并索引 {len(saved_files)} 个文件",
        "files": saved_files,
    }


@app.post("/chat", response_model=ChatResponse)
async def agent(data: ChatRequest):
    """调用 LangGraph Agent，返回最后一条模型消息。"""

    try:
        graph_agent = GraphAgent()
        result = graph_agent.execute_values(data.message, thread_id=data.thread_id)
        reply = result["messages"][-1].content
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent 调用失败：{exc}") from exc

    return {"reply": reply or "模型没有返回内容，请稍后重试。"}


if __name__ == "__main__":
    # 直接运行 `python main.py` 时启动开发服务器。
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8000")))
