"""
向量库服务。

RAG 的核心流程是：原始文档 -> Loader 读取文本 -> Splitter 切片
-> Embedding 向量化 -> Milvus 存储 -> Retriever 检索。
"""

import os
import sys

RAG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAG_ROOT not in sys.path:
    sys.path.insert(0, RAG_ROOT)

from langchain_core.documents import Document
from langchain_milvus import Milvus
from pymilvus import MilvusClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import model
from model.model_factory import embedding_model
from utils.config_handler import milvus_conf
from utils.file_handler import (
    docx_loader,
    get_file_md5_hex,
    listdir_with_allowed_type,
    pdf_loader,
    txt_loader,
)
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


class VectorStoreService:
    """管理 Milvus 向量库的写入和检索。"""

    @staticmethod
    def clear_vector_db(clear_md5: bool = True) -> dict:
        """删除 Milvus collection，并按需重置入库 md5 缓存。"""

        md5_path = os.path.abspath(get_abs_path(milvus_conf["md5_hex_store"]))
        rag_root = os.path.abspath(get_abs_path(""))

        if os.path.commonpath([rag_root, md5_path]) != rag_root:
            raise ValueError(f"Refusing to delete path outside RAG root: {md5_path}")

        collection_name = milvus_conf["collection_name"]
        client = MilvusClient(uri=milvus_conf["uri"], token=milvus_conf.get("token") or "")

        removed = {"collection": False, "md5_store": False}
        if client.has_collection(collection_name=collection_name):
            client.drop_collection(collection_name=collection_name)
            removed["collection"] = True

        if clear_md5 and os.path.exists(md5_path):
            os.remove(md5_path)
            removed["md5_store"] = True

        return {
            "uri": milvus_conf["uri"],
            "collection_name": collection_name,
            "md5_path": md5_path if clear_md5 else None,
            "removed": removed,
        }

    @staticmethod
    def clear_chroma_db(clear_md5: bool = True) -> dict:
        """兼容旧调用名；实际清理的是 Milvus collection。"""

        return VectorStoreService.clear_vector_db(clear_md5=clear_md5)

    def __init__(self):
        self.vector_store = Milvus(
            embedding_function=embedding_model,
            collection_name=milvus_conf["collection_name"],
            connection_args={
                "uri": milvus_conf["uri"],
                "token": milvus_conf.get("token") or "",
            },
            index_params=milvus_conf.get("index_params"),
            search_params=milvus_conf.get("search_params"),
            auto_id=milvus_conf.get("auto_id", True),
            drop_old=milvus_conf.get("drop_old", False),
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=milvus_conf["chunk_size"],
            chunk_overlap=milvus_conf["chunk_overlap"],
            separators=milvus_conf["separators"],
            length_function=len,
        )

    def get_retriever(self):
        """返回检索器。k 表示每次检索返回最相关的前 k 个文档切片。"""

        return self.vector_store.as_retriever(search_kwargs={"k": milvus_conf["k"]})

    def load_document(self):
        """将数据目录中的文件加载进 Milvus。"""

        def check_md5_hex(md5_for_check: str | None):
            md5_store_path = get_abs_path(milvus_conf["md5_hex_store"])
            if not md5_for_check:
                return False

            if not os.path.exists(md5_store_path):
                open(md5_store_path, "w", encoding="utf-8").close()
                return False

            with open(md5_store_path, "r", encoding="utf-8") as file:
                return any(line.strip() == md5_for_check for line in file)

        def save_md5_hex(md5_str: str | None):
            if not md5_str or check_md5_hex(md5_str):
                return

            with open(get_abs_path(milvus_conf["md5_hex_store"]), "a", encoding="utf-8") as file:
                file.write(md5_str + "\n")

        def get_file_documents(read_path: str) -> list[Document]:
            suffix = os.path.splitext(read_path)[1].lower()
            if suffix == ".pdf":
                return pdf_loader(read_path)
            if suffix == ".docx":
                return docx_loader(read_path)
            if suffix in {".txt", ".md", ".csv"}:
                return txt_loader(read_path)
            return []

        allowed_files_path = listdir_with_allowed_type(
            milvus_conf["data_path"],
            tuple(milvus_conf["allow_knowledge_file_type"]),
        )

        for file in allowed_files_path:
            md5_hex = get_file_md5_hex(file)
            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库] {file} 内容已经存在，跳过")
                continue

            try:
                documents = get_file_documents(file)
                if not documents:
                    logger.warning(f"[加载知识库] {file} 没有有效文本，跳过")
                    continue

                split_documents = self.splitter.split_documents(documents)
                if not split_documents:
                    logger.warning(f"[加载知识库] {file} 切片后没有有效文本，跳过")
                    continue

                self.vector_store.add_documents(split_documents)
                save_md5_hex(md5_hex)
                logger.info(f"[加载知识库] {file} 加载成功，切片数：{len(split_documents)}")
            except Exception as exc:
                logger.error(f"[加载知识库] {file} 加载失败：{exc}", exc_info=True)


if __name__ == "__main__":
    vs = VectorStoreService()
    vs.load_document()
    retriever = vs.get_retriever()
    for item in retriever.invoke("绿色物流"):
        print(item)
