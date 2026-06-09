"""
模型工厂。

把「创建模型」集中放在一个文件里，可以让业务代码不关心具体供应商。
后续如果要从通义千问切到 OpenAI、Ollama 或其他模型，优先改这里。
"""

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Optional

from langchain_community.chat_models.tongyi import BaseChatModel, ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.messages import AIMessage
from langchain_core.embeddings import Embeddings
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models import base as openai_chat_base

from rag.multimodal import decode_multimodal_page, image_to_data_url
from utils.config_handler import rag_conf


def patch_openai_message_reasoning_content() -> None:
    """Preserve Mimo/Qwen-style reasoning_content across tool-call turns."""

    if getattr(openai_chat_base, "_mimo_reasoning_content_patched", False):
        return

    original_dict_to_message = openai_chat_base._convert_dict_to_message
    original_message_to_dict = openai_chat_base._convert_message_to_dict

    def convert_dict_to_message_with_reasoning(message_dict):
        message = original_dict_to_message(message_dict)
        if isinstance(message, AIMessage) and "reasoning_content" in message_dict:
            message.additional_kwargs["reasoning_content"] = message_dict[
                "reasoning_content"
            ]
        return message

    def convert_message_to_dict_with_reasoning(message, api="chat/completions"):
        message_dict = original_message_to_dict(message, api)
        if isinstance(message, AIMessage):
            reasoning_content = message.additional_kwargs.get("reasoning_content")
            if reasoning_content is not None:
                message_dict["reasoning_content"] = reasoning_content
        return message_dict

    openai_chat_base._convert_dict_to_message = convert_dict_to_message_with_reasoning
    openai_chat_base._convert_message_to_dict = convert_message_to_dict_with_reasoning
    openai_chat_base._mimo_reasoning_content_patched = True


patch_openai_message_reasoning_content()


class BaseModelFactory(ABC):
    """所有模型工厂的公共接口。"""

    @abstractmethod
    def generator(self) -> Optional[BaseChatModel | Embeddings]:
        pass


class ChatModelFactory(BaseModelFactory):
    """创建 RAG 总结使用的聊天模型。"""

    def generator(self) -> Optional[BaseChatModel | Embeddings]:
        return ChatTongyi(model=rag_conf["chat_model_name"])


class EmbeddingModelFactory(BaseModelFactory):
    """创建向量化模型。"""

    def generator(self) -> Optional[BaseChatModel | Embeddings]:
        if rag_conf.get("embedding_provider", "dashscope") == "gme":
            return GMEEmbeddings(
                model=rag_conf["embedding_model_name"],
                dimension=int(rag_conf.get("embedding_dimension", 1536)),
                endpoint=rag_conf.get("gme_embedding_endpoint"),
            )
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])


class GMEEmbeddings(Embeddings):
    """HTTP wrapper for a 1536-dimension multimodal GME embedding endpoint."""

    def __init__(self, model: str, dimension: int = 1536, endpoint: str | None = None):
        self.model = model
        self.dimension = dimension
        self.endpoint = endpoint or os.environ.get("GME_EMBEDDING_ENDPOINT")
        self.api_key = os.environ.get("GME_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
        if not self.endpoint:
            base_url = os.environ.get("GME_BASE_URL", "https://api.openai.com/v1")
            self.endpoint = base_url.rstrip("/") + "/embeddings"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)

    def _embed_one(self, text: str) -> list[float]:
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(self._build_payload(text), ensure_ascii=False).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GME embedding request failed: {exc.code} {body}") from exc

        vector = self._extract_embedding(response_data)
        if len(vector) != self.dimension:
            raise ValueError(
                f"GME embedding dimension mismatch: expected {self.dimension}, got {len(vector)}"
            )
        return vector

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_payload(self, text: str) -> dict:
        payload = decode_multimodal_page(text)
        if payload:
            content = []
            page_text = payload.get("text")
            image_path = payload.get("image_path")
            if page_text:
                content.append({"type": "text", "text": page_text})
            if image_path:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(image_path)},
                    }
                )
            input_value = [{"content": content}]
        else:
            input_value = [text]

        return {
            "model": self.model,
            "input": input_value,
            "dimensions": self.dimension,
        }

    @staticmethod
    def _extract_embedding(response_data: dict) -> list[float]:
        if "data" in response_data:
            first = response_data["data"][0]
            if isinstance(first, dict) and "embedding" in first:
                return first["embedding"]
        if "embedding" in response_data:
            return response_data["embedding"]
        if "embeddings" in response_data:
            return response_data["embeddings"][0]
        raise ValueError(f"Unsupported GME embedding response: {response_data}")


class Mimo(BaseModelFactory):
    """创建支持工具调用的主 Agent 模型。"""

    def generator(self) -> Optional[BaseChatModel | Embeddings]:
        return ChatOpenAI(
            model="mimo-v2-pro",
            api_key=os.environ.get("MIMO_API_KEY"),
            base_url="https://api.xiaomimimo.com/v1",
        )


class LocalModel(BaseModelFactory):
    """本地 Ollama 模型示例。需要先在本机启动 Ollama 并 pull 对应模型。"""

    def generator(self) -> Optional[BaseChatModel | Embeddings]:
        return ChatOllama(
            model="qwen3:4b",
            base_url="http://localhost:11434",
            temperature=0,
        )


chat_model = ChatModelFactory().generator()
embedding_model = EmbeddingModelFactory().generator()
Mimo_model = Mimo().generator()
local_model = LocalModel().generator()
