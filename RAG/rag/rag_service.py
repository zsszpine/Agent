"""
RAG 总结服务。

职责：
1. 根据用户问题检索向量库；
2. 把检索到的文档拼成上下文；
3. 调用大模型基于上下文生成回答。
"""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from model.model_factory import Mimo_model,chat_model
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts


class RagService:
    """封装「检索 + 总结」链路。"""

    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = Mimo_model
        self.chain = self.__init_chain()

    def __init_chain(self):
        """
        LCEL 链式写法：
        PromptTemplate 负责填充变量，model 负责生成，StrOutputParser 负责把模型消息转成字符串。
        """

        return self.prompt_template | self.model | StrOutputParser()

    def retrieve_docs(self, query: str) -> list[Document]:
        """从向量库中检索相关文档切片。"""

        return self.retriever.invoke(query)

    def rag_summarize(self, query: str) -> str:
        """基于检索资料总结答案。"""

        context_docs = self.retrieve_docs(query)
        context_parts = []
        for index, doc in enumerate(context_docs, start=1):
            context_parts.append(
                f"[参考资料{index}]\n"
                f"内容：{doc.page_content}\n"
                f"元数据：{doc.metadata}\n"
            )

        return self.chain.invoke(
            {
                "input": query,
                "context": "\n".join(context_parts),
            }
        )
