"""RAG summary service with multimodal Mimo input."""

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from model.model_factory import Mimo_model
from rag.multimodal import build_context_text, build_multimodal_human_content
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts


class RagService:
    """Encapsulate retrieval and multimodal answer generation."""

    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_text = load_rag_prompts()
        self.model = Mimo_model

    def retrieve_docs(self, query: str) -> list[Document]:
        """Retrieve relevant documents from Milvus."""

        return self.retriever.invoke(query)

    def rag_summarize(self, query: str) -> str:
        """Generate an answer using retrieved text plus rendered PDF page images."""

        context_docs = self.retrieve_docs(query)
        prompt = self.prompt_text.format(
            input=query,
            context=build_context_text(context_docs),
        )
        response = self.model.invoke(
            [HumanMessage(content=build_multimodal_human_content(prompt, context_docs))]
        )
        return response.content
