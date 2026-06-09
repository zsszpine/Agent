"""文件读取工具函数。"""

import hashlib
import os
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from rag.multimodal import encode_multimodal_page
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


def get_file_md5_hex(filepath: str):
    """计算文件 md5，用于判断知识文件是否已经入库。"""

    if not os.path.isfile(filepath):
        logger.error(f"[md5计算] 文件不存在：{filepath}")
        return None

    md5_obj = hashlib.md5()
    chunk_size = 4096

    try:
        with open(filepath, "rb") as file:
            while chunk := file.read(chunk_size):
                md5_obj.update(chunk)
        return md5_obj.hexdigest()
    except Exception as exc:
        logger.error(f"计算文件 {filepath} md5 失败：{exc}")
        return None


def listdir_with_allowed_type(path: str, allowed_types: tuple[str, ...]) -> tuple[str, ...]:
    """
    递归列出目录下允许类型的文件。

    上传文件会放到 data/uploads 子目录，所以这里必须递归扫描。
    """

    files = []
    abs_path = get_abs_path(path)
    normalized_types = tuple(
        item if item.startswith(".") else f".{item}" for item in allowed_types
    )

    if not os.path.isdir(abs_path):
        logger.error(f"[listdir_with_allowed_type] 不是文件夹：{abs_path}")
        return tuple()

    for root, _, filenames in os.walk(abs_path):
        for filename in filenames:
            if filename.lower().endswith(normalized_types):
                files.append(os.path.join(root, filename))

    return tuple(files)


def pdf_loader(filepath: str, pwd=None):
    """读取 PDF 文件为 LangChain Document。"""

    return PyPDFLoader(filepath, pwd).load()


def pdf_page_image_loader(filepath: str, dpi: int = 500) -> list[Document]:
    """Render each PDF page to PNG and return one multimodal Document per page."""

    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for multimodal PDF ingestion") from exc

    pdf_path = Path(filepath).resolve()
    render_root = Path(get_abs_path("data/.rendered_pages"))
    file_render_dir = render_root / pdf_path.stem / str(get_file_md5_hex(str(pdf_path)))
    file_render_dir.mkdir(parents=True, exist_ok=True)

    documents: list[Document] = []
    with fitz.open(str(pdf_path)) as pdf:
        for page_index in range(pdf.page_count):
            page = pdf.load_page(page_index)
            image_path = file_render_dir / f"page_{page_index + 1:04d}_{dpi}dpi.png"

            if not image_path.exists():
                pixmap = page.get_pixmap(dpi=dpi, alpha=False)
                pixmap.save(str(image_path))

            text = page.get_text("text").strip()
            metadata = {
                "source": str(pdf_path),
                "page": page_index,
                "page_number": page_index + 1,
                "image_path": str(image_path),
                "dpi": dpi,
                "modality": "pdf_page_image",
            }
            documents.append(
                Document(
                    page_content=encode_multimodal_page(text, str(image_path), metadata),
                    metadata=metadata,
                )
            )

    return documents


def txt_loader(filepath: str, pwd=None):
    """读取纯文本类文件为 LangChain Document。md/csv 在本项目中也按文本读取。"""

    return TextLoader(file_path=filepath, encoding="utf-8").load()


def docx_loader(filepath: str, pwd=None):
    """读取 DOCX 文件为 LangChain Document。"""

    try:
        with zipfile.ZipFile(filepath) as docx_file:
            xml_content = docx_file.read("word/document.xml")
    except Exception as exc:
        logger.error(f"读取 DOCX 文件 {filepath} 失败：{exc}")
        return []

    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    root = ElementTree.fromstring(xml_content)
    paragraphs = []

    for paragraph in root.findall(".//w:p", namespace):
        text_parts = [
            node.text
            for node in paragraph.findall(".//w:t", namespace)
            if node.text
        ]
        paragraph_text = "".join(text_parts).strip()
        if paragraph_text:
            paragraphs.append(paragraph_text)

    text = "\n".join(paragraphs).strip()
    if not text:
        return []

    return [Document(page_content=text, metadata={"source": filepath})]


