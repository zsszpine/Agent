"""路径工具：为 RAG 子工程提供统一的绝对路径解析。"""

import os


def get_project_root() -> str:
    """获取 RAG 子工程根目录。"""

    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    return os.path.dirname(current_dir)


def get_abs_path(relative_path: str) -> str:
    """把相对于 RAG 目录的路径转换成绝对路径。"""

    return os.path.join(get_project_root(), relative_path)
