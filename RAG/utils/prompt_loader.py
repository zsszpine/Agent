"""提示词读取工具。"""

from utils.config_handler import prompts_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


def read_prompt(config_key: str) -> str:
    """根据 prompts.yml 中的 key 读取提示词文件。"""

    try:
        prompt_path = get_abs_path(prompts_conf[config_key])
    except KeyError as exc:
        logger.error(f"[read_prompt] prompts.yml 中缺少配置项：{config_key}")
        raise exc

    try:
        with open(prompt_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as exc:
        logger.error(f"[read_prompt] 读取提示词失败：{prompt_path}，{exc}")
        raise exc


def load_main_prompts():
    return read_prompt("main_prompt_path")


def load_rag_prompts():
    return read_prompt("rag_summarize_prompt_path")


def load_report_prompts():
    return read_prompt("report_prompt_path")
