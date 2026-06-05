"""YAML 配置读取工具。"""

import yaml

from utils.path_tool import get_abs_path


def load_yaml_config(config_path: str, encoding: str = "utf-8"):
    """读取 YAML 文件并返回字典。"""

    with open(config_path, "r", encoding=encoding) as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def load_milvus_config(config_path: str = get_abs_path("config/milvus.yml"), encoding="utf-8"):
    return load_yaml_config(config_path, encoding)


def load_prompts_config(config_path: str = get_abs_path("config/prompts.yml"), encoding="utf-8"):
    return load_yaml_config(config_path, encoding)


def load_rag_config(config_path: str = get_abs_path("config/rag.yml"), encoding="utf-8"):
    return load_yaml_config(config_path, encoding)


def load_agent_config(config_path: str = get_abs_path("config/agent.yml"), encoding="utf-8"):
    return load_yaml_config(config_path, encoding)


agent_conf = load_agent_config()
milvus_conf = load_milvus_config()
prompts_conf = load_prompts_config()
rag_conf = load_rag_config()
