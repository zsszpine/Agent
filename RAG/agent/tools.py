"""
Agent 可调用工具集合。

工具是 Agent 项目中最重要的扩展点之一：模型负责判断「什么时候调用」，
工具负责执行「确定性的外部能力」，例如检索知识库、查询业务数据、获取用户信息等。
"""

import csv
import ast
import base64
import hashlib
import json
import math
import os
import random
import re
import secrets
import string
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote, unquote
from zoneinfo import ZoneInfo

from langchain_core.tools import tool

from rag.agentic_rag_service import AgenticRagService
from rag.rag_service import RagService
from rag.vector_store import VectorStoreService
from utils.config_handler import agent_conf
from utils.file_handler import docx_loader, pdf_loader
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


rag: RagService | None = None
external_data: dict[str, dict[str, dict[str, str]]] = {}
KNOWLEDGE_ROOT = Path(get_abs_path("data")).resolve()
MAX_TOOL_TEXT_CHARS = 12000


def get_rag_service() -> AgenticRagService:
    """首次调用 RAG 工具时再连接向量库，避免服务启动强依赖 Milvus。"""

    global rag
    if rag is None:
        rag = AgenticRagService()
    return rag

# 示例项目里用随机数据模拟「用户系统」和「时间系统」。
# 真实业务中，这里通常会改成数据库、登录态、HTTP API 或设备管理平台。
USER_IDS = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010"]
MONTHS = [
    "2025-01",
    "2025-02",
    "2025-03",
    "2025-04",
    "2025-05",
    "2025-06",
    "2025-07",
    "2025-08",
    "2025-09",
    "2025-10",
    "2025-11",
    "2025-12",
]


@tool(description="从向量知识库中检索并总结与用户问题相关的资料。入参 query 是检索关键词或完整问题。")
def rag_summarize(query: str) -> str:
    """RAG 工具：把用户问题交给检索链，返回基于资料的总结。"""

    return get_rag_service().service.retrieve_docs(query)


@tool(description="获取当前日期和时间。timezone 默认 Asia/Shanghai，可传 UTC、Asia/Shanghai、America/New_York 等 IANA 时区名。")
def get_current_datetime(timezone: str = "Asia/Shanghai") -> str:
    """返回指定时区的当前时间。"""

    try:
        now = datetime.now(ZoneInfo(timezone))
    except Exception:
        return f"无效时区：{timezone}。请使用 IANA 时区名，例如 Asia/Shanghai 或 UTC。"
    return now.strftime("%Y-%m-%d %H:%M:%S %Z%z")


@tool(description="计算两个日期之间相差多少天。date 格式建议为 YYYY-MM-DD。")
def days_between(start_date: str, end_date: str) -> str:
    """计算日期差。"""

    try:
        start = datetime.fromisoformat(start_date).date()
        end = datetime.fromisoformat(end_date).date()
    except ValueError:
        return "日期格式错误，请使用 YYYY-MM-DD，例如 2026-04-29。"
    return str((end - start).days)


@tool(description="对日期加减天数。date 格式 YYYY-MM-DD，days 可为正数或负数。")
def date_offset(date: str, days: int) -> str:
    """返回日期加减指定天数后的结果。"""

    try:
        current = datetime.fromisoformat(date).date()
    except ValueError:
        return "日期格式错误，请使用 YYYY-MM-DD，例如 2026-04-29。"
    return (current + timedelta(days=days)).isoformat()


class _SafeCalculator(ast.NodeVisitor):
    """Small arithmetic evaluator for calculator tool."""

    allowed_binops = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a**b,
    }
    allowed_unary = {
        ast.UAdd: lambda a: +a,
        ast.USub: lambda a: -a,
    }
    allowed_names = {
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
    }
    allowed_funcs = {
        "abs": abs,
        "round": round,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "ceil": math.ceil,
        "floor": math.floor,
        "min": min,
        "max": max,
    }

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Constant(self, node):
        if isinstance(node.value, int | float):
            return node.value
        raise ValueError("只支持数字常量")

    def visit_Name(self, node):
        if node.id in self.allowed_names:
            return self.allowed_names[node.id]
        raise ValueError(f"不支持的名称：{node.id}")

    def visit_BinOp(self, node):
        op_type = type(node.op)
        if op_type not in self.allowed_binops:
            raise ValueError("不支持的运算符")
        return self.allowed_binops[op_type](self.visit(node.left), self.visit(node.right))

    def visit_UnaryOp(self, node):
        op_type = type(node.op)
        if op_type not in self.allowed_unary:
            raise ValueError("不支持的一元运算符")
        return self.allowed_unary[op_type](self.visit(node.operand))

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name) or node.func.id not in self.allowed_funcs:
            raise ValueError("不支持的函数")
        args = [self.visit(arg) for arg in node.args]
        return self.allowed_funcs[node.func.id](*args)

    def generic_visit(self, node):
        raise ValueError(f"不支持的表达式：{type(node).__name__}")


@tool(description="安全计算数学表达式。支持 + - * / // % **、括号、pi/e、sqrt/sin/cos/tan/log/round/min/max 等。")
def calculate(expression: str) -> str:
    """安全数学计算器，不执行 Python 代码。"""

    try:
        tree = ast.parse(expression, mode="eval")
        result = _SafeCalculator().visit(tree)
        return str(result)
    except Exception as exc:
        return f"计算失败：{exc}"


@tool(description="常用单位换算。支持长度 m/km/cm/mm/in/ft/yd/mi，质量 g/kg/lb/oz，温度 c/f/k，数据 b/kb/mb/gb/tb，时间 s/min/h/day。")
def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """常用单位转换。"""

    source = from_unit.strip().lower()
    target = to_unit.strip().lower()

    unit_groups = [
        {"m": 1, "km": 1000, "cm": 0.01, "mm": 0.001, "in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mi": 1609.344},
        {"g": 1, "kg": 1000, "lb": 453.59237, "oz": 28.349523125},
        {"b": 1, "kb": 1024, "mb": 1024**2, "gb": 1024**3, "tb": 1024**4},
        {"s": 1, "sec": 1, "min": 60, "h": 3600, "hr": 3600, "day": 86400},
    ]

    if source in {"c", "f", "k"} and target in {"c", "f", "k"}:
        celsius = value if source == "c" else (value - 32) * 5 / 9 if source == "f" else value - 273.15
        result = celsius if target == "c" else celsius * 9 / 5 + 32 if target == "f" else celsius + 273.15
        return f"{result:g} {to_unit}"

    for group in unit_groups:
        if source in group and target in group:
            result = value * group[source] / group[target]
            return f"{result:g} {to_unit}"

    return f"不支持从 {from_unit} 到 {to_unit} 的换算。"


@tool(description="统计文本信息，返回字符数、词数、行数、句子数、中文字符数、英文单词数等。")
def text_statistics(text: str) -> str:
    """统计文本基础指标。"""

    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    english_words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    sentences = re.findall(r"[^。！？.!?]+[。！？.!?]?", text)
    stats = {
        "字符数": len(text),
        "去空白字符数": len(re.sub(r"\s+", "", text)),
        "行数": 0 if not text else len(text.splitlines()),
        "句子数": len([item for item in sentences if item.strip()]),
        "中文字符数": len(chinese_chars),
        "英文单词数": len(english_words),
    }
    return json.dumps(stats, ensure_ascii=False)


@tool(description="提取文本中的邮箱、URL、手机号样式数字、日期和数字。kind 可选 email/url/phone/date/number/all。")
def extract_patterns(text: str, kind: str = "all") -> str:
    """从文本提取常见模式。"""

    patterns = {
        "email": r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+",
        "url": r"https?://[^\s<>\"]+",
        "phone": r"(?<![\d-])(?:\+?\d{1,3}[- ]?)?(?:1[3-9]\d{9}|\d{3,4}[- ]\d{7,8})(?![\d-])",
        "date": r"\b\d{4}[-/年]\d{1,2}(?:[-/月]\d{1,2}日?)?\b",
        "number": r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])",
    }
    selected = patterns.keys() if kind == "all" else [kind]
    result = {}
    for name in selected:
        if name not in patterns:
            return f"不支持的 kind：{kind}。可选 email/url/phone/date/number/all。"
        result[name] = re.findall(patterns[name], text)
    return json.dumps(result, ensure_ascii=False)


@tool(description="用正则表达式搜索文本。返回最多 max_matches 个匹配项。")
def regex_search(text: str, pattern: str, max_matches: int = 20) -> str:
    """正则搜索。"""

    try:
        matches = re.findall(pattern, text)
    except re.error as exc:
        return f"正则表达式错误：{exc}"
    limited = matches[: max(1, min(max_matches, 100))]
    return json.dumps(limited, ensure_ascii=False)


@tool(description="格式化并校验 JSON 字符串。indent 默认 2。")
def format_json(json_text: str, indent: int = 2) -> str:
    """JSON 格式化。"""

    try:
        data = json.loads(json_text)
        return json.dumps(data, ensure_ascii=False, indent=max(0, min(indent, 8)))
    except json.JSONDecodeError as exc:
        return f"JSON 解析失败：第 {exc.lineno} 行第 {exc.colno} 列，{exc.msg}"


@tool(description="提取 JSON 中的顶层字段或指定点号路径，例如 user.name 或 items.0.title。")
def json_get(json_text: str, path: str = "") -> str:
    """按点号路径读取 JSON。"""

    try:
        value = json.loads(json_text)
        if path.strip():
            for part in path.split("."):
                if isinstance(value, list):
                    value = value[int(part)]
                elif isinstance(value, dict):
                    value = value[part]
                else:
                    return "路径无法继续访问。"
        return json.dumps(value, ensure_ascii=False, indent=2)
    except Exception as exc:
        return f"JSON 读取失败：{exc}"


@tool(description="预览 CSV 文本，返回列名、行数和前 max_rows 行。")
def csv_preview(csv_text: str, max_rows: int = 5) -> str:
    """CSV 预览。"""

    try:
        rows = list(csv.DictReader(csv_text.splitlines()))
    except Exception as exc:
        return f"CSV 解析失败：{exc}"
    preview = {
        "columns": rows[0].keys() if rows else [],
        "row_count": len(rows),
        "preview": rows[: max(1, min(max_rows, 20))],
    }
    return json.dumps(preview, ensure_ascii=False, default=list)


@tool(description="对文本进行 Base64 或 URL 编码/解码。operation 可选 base64_encode/base64_decode/url_encode/url_decode。")
def encode_decode_text(text: str, operation: str) -> str:
    """编码解码工具。"""

    try:
        if operation == "base64_encode":
            return base64.b64encode(text.encode("utf-8")).decode("ascii")
        if operation == "base64_decode":
            return base64.b64decode(text).decode("utf-8")
        if operation == "url_encode":
            return quote(text)
        if operation == "url_decode":
            return unquote(text)
        return "不支持的 operation。可选 base64_encode/base64_decode/url_encode/url_decode。"
    except Exception as exc:
        return f"处理失败：{exc}"


@tool(description="计算文本哈希。algorithm 可选 md5/sha1/sha256/sha512。")
def hash_text(text: str, algorithm: str = "sha256") -> str:
    """文本哈希。"""

    algo = algorithm.lower()
    if algo not in {"md5", "sha1", "sha256", "sha512"}:
        return "不支持的 algorithm。可选 md5/sha1/sha256/sha512。"
    digest = hashlib.new(algo)
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


@tool(description="生成 UUID。count 为数量，范围 1-20。")
def generate_uuid(count: int = 1) -> str:
    """生成 UUID4。"""

    import uuid

    amount = max(1, min(count, 20))
    return json.dumps([str(uuid.uuid4()) for _ in range(amount)], ensure_ascii=False)


@tool(description="生成安全随机密码。length 范围 8-128，include_symbols 控制是否包含符号。")
def generate_password(length: int = 16, include_symbols: bool = True) -> str:
    """生成随机密码。"""

    size = max(8, min(length, 128))
    alphabet = string.ascii_letters + string.digits
    if include_symbols:
        alphabet += "!@#$%^&*()-_=+[]{}:,.?"
    return "".join(secrets.choice(alphabet) for _ in range(size))


@tool(description="随机数生成。返回 min_value 到 max_value 之间的整数，count 范围 1-100。")
def random_integers(min_value: int, max_value: int, count: int = 1) -> str:
    """生成随机整数。"""

    if min_value > max_value:
        min_value, max_value = max_value, min_value
    amount = max(1, min(count, 100))
    return json.dumps([random.randint(min_value, max_value) for _ in range(amount)], ensure_ascii=False)


@tool(description="对多行文本去重。keep_order 为 true 时保留首次出现顺序。")
def deduplicate_lines(text: str, keep_order: bool = True) -> str:
    """多行去重。"""

    lines = text.splitlines()
    if not keep_order:
        return "\n".join(sorted(set(lines)))
    seen = set()
    result = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            result.append(line)
    return "\n".join(result)


@tool(description="对多行文本排序。reverse 控制倒序，numeric 为 true 时按数字排序。")
def sort_lines(text: str, reverse: bool = False, numeric: bool = False) -> str:
    """多行排序。"""

    lines = text.splitlines()
    try:
        if numeric:
            lines.sort(key=lambda item: float(item.strip()), reverse=reverse)
        else:
            lines.sort(reverse=reverse)
        return "\n".join(lines)
    except ValueError:
        return "数字排序失败：存在无法转换为数字的行。"


def _list_knowledge_files_data() -> list[dict]:
    allowed_suffixes = {".txt", ".pdf", ".md", ".csv", ".docx"}
    files = []
    if not KNOWLEDGE_ROOT.exists():
        return files
    for item in KNOWLEDGE_ROOT.rglob("*"):
        if item.is_file() and item.suffix.lower() in allowed_suffixes:
            files.append(
                {
                    "path": str(item.relative_to(KNOWLEDGE_ROOT)),
                    "size": item.stat().st_size,
                }
            )
    return sorted(files, key=lambda file: file["path"])


@tool(description="列出知识库文件，返回 RAG/data 下 txt/pdf/md/csv/docx 文件的路径和大小。")
def list_knowledge_files() -> str:
    """列出知识库文件。"""

    return json.dumps(_list_knowledge_files_data(), ensure_ascii=False)


@tool(description="按文件名搜索知识库文件。keyword 是文件名关键词，返回匹配路径。")
def search_knowledge_files(keyword: str) -> str:
    """搜索知识库文件名。"""

    files = _list_knowledge_files_data()
    matched = [item for item in files if keyword.lower() in item["path"].lower()]
    return json.dumps(matched, ensure_ascii=False)


def _safe_knowledge_path(relative_path: str) -> Path:
    path = (KNOWLEDGE_ROOT / relative_path).resolve()
    if KNOWLEDGE_ROOT not in path.parents and path != KNOWLEDGE_ROOT:
        raise ValueError("路径超出知识库目录")
    if not path.is_file():
        raise ValueError("文件不存在")
    return path


@tool(description="读取知识库文件文本内容。relative_path 必须是 list_knowledge_files 返回的相对路径，支持 txt/md/csv/pdf/docx。")
def read_knowledge_file(relative_path: str, max_chars: int = 4000) -> str:
    """读取知识库文件内容，限制在 RAG/data 内。"""

    try:
        path = _safe_knowledge_path(relative_path)
        suffix = path.suffix.lower()
        limit = max(200, min(max_chars, MAX_TOOL_TEXT_CHARS))
        if suffix in {".txt", ".md", ".csv"}:
            text = path.read_text(encoding="utf-8", errors="replace")
        elif suffix == ".docx":
            docs = docx_loader(str(path))
            text = "\n\n".join(doc.page_content for doc in docs)
        elif suffix == ".pdf":
            docs = pdf_loader(str(path))
            text = "\n\n".join(doc.page_content for doc in docs)
        else:
            return "不支持的文件类型。"
        return text[:limit]
    except Exception as exc:
        return f"读取失败：{exc}"


@tool(description="获取指定城市的天气信息。入参 city 是城市名称，返回字符串形式的天气描述。")
def get_weather(city: str) -> str:
    """天气工具示例。这里用模拟数据演示工具调用流程。"""

    return f"{city}今天晴，温度和湿度适中，适合正常出行和室内活动。"


@tool(description="获取当前用户所在城市名称，无入参，返回纯字符串。")
def get_user_location() -> str:
    """用户位置工具示例。真实项目应从登录态或定位服务读取。"""

    return random.choice(["深圳", "成都", "上海"])


@tool(description="获取当前用户 ID，无入参，返回数字字符串，例如 1001。")
def get_user_id() -> str:
    """用户 ID 工具示例。真实项目应从认证系统读取。"""

    return random.choice(USER_IDS)


@tool(description="获取报告月份，无入参，返回 YYYY-MM 格式字符串。")
def get_month() -> str:
    """月份工具示例。这里固定在样例数据范围内随机选择。"""

    return random.choice(MONTHS)


def generate_external_data():
    """
    加载外部业务数据。

    records.csv 模拟一个业务系统，例如设备运行记录、耗材状态、清洁效率等。
    使用 csv.reader 可以正确处理带引号、逗号的字段，比手动 split 更稳。
    """

    if external_data:
        return

    path = get_abs_path(agent_conf["external_data_path"])
    if not os.path.exists(path):
        logger.error(f"外部数据文件不存在：{path}")
        return

    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            user_id = (row.get("user_id") or row.get("用户ID") or "").strip()
            month = (row.get("time") or row.get("month") or row.get("月份") or "").strip()
            if not user_id or not month:
                continue

            external_data.setdefault(user_id, {})[month] = {
                "特征": (row.get("feature") or row.get("特征") or "").strip(),
                "效率": (row.get("efficiency") or row.get("效率") or "").strip(),
                "耗材": (row.get("consumables") or row.get("耗材") or "").strip(),
                "对比": (row.get("comparision") or row.get("comparison") or row.get("对比") or "").strip(),
            }


@tool(description="从外部业务系统获取用户使用记录。入参 user_id 和 month，month 格式为 YYYY-MM。")
def fetch_external_data(user_id: str, month: str) -> str:
    """查询用户指定月份的模拟业务记录。"""

    generate_external_data()

    try:
        # 工具返回字符串更容易被大模型稳定消费，所以这里把结构化数据转成 JSON 字符串。
        return json.dumps(external_data[user_id][month], ensure_ascii=False)
    except KeyError:
        logger.warning(f"[fetch_external_data] 未检索到用户 {user_id} 在 {month} 的历史记录")
        return ""


@tool(description="报告生成前置工具。无入参，用于标记当前任务进入报告生成上下文。")
def fill_context_for_report() -> str:
    """报告生成上下文占位工具。真实项目可在这里切换提示词、加载模板或注入用户画像。"""

    return "已进入报告生成上下文。"

@tool(description="删除rag数据库文件")
def del_ragsum() -> str:
    try:
        VectorStoreService.clear_vector_db(clear_md5=True)
    except Exception as exc:
        raise exc
    return "milvus db cleared"
