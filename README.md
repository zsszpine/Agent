# Agent 学习项目

这是一个用于学习 Agent 的小型完整项目，包含 FastAPI 后端、LangGraph Agent、工具调用、RAG 知识库检索、Milvus 向量库和前端聊天工作台。

## 启动方式

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 配置环境变量：

```bash
set MIMO_API_KEY=你的_mimo_key
set DASHSCOPE_API_KEY=你的_dashscope_key
```

3. 启动服务：

```bash
python main.py
```

4. 打开浏览器访问：

```text
http://127.0.0.1:8000
```

## 已完善的功能

- 前端工作台：聊天、示例问题、上传知识文件、知识库文件列表、服务状态、明暗主题、复制会话、导出会话、清空会话、消息统计、操作日志。
- 后端接口：`/chat`、`/upload`、`/health`、`/examples`、`/knowledge/files`、`/knowledge/rebuild`。
- 上传入库：上传的 `txt/pdf/md/csv` 会保存到 `RAG/data/uploads`，并自动写入 Milvus 向量库。
- Agent 编排：使用 LangGraph 实现「模型 -> 工具 -> 模型」循环。
- 工具调用：包含 RAG 检索、天气、用户位置、用户 ID、月份、外部业务数据、报告上下文工具。
- 学习注释：关键 Python 模块和前端 CSS/JS 都补充了注释。

## 做 Agent 需要掌握的技能

### 1. Web API 与前后端交互

你需要理解浏览器如何调用后端接口、后端如何校验请求、如何返回 JSON。

项目位置：
- `main.py`：FastAPI 入口，定义所有 HTTP 接口。
- `static/agent.html`：前端页面，通过 `fetch` 调用后端接口。

难点：
- 文件上传要处理安全文件名、允许类型、保存目录。
- 后端异常不能直接崩溃，要转成前端能理解的错误信息。

### 2. LangChain 消息模型

Agent 不是简单字符串输入输出，而是由 `SystemMessage`、`HumanMessage`、模型消息、工具消息组成的消息序列。

项目位置：
- `RAG/agent/GraphAgent.py` 的 `build_messages`。
- `RAG/prompts/main_prompt.txt`。

重要点：
- 系统提示词决定 Agent 的角色、边界和工具使用策略。
- 用户消息只表达当前请求，长期上下文由 LangGraph checkpointer 维护。

### 3. LangGraph 工作流编排

LangGraph 用图来表达 Agent 的执行流程。

项目位置：
- `RAG/agent/GraphAgent.py` 的 `compile_graph`。

重要点：
- `agent` 节点负责调用大模型。
- `tools` 节点负责执行工具。
- `should_continue` 决定模型是否继续调用工具。
- `MemorySaver` 用 `thread_id` 区分会话。

难点：
- 工具调用不是每次都发生，必须由模型根据提示词和工具描述判断。
- 工具返回后还要再次交给模型整合，不能直接把工具原始数据丢给用户。

### 4. Tool Calling 工具设计

工具是 Agent 连接外部世界的入口。

项目位置：
- `RAG/agent/tools.py`。

重要点：
- 工具描述要清晰说明入参、出参和使用场景。
- 工具内部最好做确定性工作，例如查数据库、查文件、调用业务 API。
- 工具返回值建议是简洁字符串或 JSON 字符串，方便模型理解。

难点：
- 工具太多会增加模型选择成本。
- 工具描述含糊会导致模型乱调工具或漏调工具。

### 5. RAG 检索增强生成

RAG 用于让模型基于项目知识库回答，而不是只靠模型记忆。

项目位置：
- `RAG/rag/vector_store.py`：文档读取、切片、向量入库、检索器。
- `RAG/rag/rag_service.py`：检索资料并调用模型总结。
- `RAG/prompts/rag_summarize.txt`：约束模型必须基于参考资料回答。
- `RAG/config/milvus.yml`：Milvus、切片和检索参数。

重要点：
- Loader 负责把 PDF/TXT/MD/CSV 读成文本。
- Splitter 负责把长文档切成适合检索的小片段。
- Embedding 模型负责把文本转成向量。
- Retriever 根据用户问题找最相关片段。

难点：
- chunk 太小会丢上下文，太大会降低召回精度。
- 知识库资料不足时，要让模型明确说明不足，避免编造。

### 6. 模型抽象与供应商切换

实际项目里经常需要切换不同模型或同时使用多个模型。

项目位置：
- `RAG/model/model_factory.py`。
- `RAG/config/rag.yml`。

重要点：
- 主 Agent 模型需要支持工具调用。
- RAG 总结模型可以和主 Agent 模型不同。
- Embedding 模型必须和向量库数据保持一致，随意更换会影响检索质量。

### 7. 提示词工程

提示词不是简单写角色设定，而是写清楚任务边界、工具使用规则和输出格式。

项目位置：
- `RAG/prompts/main_prompt.txt`。
- `RAG/prompts/rag_summarize.txt`。
- `RAG/prompts/report_prompt.txt`。
- `RAG/utils/prompt_loader.py`。

难点：
- 提示词越长不一定越好，关键是约束明确。
- 工具顺序、禁用场景、输出格式要具体。

### 8. 数据与业务系统模拟

Agent 常见价值是把模型和业务数据连接起来。

项目位置：
- `RAG/data/external/records.csv`。
- `RAG/agent/tools.py` 的 `fetch_external_data`。

重要点：
- CSV 在这里模拟外部业务系统。
- 真实项目可替换成数据库、CRM、设备管理平台或内部 HTTP API。

### 9. 前端产品化体验

Agent 项目不能只有命令行，前端需要展示状态、错误、上下文和可操作入口。

项目位置：
- `static/agent.html`。

重要点：
- 服务状态让用户知道后端是否在线。
- 示例问题降低试用成本。
- 知识库列表让用户知道上传是否成功。
- 操作日志帮助调试。

## 推荐继续扩展的功能

- 流式输出：把 `GraphAgent.execute_stream` 接到 FastAPI `StreamingResponse`。
- 会话列表：把 `thread_id` 保存到数据库，支持多个历史会话。
- 用户登录：用真实用户 ID 替换随机 `get_user_id`。
- 真实天气：把 `get_weather` 接入天气 API。
- 工具调用轨迹：前端展示 Agent 调用了哪些工具、耗时多少。
- 知识库删除：支持删除上传文件并重建向量库。
- 文档预览：上传后展示文本切片和 metadata。
- 评测集：准备标准问题，自动测试 RAG 回答质量。
- 权限控制：限制不同用户只能访问自己的文件和报告。
- Docker 部署：固定 Python 版本、依赖和启动命令。

## 阅读顺序

1. `main.py`
2. `static/agent.html`
3. `RAG/agent/GraphAgent.py`
4. `RAG/agent/tools.py`
5. `RAG/rag/vector_store.py`
6. `RAG/rag/rag_service.py`
7. `RAG/model/model_factory.py`
8. `RAG/prompts/*.txt`
