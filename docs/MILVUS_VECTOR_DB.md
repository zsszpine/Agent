# Milvus 向量数据库使用说明

本项目已经把 RAG 向量库从 Chroma 切换为 Milvus。知识库文件上传后，会经过文本读取、切片、Embedding 向量化，然后写入 Milvus collection；聊天时由 Retriever 从 Milvus 检索相关片段，再交给模型总结回答。

## 相关文件

- `RAG/rag/vector_store.py`：Milvus 写入、检索和清空逻辑。
- `RAG/config/milvus.yml`：Milvus 连接、collection、切片和检索参数。
- `requirements.txt`：Milvus 相关依赖，包含 `langchain-milvus` 和 `pymilvus`。
- `RAG/md5.text`：已入库源文件的 md5 缓存，用于避免重复写入。

## 安装依赖

```bash
pip install -r requirements.txt
```

如果只想单独安装 Milvus 相关包：

```bash
pip install langchain-milvus pymilvus
```

## 启动 Milvus 服务

当前默认配置连接 `http://localhost:19530`，因此启动后端前需要先保证 Milvus 服务可访问。可以使用本机 Docker、公司已有 Milvus 服务或远程测试环境；只要把服务地址写入 `RAG/config/milvus.yml` 的 `uri` 即可。

快速检查端口是否可访问：

```powershell
Test-NetConnection localhost -Port 19530
```

## 配置 Milvus

配置文件位于 `RAG/config/milvus.yml`。

```yaml
collection_name: agent
uri: ./RAG/rag/milvus_demo.db
token:
auto_id: true
drop_old: false
k: 3
```

常用配置说明：

- `collection_name`：Milvus collection 名称，本项目默认是 `agent`。
- `uri`：Milvus 连接地址。
- `token`：连接鉴权信息；本地 Milvus Lite 通常留空。
- `auto_id`：由 Milvus 自动生成主键，建议保持 `true`。
- `drop_old`：初始化时是否删除旧 collection，日常使用建议保持 `false`。
- `k`：每次检索返回的文档切片数量。

本地开发建议先启动一个 Milvus 服务，然后使用默认地址：

```yaml
uri: http://localhost:19530
```

如果 Milvus 部署在其他机器，把 `uri` 改成对应服务地址：

```yaml
uri: http://你的-milvus-host:19530
```

## 写入知识库

项目支持把 `txt/pdf/md/csv/docx` 文件写入 Milvus。文件来源主要有两种：

1. 通过前端上传文件，后端会保存到 `RAG/data/uploads` 并自动入库。
2. 把文件放到 `RAG/data` 目录后，调用重建接口或手动执行入库。

手动入库：

```bash
python RAG/rag/vector_store.py
```

通过 HTTP 接口重建：

```bash
curl -X POST http://127.0.0.1:8000/knowledge/rebuild
```

注意：`RAG/md5.text` 会记录已经入库过的文件 md5。相同文件重复执行入库时会被跳过。

## 检索流程

代码中通过 `VectorStoreService.get_retriever()` 获取 LangChain Retriever：

```python
from rag.vector_store import VectorStoreService

vector_store = VectorStoreService()
retriever = vector_store.get_retriever()
docs = retriever.invoke("你的问题")
```

聊天链路中不需要手动调用这段代码。用户提问需要知识库时，Agent 会调用 `rag_summarize` 工具，内部通过 `RAG/rag/rag_service.py` 从 Milvus 检索资料并生成回答。

## 清空向量库

清空 Milvus collection 和 md5 缓存：

```bash
curl -X POST http://127.0.0.1:8000/knowledge/clear
```

代码调用：

```python
from rag.vector_store import VectorStoreService

VectorStoreService.clear_vector_db(clear_md5=True)
```

清空后，源文件不会被删除。如果需要重新入库，调用 `/knowledge/rebuild` 即可。

## 切片与检索参数

`RAG/config/milvus.yml` 中的切片参数会影响检索效果：

```yaml
chunk_size: 200
chunk_overlap: 20
k: 3
```

- `chunk_size` 太小会丢上下文，太大会降低召回精度。
- `chunk_overlap` 用于保留切片之间的连续语义。
- `k` 越大，模型拿到的参考资料越多，但回答成本和噪声也会增加。

## 常见问题

1. 启动时报 `No module named 'langchain_milvus'`

   重新安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

2. 检索不到刚上传的文件

   先确认文件类型是否在 `allow_knowledge_file_type` 中，然后调用：

   ```bash
   curl -X POST http://127.0.0.1:8000/knowledge/rebuild
   ```

3. 修改了 Embedding 模型后检索质量变差

   Embedding 模型必须和已入库向量保持一致。更换 `RAG/config/rag.yml` 中的 `embedding_model_name` 后，建议清空向量库并重新入库。

4. 想切换到远程 Milvus

   修改 `RAG/config/milvus.yml`：

   ```yaml
   uri: http://你的-milvus-host:19530
   token: 用户名:密码
   ```

   然后重启后端服务。
