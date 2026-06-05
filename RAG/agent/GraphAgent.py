"""
LangGraph Agent 编排层。

这个文件展示了一个最小但完整的 Agent 工作流：
用户消息 -> 大模型思考 -> 判断是否需要工具 -> 工具执行 -> 再次交给模型总结 -> 返回用户。

学习重点：
- StateGraph：定义 Agent 节点和边；
- ToolNode：把 LangChain tool 统一包装成图节点；
- MemorySaver：用 thread_id 保存同一个会话的上下文；
- bind_tools：让模型知道当前可以调用哪些工具。
"""
import json
from datetime import time
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage, ToolMessage
from langchain_core.messages.base import message_to_dict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from agent.tools import (
    calculate,
    convert_units,
    csv_preview,
    date_offset,
    days_between,
    deduplicate_lines,
    encode_decode_text,
    extract_patterns,
    fetch_external_data,
    fill_context_for_report,
    format_json,
    generate_password,
    generate_uuid,
    get_current_datetime,
    get_month,
    get_user_id,
    get_user_location,
    get_weather,
    hash_text,
    json_get,
    list_knowledge_files,
    rag_summarize,
    random_integers,
    read_knowledge_file,
    regex_search,
    search_knowledge_files,
    sort_lines,
    text_statistics,
)
from model.model_factory import Mimo_model,chat_model
from utils.prompt_loader import load_main_prompts

WORKDIR = Path.cwd()
TOOL_RESULTS_DIR = WORKDIR / ".task_outputs" / "tool-results"
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
TOOL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

class GraphAgent:
    """封装 Agent 图的构建和执行。"""

    def __init__(self):
        # 工具越多，模型越需要清晰的工具描述。工具描述写得好，Agent 调用准确率会明显提升。
        self.tools = [
            rag_summarize,
            get_current_datetime,
            days_between,
            date_offset,
            calculate,
            convert_units,
            text_statistics,
            extract_patterns,
            regex_search,
            format_json,
            json_get,
            csv_preview,
            encode_decode_text,
            hash_text,
            generate_uuid,
            generate_password,
            random_integers,
            deduplicate_lines,
            sort_lines,
            list_knowledge_files,
            search_knowledge_files,
            read_knowledge_file,
            get_weather,
            get_user_location,
            get_user_id,
            get_month,
            fetch_external_data,
            fill_context_for_report,
        ]
        self.model = Mimo_model.bind_tools(self.tools)
        self.tool_node = ToolNode(self.tools)

    def write_transcript(self, messages):
        """L4前置动作：保存对话快照落盘"""
        p = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for m in messages:
                f.write(json.dumps({"role": m.type, "content": str(m.content)}, default=str) + "\n")
        return p

    def compile_graph(self):
        """构建 LangGraph 执行图。"""


        def compress_context_node(state: MessagesState):
            """
            四层压缩管线预处理器 (0 API 调用，除L4外)
            严格按照 L3 -> L1 -> L2 -> L4 顺序执行。
            """
            # 我们在内存中复制一份列表来模拟原始的顺序处理，
            # 并在过程中收集所有需要对 LangGraph 底层状态做出的“修改指令”（updates）。
            current_msgs = list(state["messages"])
            updates_and_removes = []

            # ---------------------------------------------------------
            # L3: tool_result_budget — 大结果落盘 (最先执行)
            # ---------------------------------------------------------
            tool_msgs = [m for m in current_msgs if isinstance(m, ToolMessage)]
            total_tool_size = sum(len(str(m.content)) for m in tool_msgs)

            if total_tool_size > 200_000:
                print("\n\033[33m[L3 Budget: 触发大工具结果落盘]\033[0m")
                # 倒序排列，最大的优先落盘
                ranked_tools = sorted(tool_msgs, key=lambda x: len(str(x.content)), reverse=True)
                for tm in ranked_tools:
                    if total_tool_size <= 200_000:
                        break

                    content_str = str(tm.content)
                    if len(content_str) <= 2000:
                        continue  # 太小的没必要落盘

                    # 落盘
                    file_id = tm.tool_call_id or str(tm.id)
                    p = TOOL_RESULTS_DIR / f"{file_id}.txt"
                    p.write_text(content_str, encoding="utf-8")

                    # 生成占位符
                    new_content = f"<persisted-output>\nFull: {p}\nPreview:\n{content_str[:2000]}\n</persisted-output>"

                    # 关键：生成一条具有相同 ID 的新消息，LangGraph 会自动覆盖原消息
                    new_tm = ToolMessage(content=new_content, tool_call_id=tm.tool_call_id, id=tm.id, name=tm.name)
                    updates_and_removes.append(new_tm)

                    # 在当前虚拟列表中也更新，防止后续层重复计算
                    idx = current_msgs.index(tm)
                    current_msgs[idx] = new_tm
                    total_tool_size -= (len(content_str) - len(new_content))

            # ---------------------------------------------------------
            # L1: snip_compact — 裁掉无关的旧对话
            # ---------------------------------------------------------
            if len(current_msgs) > 50:
                print("\n\033[33m[L1 Snip: 触发中间对话截断]\033[0m")
                keep_head = 3
                keep_tail = 47
                snipped_count = len(current_msgs) - keep_head - keep_tail

                # 我们将中间截断的第一条消息覆盖为占位符 (保留其ID，改变类型和内容)
                first_snipped_msg = current_msgs[keep_head]
                placeholder = SystemMessage(
                    content=f"[snipped {snipped_count} messages from conversation middle]",
                    id=first_snipped_msg.id
                )
                updates_and_removes.append(placeholder)

                # 删除其余被截断的消息
                for m in current_msgs[keep_head + 1: -keep_tail]:
                    if m.id:
                        updates_and_removes.append(RemoveMessage(id=m.id))

                # 更新虚拟列表以供下一层使用
                current_msgs = current_msgs[:keep_head] + [placeholder] + current_msgs[-keep_tail:]

            # ---------------------------------------------------------
            # L2: micro_compact — 旧工具结果占位
            # ---------------------------------------------------------
            current_tool_msgs = [m for m in current_msgs if isinstance(m, ToolMessage)]
            if len(current_tool_msgs) > 3:
                # 只处理除了最后3个之外的旧工具结果
                for tm in current_tool_msgs[:-3]:
                    content_str = str(tm.content)
                    # 如果不是已经被 L3 落盘过的，且长度大于120
                    if len(content_str) > 120 and not content_str.startswith("<persisted-output>"):
                        print(f"\n\033[33m[L2 Micro: 压缩旧工具结果 (ID: {tm.id})]\033[0m")
                        new_content = "[Earlier tool result compacted. Re-run if needed.]"
                        new_tm = ToolMessage(content=new_content, tool_call_id=tm.tool_call_id, id=tm.id,
                                             name=tm.name)
                        updates_and_removes.append(new_tm)

                        idx = current_msgs.index(tm)
                        current_msgs[idx] = new_tm

            # ---------------------------------------------------------
            # L4: compact_history — LLM 全量摘要
            # ---------------------------------------------------------
            # 字符数估算 token (中文/复杂场景通常除以 3 或 4)
            char_count = sum(len(str(m.content)) for m in current_msgs)
            if char_count > 80_000:  # 约 20,000 Token 阈值
                print("\n\033[35m[L4 Auto Compact: 触发展望摘要 (LLM调用)]\033[0m")
                self.write_transcript(current_msgs)

                # 让大模型做摘要
                summary_prompt = "Summarize this coding-agent conversation so work can continue.\nPreserve: 1. current goal, 2. key findings, 3. files changed, 4. remaining work, 5. user constraints.\n\n"
                for m in current_msgs[-10:]:  # 取最近几轮
                    summary_prompt += f"{m.type}: {m.content}\n"

                # 注意：生产中这里应该用便宜/快速的模型 (如 gpt-4o-mini 或 qwen-plus)
                summary_result = self.model.invoke(summary_prompt).content

                # L4 是核武器，一旦触发，清空当前所有积累的操作指令，直接进行全量替换
                l4_updates = []
                for m in current_msgs:
                    if m.type != "system" and m.id:  # 保留初始系统提示词
                        l4_updates.append(RemoveMessage(id=m.id))

                l4_updates.append(HumanMessage(content=f"[Compacted]\n\n{summary_result}"))
                return {"messages": l4_updates}

            # 如果没有触发 L4，则返回 L1~L3 积累的精准修改指令
            return {"messages": updates_and_removes}

        def call_model(state: MessagesState):
            """Agent 节点：包含应急响应 (Reactive Compact)"""
            messages = state["messages"]

            try:
                response = self.model.invoke(messages)
                return {"messages": [response]}
            except Exception as e:
                # ---------------------------------------------------------
                # 应急: reactive_compact — 兜底熔断机制
                # ---------------------------------------------------------
                if "prompt_too_long" in str(e).lower() or "too many tokens" in str(
                        e).lower() or "maximum context" in str(e).lower():
                    print("\n\033[41m[Reactive Compact: API 拒绝服务，触发强制截断!]\033[0m")
                    self.write_transcript(messages)

                    # 极端的截断方案：只保留 System 提示词和最后 4 条消息
                    removes = []
                    for m in messages[:-4]:
                        if m.type != "system" and m.id:
                            removes.append(RemoveMessage(id=m.id))

                    fallback_msg = HumanMessage(
                        content="[Reactive compact performed due to API limits. Some immediate context was lost.]")
                    return {"messages": removes + [fallback_msg]}

                # 其他异常抛出
                raise

        def should_continue(state: MessagesState):
            """
            条件边：如果模型返回 tool_calls，就进入 tools 节点；否则结束图执行。
            """

            message = state["messages"][-1]
            if message.tool_calls:
                return "tools"
            return END

        builder = StateGraph(MessagesState)
        builder.add_node("compress_pipeline", compress_context_node)
        builder.add_node("agent", call_model)
        builder.add_node("tools", self.tool_node)

        builder.add_edge(START, "compress_pipeline")
        builder.add_edge("compress_pipeline", "agent")
        builder.add_conditional_edges("agent", should_continue)
        # 工具执行完后，必须再次经过压缩管线（防止刚执行的工具拉取了10MB的数据）
        builder.add_edge("tools", "compress_pipeline")

        # MemorySaver 是内存级会话记忆。生产环境通常会换成 Redis、Postgres 等持久化 checkpointer。
        return builder.compile(checkpointer=MemorySaver())

    def build_messages(self, query: str):
        """把系统提示词和用户问题组合成 LangChain messages。"""

        return {
            "messages": [
                SystemMessage(content=load_main_prompts()),
                HumanMessage(content=query),
            ]
        }

    def execute_stream(self, query: str, thread_id: str = "web-demo"):
        """流式执行示例。当前前端未使用，保留给后续改造成流式输出。"""

        graph = self.compile_graph()
        config = {"configurable": {"thread_id": thread_id}}

        for chunk in graph.stream(self.build_messages(query), config, stream_mode="values"):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"

    def execute_values(self, query: str, thread_id: str = "web-demo"):
        """普通阻塞式执行：等待 Agent 完整回答后一次性返回。"""

        graph = self.compile_graph()
        config = {"configurable": {"thread_id": thread_id}}
        return graph.invoke(self.build_messages(query), config)

    def execute_full_values(self, query: str, thread_id: str = "web-debug"):
        """Run the agent and return full LangChain message payloads for debugging."""

        graph = self.compile_graph()
        config = {"configurable": {"thread_id": thread_id}}
        result = graph.invoke(self.build_messages(query), config)
        return {
            **result,
            "messages": [message_to_dict(message) for message in result["messages"]],
        }

    def execute_debug_stream(self, query: str, thread_id: str = "web-debug"):
        """Yield LangGraph debug events, including node inputs and outputs."""

        graph = self.compile_graph()
        config = {"configurable": {"thread_id": thread_id}}
        yield from graph.stream(self.build_messages(query), config, stream_mode="debug")


if __name__ == "__main__":
    agent = GraphAgent()

    result = agent.execute_full_values("")
    print(result)
    # graph = agent.compile_graph()
    # config = {"configurable": {"thread_id": "1"}}
    #
    # result = graph.invoke(agent.build_messages("扫地机器人建图不完整、地图错乱怎么办？"), config)
    # print(result)


