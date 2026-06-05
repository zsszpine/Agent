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

from langchain_core.messages import HumanMessage, SystemMessage
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

    def compile_graph(self):
        """构建 LangGraph 执行图。"""

        def call_model(state: MessagesState):
            """Agent 节点：把当前 messages 交给模型，让模型决定回答或发起工具调用。"""

            messages = state["messages"]
            response = self.model.invoke(messages)
            return {"messages": [response]}

        def should_continue(state: MessagesState):
            """
            条件边：如果模型返回 tool_calls，就进入 tools 节点；否则结束图执行。
            """

            message = state["messages"][-1]
            if message.tool_calls:
                return "tools"
            return END

        builder = StateGraph(MessagesState)
        builder.add_node("agent", call_model)
        builder.add_node("tools", self.tool_node)
        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", should_continue)
        builder.add_edge("tools", "agent")

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


