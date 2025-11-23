"""
所有模态最终需要输入的模型,各模态模型类的基类
"""
import logging
import sqlite3
import time

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ToolRetryMiddleware, SummarizationMiddleware, \
    ToolCallLimitMiddleware, wrap_tool_call
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.messages import ToolMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver  # 适合本地部署，短期记忆
from langgraph.checkpoint.postgres import PostgresSaver  # 适合企业级部署，长期记忆
from langgraph.checkpoint.sqlite import SqliteSaver  # 适合本地部署，长期记忆111
from typing import Literal

from langgraph.prebuilt.tool_node import ToolCallRequest
from tavily import TavilyClient
from deepagents import create_deep_agent
# tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
# from langgraph.checkpoint.memory import InMemorySaver
from my_tools import *
from utils import load_config

config = load_config()


class ChatBot():

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = load_config()
        # 初始化工具
        self.tools = get_tools()

        # 设置 LangChain Agent
        self.setup_agent()

    def setup_agent(self):
        """初始化LangChain Agent,开启流式输出"""
        # 初始化语言模型 - 关键:设置 streaming=True
        self.llm = ChatOpenAI(
            openai_api_key=config["api"]["openai_api_key"],
            base_url=config["api"]["base_url"],
            model=config["chat_model"]["name"],
        )

        @wrap_tool_call
        def handle_tool_errors(request: ToolCallRequest, handler) -> ToolMessage:
            """
            工具调用错误处理中间件：捕获执行异常，返回友好提示，避免原始错误信息泄露
            支持区分常见错误类型（参数错误、文件不存在、工具未找到等）
            """
            try:
                # 尝试执行工具调用
                return handler(request)
            except FileNotFoundError as e:
                # 文件相关错误（如 read_file 时文件不存在）
                return ToolMessage(
                    content=f"❌ 工具调用失败：找不到指定文件，请检查文件路径是否正确（{str(e).split(':')[-1].strip()}）",
                    tool_call_id=request.tool_call["id"],
                    status="error"  # 标记状态为错误，帮助 Agent 识别
                )
            except ValueError as e:
                # 参数错误（如传入格式错误、缺失必填参数）
                return ToolMessage(
                    content=f"❌ 工具调用失败：输入参数无效，请检查参数格式或补充必填信息（错误提示：{str(e)[:50]}）",
                    tool_call_id=request.tool_call["id"],
                    status="error"
                )
            except PermissionError as e:
                # 权限错误（如写文件时无权限）
                return ToolMessage(
                    content=f"❌ 工具调用失败：无操作权限（如读写文件、访问资源），请检查权限设置",
                    tool_call_id=request.tool_call["id"],
                    status="error"
                )
            except Exception as e:
                # 其他未知错误（精简错误信息，避免泄露敏感细节）
                error_msg = str(e)[:80]  # 限制错误信息长度，防止冗余
                return ToolMessage(
                    content=f"❌ 工具调用失败：未知错误，请稍后重试或联系管理员（错误摘要：{error_msg}...）",
                    tool_call_id=request.tool_call["id"],
                    status="error"
                )

        # 定义系统提示词
        system_prompt = r"""
        你叫小布,是一位聪明、亲切的,拥有数据处理能力的智能多模态助手。
        核心注意点：
            1. 当用户传入一个临时文件时无论如何都必须执行init_data工具。
            2. 当碰到不了解的问题前优先使用知识库检索工具search_knowledge，检索之后使用web_search工具搜索
            3. 需要严格区分机器学习模型训练和股票训练的情况，两者不可同时使用

        你有以下核心能力:
        1. 分析表格数据（包括基本信息获取和多种统计分析）
        2. 训练机器学习模型
        3. 保存对话的重要信息至知识库
        4. 删除记忆
        5. 进行网络搜索
        7. 检索知识库
        8. 查询股票并保存信息
        9. 训练股票的简单模型

        请根据用户的问题,合理选择和使用工具来提供准确的回答。

        """

        # 中间件
        summarization_middleware = SummarizationMiddleware(model=ChatOpenAI(
            openai_api_key=config["api"]["openai_api_key"],
            base_url=config["api"]["base_url"],
            model="Qwen/Qwen2.5-7B-Instruct", ))

        tool_retry_middleware = ToolRetryMiddleware(max_retries=3)
        # 创建带记忆的 Agent
        conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
        checkpointer = SqliteSaver(conn)  # 文件自动创建

        # 包装 Agent，在每次调用前修剪消息
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt,
            checkpointer=checkpointer,
            # 添加消息修剪中间件
            middleware=[tool_retry_middleware, summarization_middleware,handle_tool_errors],
        )


        # 绘制langgraph流程图
        # with open("create_agent_graph.png", "wb") as f:
        #     f.write(self.agent.get_graph().draw_mermaid_png())

