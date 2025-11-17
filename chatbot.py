"""
所有模态最终需要输入的模型,各模态模型类的基类
"""
import logging
import sqlite3
import time

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_classic.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver  # 适合本地部署，短期记忆
from langgraph.checkpoint.postgres import PostgresSaver  # 适合企业级部署，长期记忆
from langgraph.checkpoint.sqlite import SqliteSaver  # 适合本地部署，长期记忆

from Middleware import MessageTrimmerMiddleware
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

        # 定义系统提示词
        system_prompt = r"""
        你叫小布,是一位聪明、亲切的,拥有数据处理能力的智能多模态助手。
        当用户传入一个临时文件时无论如何都必须执行init_data工具。
        当碰到不了解的问题前一定是先去知识库检索search_knowledge，检索完再去web_search搜索
        你有以下能力:
        1. 分析表格数据(核心能力)
        2. 训练机器学习模型(核心能力)
        3. 查询天气信息
        4. 获取当前时间
        5. 进行网络搜索
        6. 根据起始地点（从省级开始描述）和目标地点（从省级开始描述）获取导航信息
        7. 检索知识库

        请根据用户的问题,合理选择和使用工具来提供准确的回答。
        如果不需要使用工具,可以直接回答用户的问题。

        重要提示:
        - 当调用工具时，确保tool_calls的格式如下：
            {
        "name": "工具名",
        "args": {  # 注意：args必须是字典，而非字符串
            "参数1": "值1",
            "参数2": "值2"
        }
        }
        不要将args包裹在引号中，也不要返回JSON字符串，并且需要额外注意是否会出现"\"导致的转义符解析错误。
        """

        # 创建带记忆的 Agent

        conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
        checkpointer = SqliteSaver(conn)  # 文件自动创建
        # checkpointer = MemorySaver()  # 文件自动创建
        # 包装 Agent，在每次调用前修剪消息
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt,
            checkpointer=checkpointer,
            # 添加消息修剪中间件
            middleware=[MessageTrimmerMiddleware(max_messages=15)],

        )
        # with open("create_agent_graph.png", "wb") as f:
        #     f.write(self.agent.get_graph().draw_mermaid_png())

        # 初始化对话线程ID
        self.thread_id = "conversation_1"

    def clear_conversation_history(self):
        """清除对话历史"""
        pass
