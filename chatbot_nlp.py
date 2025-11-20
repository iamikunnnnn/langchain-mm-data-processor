import os

from langchain_core.tracers import ConsoleCallbackHandler

from utils import load_config
from chatbot import ChatBot
from utils import load_config
os.environ["USER_AGENT"] = "my-app/0.1"

class nlp_chatbot(ChatBot):
    def __init__(self):
        super().__init__()
        self.config = load_config()

    def get_llm_response(self,query):
        """
        调用 agent 并返回最终答案
        """
        try:
            result = self.agent.invoke(
                {"messages": [{"role": "user", "content": query}],"verbose":True},
                config= {"configurable": {"thread_id": self.thread_id}},
                stream_mode ="values"
            )

            # 从结果中提取最后一条消息
            return result["messages"][-1].content
        except ValueError as e:
            return f"调用失败，重新尝试，或重启agent{e}"

if __name__ == '__main__':

    chatbot = nlp_chatbot()

    # ret7 = chatbot.get_llm_response("你好，现在几点了？")
    ret8 = chatbot.get_llm_response("对这个股票训练一个预测模型")

    # print("ret7",ret7)
    print("ret8",ret8)