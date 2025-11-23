import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.tracers import ConsoleCallbackHandler

from utils import load_config
from chatbot import ChatBot
from utils import load_config


os.environ["USER_AGENT"] = "my-app/0.1"
import deep_my_tools
class nlp_chatbot(ChatBot):
    def __init__(self):
        super().__init__()
        self.config = load_config()


    def get_llm_response(self,query,thread_id):
        """
        调用 agent 并返回最终答案
        """
        # query = f"当前分析结果：{my_tools.ANALYZE_RESULT}。用户问题：{query}"
        try:
            result = self.agent.invoke(
                {"messages": [{"role": "user", "content": query}],"verbose":True},
                config= {"configurable": {"thread_id": thread_id},},
                stream_mode ="values"
            )
            # result =StrOutputParser().parse(result)
            # 从结果中提取最后一条消息
            return result["messages"][-1].content
        except ValueError as e:
            return f"调用失败，重新尝试，或重启agent{e}"

if __name__ == '__main__':

    chatbot = nlp_chatbot()

    # ret7 = chatbot.get_llm_response("删除记忆")
    ret8 = chatbot.get_llm_response(r"你好",thread_id="test_thread_id")
    # ret9 = chatbot.get_llm_response(r"你是谁？",thread_id="test_thread_id")
    # ret10 = chatbot.get_llm_response(r"你能做什么",thread_id="test_thread_id")
    # ret11 = chatbot.get_llm_response(r"我上一句话的上一句话的上一句话是什么？",thread_id="test_thread_id")
    # print("ret7",ret7)
    print("ret8",ret8)
    # print("ret11",ret11)