"""
主程序入口，负责初始化各个模块并启动AI Agent
"""
import logging
import threading
from typing import Literal  # 用于模式参数的枚举限制
from typing import Union, List, Dict
from load_rag import LoadRag
import streamlit
from fastapi import FastAPI, Body
from fastapi import Query  # 导入Query
from streamlit.runtime.uploaded_file_manager import UploadedFile

from chatbot_cv import Cv_Chatbot
from chatbot_nlp import nlp_chatbot
from utils import setup_logging
from chatbot_voice import voice_ChatBot
from voice_wake_up import WakeWordDetector

# 初始化FastAPI
app = FastAPI()

# 全局初始化AIAgent（避免重复创建）
setup_logging()  # 提前初始化日志配置


class AIAgent:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.wake_detector = WakeWordDetector()
        self.voice_chatbot = voice_ChatBot()
        self.cv_chatbot = Cv_Chatbot()
        self.running = False
        self.is_image = False
        self.nlp_chatbot = nlp_chatbot()
        # 用于存储对话结果的属性（初始化默认值）
        self.nlp_response = None
        self.voice_response = None
        # 线程安全锁（处理多线程读写结果）
        self.response_lock = threading.Lock()

    def _wake_callback(self,thread_id):
        """唤醒回调函数"""
        self.logger.info("检测到唤醒词，开始录音...")
        # 启动语音对话线程（避免阻塞回调）
        chat_thread = threading.Thread(target=self._start_voice_conversation,args=(thread_id,))
        chat_thread.start()

    def _start_voice_conversation(self,thread_id):
        """语音对话入口"""
        user_input = self.voice_chatbot.listen_and_recognize()
        if user_input:
            self.logger.info(f"用户输入: {user_input}")
            # 生成回复并语音播报
            response = self.voice_chatbot.chat(user_input, enable_voice=True,thread_id=thread_id)
            # 线程安全地更新结果
            with self.response_lock:
                self.voice_response = response
            self.logger.info(f"AI回复: {response}")
            # 对话完成后可停止监听（根据需求决定是否保持运行）
            self.stop()

    def start_monitor(self,thread_id):
        if self.running:
            self.logger.warning("AI Agent已在运行中")
            return
        self.logger.info("启动AI Agent...")
        self.running = True
        with self.response_lock:
            self.voice_response = None

        def _wrapped_wake_callback():
            self._wake_callback(thread_id)  # 这里传入 thread_id
        # 绑定回调后，打印日志确认
        self.wake_detector.on_wake_callback = _wrapped_wake_callback
        self.logger.info(f"回调函数绑定状态：{self.wake_detector.on_wake_callback is not None}")  # 验证是否绑定成功
        self.wake_thread = threading.Thread(target=self.wake_detector.run,args=(thread_id,))
        self.wake_thread.start()

        self.logger.info(f"监听线程是否存活：{self.wake_thread.is_alive()}")  # 验证线程是否启动

    def start_nlp_chatbot(self, prompt,thread_id):
        """NLP模式处理，同步返回结果"""
        self.nlp_response = self.nlp_chatbot.get_llm_response(prompt,thread_id)
        print(self.nlp_response)
        return self.nlp_response

    def start_cv_chatbot(self, prompt,thread_id):
        """"""
        self.cv_response = self.cv_chatbot.get_llm_response(prompt,thread_id)

        return self.cv_response

    def stop(self):
        """停止AI Agent"""
        if not self.running:
            return
        self.running = False
        self.wake_detector.stop()
        self.logger.info("AI Agent已停止")


# 实例化全局AIAgent（避免重复初始化资源）
ai_agent = AIAgent()


@app.post("/chat")
def chat(
        prompt: Union[str, List[Dict]] = Body(..., embed=True),
        thread_id: Union[str] = Body(..., embed=True),
        mode: Literal["nlp", "voice", "cv"] = Query("nlp")
):
    if mode == "nlp":
        response = ai_agent.start_nlp_chatbot(prompt,thread_id)
        print(response)
        return {"response": response}
    elif mode == "cv":
        response = ai_agent.start_cv_chatbot(prompt,thread_id)
        print("aaa", response)
        return {"response": response}
    elif mode == "voice" and prompt == "开始语音监听":
        ai_agent.start_monitor(thread_id)
        return {
            "msg": "已启动语音模式，请说出唤醒词并对话",
            "提示": "可通过另一个接口（如/voice_result）查询结果"
        }
    elif mode == "voice" and prompt == "关闭语音监听":
        ai_agent.stop()
        return {
            "msg": "已关闭语音模式"
        }

@app.post("/chat/rag")
async def get_chat_rag(
        file_path_list: List[str] = Body(..., embed=True)):
    loader = LoadRag(file_path_list)
    # 读取并处理
    await loader.load_process_files()
    # 加载至知识库
    await loader.load_rag()


@app.post("/chat/visualization")
async def get_chat_visualization():
    from deep_my_tools import visualization
    import base64
    _, base64_binary_fig, base64_fit_true_fig, base64_predict_true_fig= visualization()


    if not _:
        return {
            "success": False,
            "msg": "暂未训练模型",
            "response": {
                "binary_fig": base64_binary_fig,  # 第一个图表的Base64编码字符串
                "fit_true_fig": base64_fit_true_fig,  # 第二个图表的Base64编码字符串
                "predict_true_fig": base64_predict_true_fig  # 第三个图表的Base64编码字符串
            }
            # 这里的fig和ax均为None
        }
    else:
        return {
            "success": True,
            "msg": "可视化成功",
            "response": {
                "binary_fig": base64_binary_fig,  # 第一个图表的Base64编码字符串
                "fit_true_fig": base64_fit_true_fig,  # 第二个图表的Base64编码字符串
                "predict_true_fig": base64_predict_true_fig  # 第三个图表的Base64编码字符串
            }
        }



if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app)
