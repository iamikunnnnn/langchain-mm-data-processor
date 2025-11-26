# chatbot_pool.py - 完整版本
import queue
from contextlib import contextmanager
from chatbot_nlp import nlp_chatbot
from chatbot_cv import Cv_Chatbot
from chatbot_voice import voice_ChatBot


class ChatBotPool:
    def __init__(self, nlp_size=10, cv_size=5, voice_size=3):
        """
        初始化三种类型的chatbot池
        - nlp_size: NLP实例数量（使用最频繁）
        - cv_size: CV实例数量（相对较少）
        - voice_size: Voice实例数量（最少，因为语音需要独占麦克风）
        """
        print("🚀 初始化 ChatBot 池...")

        # NLP池
        self.nlp_pool = queue.Queue(maxsize=nlp_size)
        for i in range(nlp_size):
            self.nlp_pool.put(nlp_chatbot())
            print(f"  ✓ NLP实例 {i + 1}/{nlp_size} 创建完成")

        # CV池
        self.cv_pool = queue.Queue(maxsize=cv_size)
        for i in range(cv_size):
            self.cv_pool.put(Cv_Chatbot())
            print(f"  ✓ CV实例 {i + 1}/{cv_size} 创建完成")

        # Voice池
        self.voice_pool = queue.Queue(maxsize=voice_size)
        for i in range(voice_size):
            self.voice_pool.put(voice_ChatBot())
            print(f"  ✓ Voice实例 {i + 1}/{voice_size} 创建完成")

        print("✅ ChatBot池初始化完成")

    @contextmanager
    def acquire_nlp(self, timeout=5):
        """获取NLP实例"""
        chatbot = self.nlp_pool.get(timeout=timeout)
        try:
            yield chatbot
        finally:
            self.nlp_pool.put(chatbot)

    @contextmanager
    def acquire_cv(self, timeout=5):
        """获取CV实例"""
        chatbot = self.cv_pool.get(timeout=timeout)
        try:
            yield chatbot
        finally:
            self.cv_pool.put(chatbot)

    @contextmanager
    def acquire_voice(self, timeout=10):
        """获取Voice实例（超时设长一点，因为语音处理慢）"""
        chatbot = self.voice_pool.get(timeout=timeout)
        try:
            yield chatbot
        finally:
            self.voice_pool.put(chatbot)


# 全局池
chatbot_pool = ChatBotPool(nlp_size=10, cv_size=5, voice_size=3)