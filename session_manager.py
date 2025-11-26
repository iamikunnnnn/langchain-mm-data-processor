# session_manager.py
import threading
from typing import Dict, Any, Optional

class SessionState:
    """定义每个用户独立的会话状态"""
    def __init__(self):
        self.progressor = None  # 之前 my_tools 里的全局 progressor
        self.model = None       # 之前 my_tools 里的全局 model
        self.analyze_result = "无" # 之前 my_tools 里的全局 ANALYZE_RESULT
        self.enhanced_data_analyzer=None


# session_manager.py - 修改后
import threading
from typing import Dict
from collections import defaultdict

# session_manager.py - 修改后
import threading
from typing import Dict
from collections import defaultdict


class SessionManager:
    _instance = None
    _global_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._global_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.sessions: Dict[str, SessionState] = {}
                    cls._instance._session_locks = defaultdict(threading.Lock)  # 每个用户独立锁
        return cls._instance

    def get_session(self, thread_id: str) -> SessionState:
        """获取会话,避免全局锁"""
        # 只在创建新session时用全局锁
        if thread_id not in self.sessions:
            with self._global_lock:
                if thread_id not in self.sessions:
                    self.sessions[thread_id] = SessionState()
        return self.sessions[thread_id]

    def get_session_lock(self, thread_id: str) -> threading.Lock:
        """获取某个用户的专属锁"""
        return self._session_locks[thread_id]

# 全局单例实例
session_store = SessionManager()