# main.py - 完整修改版
from chatbot_pool import chatbot_pool
import logging
import threading
from typing import Literal, Union, List, Dict
from fastapi import FastAPI, Body, Query, HTTPException
from load_rag import LoadRag
from utils import setup_logging
from voice_wake_up import WakeWordDetector

app = FastAPI()
setup_logging()

# ⚠️ 语音模式的特殊处理：需要维护唤醒检测器的状态
# 因为语音监听是长时间运行的，不适合用对象池
class VoiceMonitorManager:
    """管理每个用户的语音监听会话"""
    def __init__(self):
        self.sessions = {}  # {thread_id: {"detector": WakeWordDetector, "chatbot": voice_ChatBot}}
        self.lock = threading.Lock()
    
    def start_monitor(self, thread_id: str):
        """启动某个用户的语音监听"""
        with self.lock:
            if thread_id in self.sessions:
                return {"error": "该用户已在监听中"}
            
            # 从池中获取voice_chatbot（用完不归还，直到停止监听）
            try:
                voice_chatbot = chatbot_pool.voice_pool.get(timeout=5)
            except:
                raise HTTPException(status_code=503, detail="语音服务繁忙，请稍后再试")
            
            detector = WakeWordDetector()
            
            # 设置唤醒回调
            def wake_callback():
                user_input = voice_chatbot.listen_and_recognize()
                if user_input:
                    response = voice_chatbot.chat(user_input, enable_voice=True, thread_id=thread_id)
                    logging.info(f"[{thread_id}] AI回复: {response}")
            
            detector.on_wake_callback = wake_callback
            
            # 启动监听线程
            monitor_thread = threading.Thread(target=detector.run, args=(thread_id,))
            monitor_thread.start()
            
            self.sessions[thread_id] = {
                "detector": detector,
                "chatbot": voice_chatbot,
                "thread": monitor_thread
            }
            
            return {"msg": "语音监听已启动"}
    
    def stop_monitor(self, thread_id: str):
        """停止某个用户的语音监听"""
        with self.lock:
            if thread_id not in self.sessions:
                return {"error": "该用户未在监听中"}
            
            session = self.sessions[thread_id]
            session["detector"].stop()
            
            # 归还chatbot到池中
            chatbot_pool.voice_pool.put(session["chatbot"])
            
            del self.sessions[thread_id]
            return {"msg": "语音监听已停止"}

voice_monitor_manager = VoiceMonitorManager()


# ==================== API 路由 ====================

@app.post("/chat")
def chat(
    prompt: Union[str, List[Dict]] = Body(..., embed=True),
    thread_id: str = Body(..., embed=True),
    mode: Literal["nlp", "voice", "cv"] = Query("nlp")
):
    """统一聊天接口"""
    
    # 1. NLP模式
    if mode == "nlp":
        try:
            with chatbot_pool.acquire_nlp(timeout=5) as chatbot:
                response = chatbot.get_llm_response(prompt, thread_id)
            return {"response": response}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"NLP服务错误: {str(e)}")
    
    # 2. CV模式
    elif mode == "cv":
        try:
            with chatbot_pool.acquire_cv(timeout=5) as chatbot:
                response = chatbot.get_llm_response(prompt, thread_id)
            return {"response": response}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"CV服务错误: {str(e)}")
    
    # 3. Voice模式 - 控制命令
    elif mode == "voice":
        if prompt == "开始语音监听":
            result = voice_monitor_manager.start_monitor(thread_id)
            return result
        
        elif prompt == "关闭语音监听":
            result = voice_monitor_manager.stop_monitor(thread_id)
            return result
        
        else:
            # 一次性语音对话（不启动持续监听）
            try:
                with chatbot_pool.acquire_voice(timeout=10) as chatbot:
                    response = chatbot.chat(prompt, enable_voice=False, thread_id=thread_id)
                return {"response": response}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"语音服务错误: {str(e)}")


@app.post("/chat/rag")
async def get_chat_rag(file_path_list: List[str] = Body(..., embed=True)):
    """加载文档到知识库"""
    try:
        loader = LoadRag(file_path_list)
        await loader.load_process_files()
        await loader.load_rag()
        return {"msg": "知识库加载成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/visualization")
async def get_chat_visualization():
    """获取模型可视化结果"""
    from deep_my_tools import visualization
    
    success, binary_fig, fit_true_fig, predict_true_fig = visualization()
    
    return {
        "success": success,
        "msg": "可视化成功" if success else "暂未训练模型",
        "response": {
            "binary_fig": binary_fig,
            "fit_true_fig": fit_true_fig,
            "predict_true_fig": predict_true_fig
        }
    }


@app.on_event("shutdown")
def shutdown_event():
    """服务关闭时清理资源"""
    logging.info("正在关闭所有语音监听...")
    for thread_id in list(voice_monitor_manager.sessions.keys()):
        voice_monitor_manager.stop_monitor(thread_id)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)