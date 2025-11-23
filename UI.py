import base64
import io
import os
import time
import tempfile
import requests
import streamlit as st
import json
import uuid

from PIL import Image

from chatbot_cv import Cv_Chatbot
from machine_learning_model import *

temp_file_path_list = []  # 保存临时文件的名字用于后续删除（注：删除功能暂未完成）

st.set_page_config(page_title="AI 数据分析", page_icon="🤖", layout="wide")

st.title("🤖 AI 数据分析")

# 初始化会话状态
if "sessions" not in st.session_state:
    # 存储所有会话，格式：{session_id: {"name": str, "history": list, "thread_id": str}}
    default_session_id = str(uuid.uuid4())
    st.session_state.sessions = {
        default_session_id: {
            "name": "会话 1",
            "history": [],
            "thread_id": str(uuid.uuid4())
        }
    }
    st.session_state.current_session_id = default_session_id

if "is_voice_mode" not in st.session_state:
    st.session_state.is_voice_mode = False

# 后端接口配置
FASTAPI_URL = "http://127.0.0.1:8000"
CHAT_ENDPOINT = f"{FASTAPI_URL}/chat"
RAG_ENDPOINT = f"{FASTAPI_URL}/chat/rag"
VISUALIZATION_ENDPOINT = f"{FASTAPI_URL}/chat/visualization"

cv_chatbot = Cv_Chatbot()


# 获取当前会话
def get_current_session():
    return st.session_state.sessions[st.session_state.current_session_id]


def get_current_thread_id():
    return get_current_session()["thread_id"]


# 侧边栏：会话管理
with st.sidebar:
    st.header("📋 会话管理")

    # 会话选择器
    session_names = {sid: sdata["name"] for sid, sdata in st.session_state.sessions.items()}
    selected_session_name = st.selectbox(
        "选择会话",
        options=list(session_names.values()),
        index=list(session_names.keys()).index(st.session_state.current_session_id)
    )

    # 更新当前会话ID
    for sid, name in session_names.items():
        if name == selected_session_name:
            st.session_state.current_session_id = sid
            break

    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ 新建会话", use_container_width=True):
            new_session_id = str(uuid.uuid4())
            session_count = len(st.session_state.sessions) + 1
            st.session_state.sessions[new_session_id] = {
                "name": f"会话 {session_count}",
                "history": [],
                "thread_id": str(uuid.uuid4())
            }
            st.session_state.current_session_id = new_session_id
            st.rerun()

    with col2:
        if st.button("🗑️ 删除会话", use_container_width=True):
            if len(st.session_state.sessions) > 1:
                del st.session_state.sessions[st.session_state.current_session_id]
                st.session_state.current_session_id = list(st.session_state.sessions.keys())[0]
                st.rerun()
            else:
                st.warning("至少保留一个会话")

    # 重命名会话
    new_name = st.text_input("重命名当前会话", value=get_current_session()["name"])
    if new_name != get_current_session()["name"]:
        get_current_session()["name"] = new_name
        st.rerun()

    st.divider()

    # 显示当前会话的thread_id（调试用）
    with st.expander("🔍 会话信息"):
        st.text(f"Thread ID: {get_current_thread_id()[:8]}...")
        st.text(f"消息数: {len(get_current_session()['history'])}")

    st.divider()

    # 文件上传
    st.header("📁 文件处理")
    uploaded_files = st.file_uploader(
        "选择要上传的文件（可多选）",
        type=None,
        accept_multiple_files=True
    )

    file_path_list = []
    if uploaded_files is not None:
        if st.button("处理文件并上传知识库"):
            for file in uploaded_files:
                st.write(f"✅ 已上传: {file.name}")
                ext = os.path.splitext(file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
                    f.write(file.getvalue())
                    f.flush()
                    temp_file_path = f.name
                    temp_file_path_list.append(temp_file_path)
                    file_path_list.append(temp_file_path)

            if not st.session_state.is_voice_mode:
                response = requests.post(
                    RAG_ENDPOINT,
                    json={
                        "file_path_list": file_path_list,
                        "thread_id": get_current_thread_id()
                    },
                    params={"mode": "nlp"}
                )

    if st.button("📊 可视化"):
        response = requests.post(
            VISUALIZATION_ENDPOINT,
            json={"thread_id": get_current_thread_id()}
        )
        if response.json().get("success"):
            fig_dict = response.json().get("response")

            binary_fig_b64 = fig_dict["binary_fig"]
            fit_true_fig_b64 = fig_dict["fit_true_fig"]
            predict_true_b64 = fig_dict["predict_true_fig"]

            binary_fig_bytes = base64.b64decode(binary_fig_b64)
            fit_true_fig_bytes = base64.b64decode(fit_true_fig_b64)
            predict_true_bytes = base64.b64decode(predict_true_b64)

            binary_fig_img = Image.open(io.BytesIO(binary_fig_bytes))
            fit_true_fig_img = Image.open(io.BytesIO(fit_true_fig_bytes))
            predict_true_img = Image.open(io.BytesIO(predict_true_bytes))

            get_current_session()["history"].append({"role": "assistant", "content": binary_fig_img, "type": "image"})
            get_current_session()["history"].append({"role": "assistant", "content": fit_true_fig_img, "type": "image"})
            get_current_session()["history"].append({"role": "assistant", "content": predict_true_img, "type": "image"})

            st.rerun()

# 显示当前会话的对话历史
for msg in get_current_session()["history"]:
    with st.chat_message(msg["role"]):
        if msg["type"] == "text":
            st.markdown(msg["content"])
        elif msg["type"] == "image":
            if isinstance(msg["content"], str):
                st.text(msg["content"])
            else:
                msg["content"].seek(0)
                st.image(msg["content"])

# 核心逻辑：根据模式处理对话
if input_msg := st.chat_input("来和我聊天吧~~~", accept_file="multiple",
                              file_type=['png', 'jpg', 'jpeg', "xlsx", "csv"]):
    if input_msg.text:
        if not st.session_state.is_voice_mode:
            try:
                response = requests.post(
                    CHAT_ENDPOINT,
                    json={
                        "prompt": input_msg.text,
                        "thread_id": get_current_thread_id()
                    },
                    params={"mode": "nlp"}
                )
                print(response.json())
                response.raise_for_status()

                get_current_session()["history"].append({"role": "human", "content": input_msg.text, "type": "text"})
                get_current_session()["history"].append({"role": "assistant", "content": response.text, "type": "text"})

                st.rerun()

            except ValueError as e:
                st.error(f"文本对话失败：{str(e)}")

    if input_msg.files:
        st.warning("文件读取中")

        for file in input_msg.files:
            file_name = file.name.lower()
            file_type = None

            if file_name.endswith(('.png', '.jpg', '.jpeg')):
                file_type = "image"
                st.warning(f"检测到图像文件：{file_name}")
            elif file_name.endswith(('.xlsx', '.csv', '.json')):
                file_type = "tabel"
                st.warning(f"检测到表格文件：{file_name}")
            else:
                st.error(f"不支持的文件类型：{file_name}")
                continue

            if file_type == "image":
                with st.chat_message("human"):
                    st.image(file, width=100)
                    get_current_session()["history"].append({"role": "human", "content": file, "type": "image"})
                    get_current_session()["history"].append(
                        {"role": "human", "content": input_msg.text, "type": "text"})

                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as f:
                        f.write(file.getvalue())
                        f.flush()
                        temp_file_path = f.name

                        base64_str = cv_chatbot.img2base64(temp_file_path)
                        prompt = cv_chatbot.get_prompt(base64_str)
                        print(prompt)

                    if not st.session_state.is_voice_mode:
                        response = requests.post(
                            CHAT_ENDPOINT,
                            json={
                                "prompt": prompt,
                                "thread_id": get_current_thread_id()
                            },
                            params={"mode": "cv"}
                        )

                        get_current_session()["history"].append(
                            {"role": "assistant", "content": response.json(), "type": "text"})
                        st.rerun()
                except Exception as e:
                    st.error(e)

            if file_type == "tabel":
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
                        f.write(file.getvalue())
                        f.flush()
                        temp_file_path = f.name
                        prompt = temp_file_path
                        temp_file_path_list.append(temp_file_path)
                        get_current_session()["history"].append(
                            {"role": "human", "content": temp_file_path, "type": "text"})

                    if not st.session_state.is_voice_mode:
                        response = requests.post(
                            CHAT_ENDPOINT,
                            json={
                                "prompt": prompt,
                                "thread_id": get_current_thread_id()
                            },
                            params={"mode": "nlp"}
                        )
                        get_current_session()["history"].append(
                            {"role": "assistant", "content": response.json(), "type": "text"})
                        st.rerun()
                except Exception as e:
                    print(e)

# 语音模式
try:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎤 启动语音监听"):
            st.session_state.is_voice_mode = True
            response = requests.post(
                CHAT_ENDPOINT,
                json={
                    "prompt": "开始语音监听",
                    "thread_id": get_current_thread_id()
                },
                params={"mode": "voice"}
            )
            print(response.json())
            response.raise_for_status()
            st.chat_message("human").info("已启动语音模式，请说出唤醒词...")

    with col2:
        if st.button("🔇 关闭语音监听"):
            st.session_state.is_voice_mode = False
            response = requests.post(
                CHAT_ENDPOINT,
                json={
                    "prompt": "关闭语音监听",
                    "thread_id": get_current_thread_id()
                },
                params={"mode": "voice"}
            )
            print(response.json())
            response.raise_for_status()
            st.chat_message("human").info("已关闭语音模式")

except ValueError as e:
    st.error(f"语音模式操作失败：{str(e)}")
finally:
    st.session_state.is_voice_mode = False