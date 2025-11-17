import base64
import io
import os
import time
import tempfile
import requests
import streamlit as st
import json

from PIL import Image

from chatbot_cv import Cv_Chatbot
from machine_learning_model import *
temp_file_path_list = []  # 保存临时文件的名字用于后续删除（注：删除功能暂未完成）

st.title("AI 数据分析")

# 初始化会话状态：存储对话历史、轮询状态、语音模式开关
if "history" not in st.session_state:
    st.session_state.history = []  # 对话历史
if "is_voice_mode" not in st.session_state:
    st.session_state.is_voice_mode = False  # 是否开启语音模式
if "polling" not in st.session_state:
    st.session_state.polling = False  # 是否正在轮询
if "voice_result_received" not in st.session_state:
    st.session_state.voice_result_received = False  # 是否已收到语音结果


# 后端接口配置（与 FastAPI 对应）
FASTAPI_URL = "http://127.0.0.1:8000"
CHAT_ENDPOINT = f"{FASTAPI_URL}/chat"
RAG_ENDPOINT = f"{FASTAPI_URL}/chat/rag"
VISUALIZATION_ENDPOINT = f"{FASTAPI_URL}/chat/visualization"
VOICE_RESULT_ENDPOINT = f"{FASTAPI_URL}/voice_result"

st.session_state.is_voice_mode = False

cv_chatbot = Cv_Chatbot()

# 显示对话历史
# 每次也买你刷新先显示历史聊天记录
for msg in st.session_state["history"]:
    with st.chat_message(msg["role"]):
        if msg["type"] == "text":
            st.markdown(msg["content"])
        elif msg["type"] == "image":
            msg["content"].seek(0)
            st.image(msg["content"])
with st.sidebar:
    uploaded_files = st.file_uploader(
        "选择要上传的文件（可多选）",
        type=None,  # 允许所有文件类型
        accept_multiple_files=True  # 关键参数：允许多文件上传
    )
    file_path_list=[]# 用于保存加载的所有文件
    if uploaded_files is not None:
        if st.button("处理文件并上传知识库"):
            for file in uploaded_files:

                # 显示文件名
                st.write(f"已上传文件: {file.name}")
                ext = os.path.splitext(file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
                    f.write(file.getvalue())
                    f.flush()
                    temp_file_path = f.name
                    temp_file_path_list.append(temp_file_path) # 这个是整体的，用于后续删除临时文件
                    file_path_list.append(temp_file_path) # 这个是局部的，仅用于这里的文件地址传输
            if not st.session_state.is_voice_mode:
                response = requests.post(
                    RAG_ENDPOINT,
                    json={"file_path_list": file_path_list},  # 使用JSON格式传递
                    params={"mode": "nlp"}
                )
    if st.button("可视化"):
        response = requests.post(
            VISUALIZATION_ENDPOINT,
        )
        if response.json().get("success"):
            fig_dict = response.json().get("response")

            binary_fig_b64 = fig_dict["binary_fig"]  # 获取Base64字符串
            fit_true_fig_b64 = fig_dict["fit_true_fig"]  # 获取Base64字符串
            predict_true_b64 = fig_dict["predict_true_fig"]  # 获取Base64字符串

            binary_fig_bytes = base64.b64decode(binary_fig_b64)  # 解码为二进制数据
            fit_true_fig_bytes = base64.b64decode(fit_true_fig_b64)  # 解码为二进制数据
            predict_true_bytes = base64.b64decode(predict_true_b64)  # 解码为二进制数据

            binary_fig_img = Image.open(io.BytesIO(binary_fig_bytes))
            fit_true_fig_img = Image.open(io.BytesIO(fit_true_fig_bytes))
            predict_true_img = Image.open(io.BytesIO(predict_true_bytes))

            st.session_state.history.append({"role": "assistant", "content": binary_fig_img, "type": "image"})
            st.session_state.history.append({"role": "assistant", "content": fit_true_fig_img, "type": "image"})
            st.session_state.history.append({"role": "assistant", "content": predict_true_img, "type": "image"})

            st.rerun()


# 核心逻辑：根据模式处理对话
if input_msg := st.chat_input("来和我聊天吧~~~", accept_file="multiple", file_type=['png', 'jpg', 'jpeg', "xlsx", "csv"]):
    if input_msg.text:  # 检查是否有文本内容
        # 1. 文本模式（默认）：直接调用 /chat 接口
        if not st.session_state.is_voice_mode:
            try:
                # 发送文本请求到 FastAPI（nlp 模式）


                response = requests.post(
                    CHAT_ENDPOINT,
                    json={"prompt": input_msg.text},  # 使用JSON格式传递
                    params={"mode": "nlp"}
                )
                print(response.json())
                response.raise_for_status()

                # 更新对话历史并显示
                st.session_state.history.append({"role": "human", "content": input_msg.text, "type": "text"})
                st.session_state.history.append({"role": "assistant", "content": response.text, "type": "text"})

                # 刷新页面显示新消息（Streamlit 会自动重渲染）
                st.rerun()

            except ValueError as e:
                st.error(f"文本对话失败：{str(e)}")
    # 3. 文件(包括图片)模式

    if input_msg.files:  # 检查是否有文件
        st.warning("文件读取中")

        # 使用for循环地遍历所有文件
        for file in input_msg.files:
            file_name = file.name.lower()  # 文件名转小写，避免大小写问题
            file_type = None

            # 根据文件名后缀判断类型
            if file_name.endswith(('.png', '.jpg', '.jpeg')):
                file_type = "image"
                st.warning(f"检测到图像文件：{file_name}")
            elif file_name.endswith(('.xlsx', '.csv','.json')):
                file_type = "tabel"
                st.warning(f"检测到Excel文件：{file_name}")
            else:
                st.error(f"不支持的文件类型：{file_name}")
                continue  # 跳过不支持的文件
            if file_type == "image":
                with st.chat_message("human"):
                    st.image(file, width=100)
                    st.session_state.history.append({"role": "human", "content": file, "type": "image"})
                    st.session_state.history.append({"role": "human", "content": input_msg.text, "type": "text"})
                try:
                    # 由于predict函数期望的是文件路径，这里先存入临时文件并把路径给函数predict
                    # delete=True表示with块结束后自动删除文件，因为得到prompt后就不需要这个文件了
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as f:
                        f.write(file.getvalue())
                        f.flush()
                        temp_file_path = f.name  # 保存路径用于后续删除

                        base64_str = cv_chatbot.img2base64(temp_file_path)

                        prompt = cv_chatbot.get_prompt(base64_str)
                        print(prompt)
                    if not st.session_state.is_voice_mode:
                        # 发送文本请求到 FastAPI
                        response = requests.post(
                            CHAT_ENDPOINT,
                            json={"prompt": prompt},  # 使用JSON格式传递
                            params={"mode": "cv"}
                        )

                        # 更新对话历史并显示
                        st.session_state.history.append(
                            {"role": "assistant", "content": response.json(), "type": "text"})
                        # 刷新页面显示新消息（Streamlit 会自动重渲染）
                        st.rerun()
                except Exception as e:
                    st.error(e)
            if file_type == "tabel":
                try:
                    # 由于predict函数期望的是文件路径，这里先存入临时文件并把路径给函数predict
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
                        f.write(file.getvalue())
                        f.flush()
                        temp_file_path = f.name  # 保存路径用于后续删除
                        prompt = temp_file_path
                        temp_file_path_list.append(temp_file_path)
                        st.session_state.history.append({"role": "human", "content": temp_file_path, "type": "text"})
                    if not st.session_state.is_voice_mode:
                        # 发送文本请求到 FastAPI
                        import json

                        response = requests.post(
                            CHAT_ENDPOINT,
                            json={"prompt": prompt},  # 使用JSON格式传递
                            params={"mode": "nlp"}
                        )
                        # 更新对话历史并显示
                        st.session_state.history.append(
                            {"role": "assistant", "content": response.json(), "type": "text"})
                        st.rerun()
                except Exception as e:
                    print(e)

# 2. 语音模式：启动监听 + 轮询结果
try:
    if st.button("启动语音监听"):
        st.session_state.is_voice_mode = True
        # 第一步：发送请求启动 FastAPI 语音监听（mode=voice）
        response = requests.post(
            CHAT_ENDPOINT,
            json={"prompt": "开始语音监听"},  # 使用JSON格式传递
            params={"mode": "voice"}
        )
        print(response.json())
        response.raise_for_status()

        # 显示“启动成功”提示
        st.session_state.history.append({"role": "human", "content": "已启动语音模式，请说出唤醒词...", "type": "text"})
        st.chat_message("human").markdown("已启动语音模式，请说出唤醒词...")
        # 第二步：开始轮询查询结果（设置轮询状态为 True）
        st.session_state.polling = True
        st.session_state.voice_result_received = False
        polling_count = 0  # 轮询计数器（避免无限轮询）
        max_polls = 60  # 最大轮询次数（60次 × 1秒 = 1分钟超时）
        # 轮询循环：每秒查询一次 /voice_result 接口
        while st.session_state.polling and polling_count < max_polls:
            try:
                # 调用结果查询接口
                result_response = requests.get(VOICE_RESULT_ENDPOINT)
                result_data = result_response.json()
                # 情况1：已获取到语音结果
                if "response" in result_data and result_data["response"] is not None:
                    st.session_state.history.append({
                        "role": "assistant",
                        "content": result_data["response"],
                        "type": "text"
                    })
                    st.session_state.voice_result_received = True
                    st.session_state.polling = False  # 停止轮询
                    break
                # 情况2：未获取到结果（继续轮询）
                else:
                    polling_count += 1
                    # 显示轮询状态（用空消息占位，避免重复渲染）
                    with st.spinner(f"等待语音对话完成...（{polling_count}/{max_polls}）"):
                        time.sleep(1)  # 每秒查询一次
            except Exception as e:
                st.warning(f"轮询失败：{str(e)}，将继续尝试...")
                time.sleep(1)
                polling_count += 1
        # 轮询结束处理
        if st.session_state.voice_result_received:
            st.success("语音对话完成！")
            st.rerun()  # 刷新显示AI回复
        else:
            st.error("轮询超时，未获取到语音结果（请检查唤醒词是否触发、后端是否正常运行）")
            # 重置状态
            st.session_state.polling = False
            st.session_state.voice_result_received = False
    if st.button("关闭语音监听"):
        st.session_state.is_voice_mode = True
        # 第一步：发送请求启动 FastAPI 语音监听（mode=voice）
        response = requests.post(
            CHAT_ENDPOINT,
            json={"prompt": "关闭语音监听"},  # 使用JSON格式传递
            params={"mode": "voice"}
        )
        print(response.json())
        response.raise_for_status()

except ValueError as e:
    st.error(f"启动语音模式失败：{str(e)}")
finally:
    st.session_state.is_voice_mode = False

# 手动停止轮询按钮（可选，提升用户体验）
if st.session_state.polling:
    if st.button("停止轮询"):
        st.session_state.polling = False
        st.success("已停止轮询")
        st.rerun()
