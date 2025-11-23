"""
多模态语音聊天功能实现,使用LangChain 1.0.5构建Agent
统一文本和语音输入,支持流式输出
"""
import asyncio
import io
import logging
import os
import queue
import re
import tempfile
import threading
import time

import edge_tts
import pyaudio
import pygame
from openai import OpenAI
from pydub import AudioSegment

from chatbot import ChatBot
from utils import load_config


class Recorder:
    """录音器类,用于录制音频并保存为MP3文件"""

    def __init__(self):
        """初始化录音器对象"""
        self.is_recording = False
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 1024
        self.audio = None
        self.stream = None
        self.frames = []
        os.makedirs('./temp', exist_ok=True)
        self.output_file = "./temp/output.mp3"

    def start(self, duration_sec=10):
        """开始录音,自动在 duration_sec 秒后停止"""
        self.audio = pyaudio.PyAudio()
        self.frames = []

        try:
            self.stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
                input_device_index=1
            )
        except Exception as e:
            print(f"无法打开麦克风: {e}")
            return False

        self.is_recording = True
        print(f"开始录音,最长 {duration_sec} 秒...")

        start_time = time.time()
        try:
            while self.is_recording:
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                self.frames.append(data)

                if time.time() - start_time >= duration_sec:
                    print("录音时间到,自动停止。")
                    self.stop()
        except Exception as e:
            print(f"录音出错: {e}")
            self.stop()
        return True

    def stop(self):
        """停止录音并保存为 mp3 文件"""
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()
        self.save_to_mp3()

    def save_to_mp3(self):
        """将录音数据保存为 mp3文件"""
        try:
            audio_data = b''.join(self.frames)
            if len(audio_data) % 2 != 0:
                audio_data = audio_data[:-1]

            audio = AudioSegment(
                data=audio_data,
                sample_width=2,
                frame_rate=self.RATE,
                channels=self.CHANNELS
            )
            audio.export(self.output_file, format="mp3", bitrate="128k")
            logging.info(f"录音保存为 {self.output_file}")
        except Exception as e:
            logging.error(f"保存录音失败: {e}")


class SoundToText:
    """语音识别类,将音频文件转换为文本"""

    def __init__(self, client):
        """初始化语音识别模块"""
        self.client = client
        self.model = 'FunAudioLLM/SenseVoiceSmall'
        self.path = None

    def get_path(self, path: str):
        """设置音频文件路径"""
        self.path = path

    def run(self):
        """识别音频文件并返回文本"""
        try:
            if not self.path or not os.path.exists(self.path):
                raise FileNotFoundError(f"音频文件不存在: {self.path}")
            result = self._recognize_with_client()
            print(f"[SoundToText] 识别结果:{result}")
            return result
        except Exception as e:
            print(f"[SoundToText] 识别失败:{e}")
            return ""

    def _recognize_with_client(self):
        """使用云端 API 进行语音识别"""
        try:
            with open(self.path, "rb") as f:
                response = self.client.audio.transcriptions.create(
                    file=f,
                    model=self.model
                )
            return response.text
        except Exception as e:
            print(f"[SoundToText] 云端识别失败: {e}")
            return ""


class TextToSound:
    """文本转语音类"""

    def __init__(self, tts_provider='LOCAL_SERVICE'):
        """初始化文本转语音模块"""
        self.tts_provider = tts_provider
        self.local_voice = 'zh-CN-XiaoyiNeural'

    async def run(self, text: str) -> bytes:
        """传入文本,返回合成后的 mp3 音频字节"""
        max_retries = 2
        for attempt in range(max_retries):
            try:
                if self.tts_provider == 'LOCAL_SERVICE':
                    return await self.local_tts(text)
            except Exception as e:
                print(f"[TextToSound] 第 {attempt + 1} 次合成失败: {e}")
                await asyncio.sleep(1)
        return b''

    async def local_tts(self, text: str) -> bytes:
        """使用 edge-tts 本地服务生成语音"""
        communicate = edge_tts.Communicate(
            text=text,
            voice=self.local_voice,
            rate="+0%",
            pitch="+0Hz",
            volume="+0%",
        )

        chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])

        audio_data = b"".join(chunks)
        return audio_data


def play_audio(audio_data):
    """同步播放音频文件"""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_mp3:
        temp_mp3_path = temp_mp3.name

    try:
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
        audio_segment.export(temp_mp3_path, format="mp3")

        sound = pygame.mixer.Sound(temp_mp3_path)
        channel = sound.play()

        while channel.get_busy():
            time.sleep(0.1)
    finally:
        if os.path.exists(temp_mp3_path):
            os.remove(temp_mp3_path)


class StreamingResponseHandler:
    """处理流式响应的工具类"""

    def __init__(self, sentence_queue, text_to_sound):
        self.sentence_queue = sentence_queue
        self.text_to_sound = text_to_sound
        self.token_buffer = ''

    def process_chunk(self, chunk_text):
        """处理流式输出的文本片段"""
        if chunk_text:
            self.token_buffer += chunk_text
            if len(self.token_buffer) > 10:
                sentence, self.token_buffer = self.split_token_to_sentence(self.token_buffer)
                if sentence:
                    self.sentence_queue.put(sentence)

    def flush_remaining(self):
        """刷新剩余缓冲区内容"""
        if self.token_buffer.strip():
            self.sentence_queue.put(self.token_buffer.strip())
            self.token_buffer = ''

    @staticmethod
    def split_token_to_sentence(token_buffer, max_len=40):
        """将模型输出的文本缓冲区分割成完整句子"""
        strong_pattern = r'([^。!?~]*[。!?~])'
        strong_sentences = re.findall(strong_pattern, token_buffer)

        if strong_sentences:
            last_sentence = strong_sentences[-1].strip()
            remaining_buffer = token_buffer[token_buffer.rfind(last_sentence) + len(last_sentence):]
            return last_sentence, remaining_buffer

        if len(token_buffer) >= max_len:
            comma_index = token_buffer.rfind(',', 0, max_len)
            if comma_index != -1:
                sentence = token_buffer[:comma_index + 1].strip()
                remaining_buffer = token_buffer[comma_index + 1:]
                return sentence, remaining_buffer
            else:
                return None, token_buffer

        return None, token_buffer


class voice_ChatBot(ChatBot):
    """
    多模态语音聊天机器人类,使用LangChain 1.0.5构建Agent
    统一支持文本和语音输入输出,使用流式响应
    """

    def __init__(self):
        super(voice_ChatBot, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.config = load_config()

        # 初始化语音相关组件

        ### 初始化 OpenAI 客户端
        self.openai_client = OpenAI(
            api_key='sk-xtgeyahfmjxvrxjygvnwfywbezskstroipqofybruqldkgor',
            base_url='https://api.siliconflow.cn/v1'
        )
        self.recorder = Recorder()
        self.sound_to_text = SoundToText(self.openai_client)
        self.text_to_sound = TextToSound(tts_provider='LOCAL_SERVICE')

        # 初始化 pygame mixer(用于音频播放)
        pygame.mixer.init()

        # 初始化句子队列(用于流式响应)
        self.sentence_queue = queue.Queue()

        # 设置 LangChain Agent
        self.setup_agent()

    def listen_and_recognize(self):
        """
        监听语音并进行识别
        Returns:
            str: 识别到的文本
        """
        print("\n请说话(最长15秒)...")
        if not self.recorder.start(duration_sec=15):
            return ""
        self.sound_to_text.get_path(self.recorder.output_file)
        user_input = self.sound_to_text.run()

        if not user_input:
            self.logger.warning("未识别到有效语音输入")
            return ""

        self.logger.info(f"识别到语音: {user_input}")
        return user_input

    def generate_response_streaming(self, user_input, enable_voice=False,thread_id=None):
        """
        使用LangChain Agent生成流式响应
        Args:
            user_input: 用户输入的文本
            enable_voice: 是否启用语音播报
        Returns:
            str: 完整的机器人回复文本
        """
        try:
            # 清空队列
            while not self.sentence_queue.empty():
                self.sentence_queue.get()

            # 创建流式响应处理器
            handler = StreamingResponseHandler(self.sentence_queue, self.text_to_sound)

            # 完整回复文本
            full_response = ""

            # 定义流式处理函数
            def stream_agent_response():
                nonlocal full_response
                try:
                    # 使用 stream 方法获取流式响应
                    for event in self.agent.stream(
                            {"messages": [{"role": "user", "content": user_input}]},
                            config={"configurable": {"thread_id": thread_id}},
                            stream_mode="values"):

                        # 获取最新的消息
                        messages = event.get("messages", [])
                        if messages:
                            last_message = messages[-1]

                            # 只处理 AI 的回复
                            if hasattr(last_message, 'content') and last_message.type == 'ai':
                                chunk_text = last_message.content

                                # 计算新增的文本
                                if len(chunk_text) > len(full_response):
                                    new_text = chunk_text[len(full_response):]
                                    full_response = chunk_text

                                    # 处理新增文本
                                    handler.process_chunk(new_text)

                    # 刷新剩余缓冲区
                    handler.flush_remaining()

                except Exception as e:
                    self.logger.error(f"流式生成出错: {e}")
                    import traceback
                    traceback.print_exc()

            if enable_voice:
                # 启动子线程进行流式生成
                stream_thread = threading.Thread(target=stream_agent_response)
                stream_thread.start()

                # 主线程实时消费队列并播报
                while True:
                    try:
                        sentence = self.sentence_queue.get(block=True, timeout=15)
                        print(f"[播报] {sentence}")

                        # 异步合成并播放语音
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            audio_data = loop.run_until_complete(self.text_to_sound.run(sentence))
                            if audio_data:
                                play_audio(audio_data)
                        finally:
                            loop.close()

                    except queue.Empty:
                        # 队列空且子线程已结束,说明所有内容已处理完毕
                        if not stream_thread.is_alive():
                            break
                        else:
                            continue

                # 等待子线程完成
                stream_thread.join(timeout=10)
            else:
                # 纯文本模式,直接在主线程执行
                stream_agent_response()

            self.logger.info(f"机器人回复: {full_response}")
            return full_response

        except Exception as e:
            self.logger.error(f"生成回复时出错: {e}")
            import traceback
            traceback.print_exc()
            return "抱歉,我在处理您的问题时遇到了一些困难。"

    def chat(self, user_input, enable_voice=False,thread_id=None):
        """
        统一的对话接口

        Args:
            user_input: 用户输入的文本(可以来自键盘或语音识别)
            enable_voice: 是否启用语音播报

        Returns:
            str: 机器人的回复
        """
        return self.generate_response_streaming(user_input, enable_voice=enable_voice,thread_id=thread_id)

    def __del__(self):
        """析构函数,清理资源"""
        try:
            pygame.mixer.quit()
        except:
            pass
