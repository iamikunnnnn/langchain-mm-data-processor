"""
语音唤醒功能实现
"""
import json
import logging

import pyaudio
from pypinyin import lazy_pinyin
from vosk import Model, KaldiRecognizer

from utils import load_config

config = load_config()

class WakeWordDetector:
    """
    基于 Vosk 的唤醒词检测器
    支持中文唤醒词，通过拼音匹配实现
    """
    def __init__(self, sample_rate=config["speech_recognition"]["sample_rate"], model_path='models/vosk-model-small-cn-0.22'):
        """
        初始化唤醒词检测器

        Args:
            sample_rate (int): 音频采样率，默认16kHz
            model_path (str): Vosk模型路径
        """
        self.sample_rate = sample_rate
        self.running = False
        self.on_wake_callback = None
        self.logger = logging.getLogger(__name__)

        # 加载 Vosk 模型
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        self.recognizer.SetWords(True)

        # 音频流相关
        self.audio = None
        self.stream = None

        # 帧参数：每帧 20ms
        self.frame_duration = 20
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000)

        # 唤醒词列表
        self.wake_words = config["wake_words"]

    def run(self):
        """
        启动监听线程，实时捕获音频并识别唤醒词
        """
        self.audio = pyaudio.PyAudio()
        self.running = True

        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=config["speech_recognition"]["channels"],
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.frame_size,
            input_device_index=config["speech_recognition"]["input_device_index"]
        )


        self.logger.info("开始监听唤醒词...")
        while self.running:
            data = self.stream.read(self.frame_size, exception_on_overflow=False)
            self._process(data)

        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

    def stop(self):
        """停止唤醒监听"""
        self.running = False


    def _process(self, data):
        """
        处理每帧音频，进行识别并检测唤醒词

        Args:
            data (bytes): 音频帧字节流
        """
        if self.recognizer.AcceptWaveform(data):
            full_result = json.loads(self.recognizer.Result()).get("text", "")
            self._check_wake_word(full_result, is_partial=False)

        partial_text = json.loads(self.recognizer.PartialResult()).get("partial", "")
        self._check_wake_word(partial_text, is_partial=True)

    def _check_wake_word(self, text, is_partial=True):
        """
        检查识别文本中是否包含唤醒词

        Args:
            text (str): 识别结果文本
            is_partial (bool): 是否为部分识别结果
        """
        text = text.strip().replace(" ", "")
        if not text:
            return

        text_pinyin = ''.join(lazy_pinyin(text))
        for word in self.wake_words:
            word_pinyin = ''.join(lazy_pinyin(word))
            if word_pinyin in text_pinyin:
                self.logger.info(f"检测到唤醒词: {word}")
                if callable(self.on_wake_callback):
                    self.on_wake_callback()
                self.recognizer.Reset()
                break