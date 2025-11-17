"""
"""

import base64
from io import BytesIO

from PIL import Image

from chatbot import ChatBot
from utils import load_config
config =load_config()

class Cv_Chatbot(ChatBot):
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


    def img2base64(self, img_path):
        try:
            # 打开图片
            with Image.open(img_path) as image:
                image = image.resize((224, 224), Image.Resampling.LANCZOS)
                # 压缩质量并转为JPG
                # 创建字节缓冲区
                buffer = BytesIO()
                # 获取图片格式（如 JPEG、PNG），若无则默认用 JPEG
                img_format = image.format or "JPEG"
                # 将图片保存到缓冲区（保持原格式或指定格式）
                image.save(buffer, format=img_format)
                # 从缓冲区读取二进制数据并编码为 Base64 字符串
                base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

                return base64_str
        except Exception as e:
            print(f"图片转 Base64 失败：{e}")
            return None

    def get_prompt(self, base64_image, text="描述图片"):
        multimodal_content = [
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ]
        return multimodal_content

if __name__ == '__main__':
    chatbot = Cv_Chatbot()
    base64_image = chatbot.img2base64(r"F:\多模态智能语音助手\my_assistant\c2af0b52ef22a3ef221bc81c48542abd.jpg")

    multimodal_content = chatbot.get_prompt(base64_image,"jpeg")
    ret = chatbot.get_llm_response(multimodal_content)
    print(ret)