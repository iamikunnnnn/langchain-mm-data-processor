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
    def get_llm_response(self,query,thread_id):
        """
        调用 agent 并返回最终答案
        """
        try:
            result = self.agent.invoke(
                {"messages": [{"role": "user", "content": query}],"verbose":True},
                config= {"configurable": {"thread_id": thread_id}},
                stream_mode ="values"
            )


            # 从结果中提取最后一条消息
            return result["messages"][-1].content
        except ValueError as e:
            return f"调用失败，重新尝试，或重启agent{e}"

    def img2base64(self, img_path):
        try:
            with Image.open(img_path) as image:
                # 1. 处理 RGBA：强制转为 PNG 格式（保留透明）
                if image.mode == "RGBA":
                    img_format = "PNG"
                else:
                    # 非 RGBA 图像保持原格式或默认 JPEG
                    img_format = image.format or "JPEG"

                # 2. 缩放
                image = image.resize((224, 224), Image.Resampling.LANCZOS)

                # 3. 保存到缓冲区（按处理后的格式）
                buffer = BytesIO()
                # PNG 保存时可指定优化级别（1-9，默认6，0无压缩）
                image.save(buffer, format=img_format, optimize=True)

                # 4. 编码为 Base64
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

    base64_image = chatbot.img2base64(r"F:\多模态智能语音助手\my_assistant\test_data\img.png")

    multimodal_content = chatbot.get_prompt(base64_image,"jpeg")
    ret = chatbot.get_llm_response(multimodal_content,thread_id="test_thread")
    print(ret)