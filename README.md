# README.md - 项目文档
"""
# 数据训练AI

这是一个集成了语音唤醒和聊天功能的AI智能体项目，支持多模态交互和智能对话。

## 功能特性

- 语音唤醒：支持自定义唤醒词，如“小布小布”
- 语音识别：将用户语音转换为文本
- 智能对话：基于预训练模型的对话生成
- 模块化设计：代码结构清晰，易于维护和扩展

## 项目结构
ai_agent/
├── main.py          # 主程序入口
├── voice_wake_up.py # 语音唤醒功能
├── chatbot.py       # 聊天功能
├── utils.py         # 辅助函数
├── config.yaml      # 配置文件
├── requirements.txt # 依赖库列表
└── README.md        # 项目文档

## 运行
在config_git.yaml修改对应内容，例如openai_api_key、base_url、gaode_api_key和对应的语音配置
在终端运行 streamlit run UI.py



