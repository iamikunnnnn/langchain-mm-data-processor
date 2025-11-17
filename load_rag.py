import os
import asyncio
from typing import List, Any, Tuple
from langchain_community.document_loaders import (
    PyPDFLoader, CSVLoader, TextLoader, UnstructuredMarkdownLoader,
    JSONLoader, BSHTMLLoader, UnstructuredExcelLoader, PythonLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from utils import load_config

config = load_config()

class LoadRag:
    def __init__(self, file_path_list: List):
        self.file_path_list = file_path_list
        # 预初始化所有分割器（避免重复创建）
        self.default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            length_function=len,
            add_start_index=True,

        )
        self.chinese_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "。", "！", "？", "；", "\n", "，", " "],
            chunk_size=200,
            chunk_overlap=40,
            keep_separator=True
        )
        self.python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=300,
            chunk_overlap=50
        )

        self.embedding = HuggingFaceEmbeddings(model_name=config["embedding"]["model"])

    async def _load_and_split_single_file(self, file_path: str) -> Any:
        """加载单个文件并自动分割（合并加载和分割逻辑）"""
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            loader = None
            splitter = self.default_splitter  # 默认分割器

            # 根据文件类型选择加载器和分割器
            if file_extension == ".pdf":
                loader = PyPDFLoader(file_path)
            elif file_extension == ".csv":
                loader = CSVLoader(file_path, encoding="utf-8")
            elif file_extension == ".md":
                loader = UnstructuredMarkdownLoader(file_path,encodings="utf-8")
                splitter = self.chinese_splitter  # Markdown可能含中文
            elif file_extension == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
                splitter = self.chinese_splitter  # TXT优先用中文分割器
            elif file_extension == ".json":
                loader = JSONLoader(file_path, jq_schema=".",text_content=False)
            elif file_extension == ".html":
                loader = BSHTMLLoader(file_path)
            elif file_extension == ".xlsx":
                loader = UnstructuredExcelLoader(file_path)
            elif file_extension == ".py":  # 修正：漏了点（.py）
                loader = PythonLoader(file_path)
                splitter = self.python_splitter  # Python专用分割器
            else:
                return {"status": "error", "msg": f"不支持的文件类型：{file_extension}"}

            # 异步加载文档
            documents = await loader.aload()
            # 根据文件类型分割文档
            split_docs = splitter.split_documents(documents)
            # 提取分割后的内容（保留元数据）
            result = {
                "status": "success",
                "file_path": file_path,
                "original_docs": len(documents),  # 原始文档数量
                "split_chunks": [{"content": doc.page_content, "metadata": doc.metadata} for doc in split_docs]
            }
            return result

        except Exception as e:
            return {"status": "error", "msg": f"处理文件 {file_path} 失败：{str(e)}"}

    async def load_process_files(self) -> tuple[Any]:
        """并发处理所有文件（加载+分割）"""
        tasks = [self._load_and_split_single_file(path) for path in self.file_path_list]
        self.result = await asyncio.gather(*tasks)
        return self.result

    async def load_rag(self):
        texts = []
        for file_result in self.result:
            if file_result["status"] == "success":
                texts.extend([chunk["content"] for chunk in file_result["split_chunks"]])

        if not texts:
            print("没有可用文本生成索引")

            return
        vs = FAISS.from_texts(texts, embedding=self.embedding)
        vs.save_local("faiss_index")



#---------------------------------------------------------------
# 使用示例
async def main():
    file_paths = [
        r"F:\多模态智能语音助手\my_assistant\param_knowledge.md",
        # 可添加更多文件路径："test.pdf", "demo.py", "readme.md" 等
    ]
    processor = LoadRag(file_paths)
    results = await processor.load_process_files()
    await processor.load_rag()
    # 打印处理结果
    for res in results:
        if res["status"] == "success":
            print(f"文件 {res['file_path']} 处理成功：")
            print(f"  原始页数：{res['original_docs']}")
            print(f"  分割后片段数：{len(res['split_chunks'])}")
            print(f"  第一个片段预览：{res['split_chunks'][0]['content'][:100]}...\n")
        else:
            print(f"处理失败：{res['msg']}\n")


if __name__ == '__main__':
    asyncio.run(main())
