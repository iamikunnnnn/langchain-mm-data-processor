import os
import asyncio
from typing import List, Any
from langchain_community.document_loaders import (
    PyPDFLoader, CSVLoader, TextLoader, UnstructuredMarkdownLoader,
    JSONLoader, BSHTMLLoader, UnstructuredExcelLoader, PythonLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language, RecursiveJsonSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from utils import load_config

config = load_config()


class LoadRag:
    def __init__(self, file_path_list: List, chunk_documents: bool = True):
        """
        :param file_path_list: 文件路径列表
        :param chunk_documents: 是否进行分块，False 表示整篇文章直接存入
        """
        self.file_path_list = file_path_list
        self.chunk_documents = chunk_documents

        # 初始化分割器
        self.default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            length_function=len,
            add_start_index=True,
        )

        self.chinese_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "。", "！", "？", "；", "\n", "，", " "],
            chunk_size=300,
            chunk_overlap=50,
            keep_separator=True,
        )

        self.json_splitter = RecursiveJsonSplitter(
            max_chunk_size=300,
            min_chunk_size=100
        )

        self.python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=300,
            chunk_overlap=50
        )

        self.embedding = HuggingFaceEmbeddings(model_name=config["embedding"]["model"])

    async def _load_and_split_single_file(self, file_path: str) -> Any:
        """加载单个文件并可选分割"""
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            loader = None
            splitter = self.default_splitter

            # 自动选择 Loader
            if file_extension == ".pdf":
                loader = PyPDFLoader(file_path)
            elif file_extension == ".csv":
                loader = CSVLoader(file_path, encoding="utf-8")
            elif file_extension == ".md":
                loader = UnstructuredMarkdownLoader(file_path, encodings="utf-8")
                splitter = self.chinese_splitter
            elif file_extension == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
                splitter = self.chinese_splitter
            elif file_extension == ".json":
                loader = JSONLoader(file_path, jq_schema=".", text_content=False)
                splitter = self.json_splitter
            elif file_extension == ".html":
                loader = BSHTMLLoader(file_path)
            elif file_extension == ".xlsx":
                loader = UnstructuredExcelLoader(file_path)
            elif file_extension == ".py":
                loader = PythonLoader(file_path)
                splitter = self.python_splitter
            else:
                return {"status": "error", "msg": f"不支持的文件类型：{file_extension}"}

            # 异步加载文档
            documents: List[Document] = await loader.aload()

            # 给每个文档增加文件来源
            for doc in documents:
                doc.metadata["source"] = file_path

            # 是否分块
            if self.chunk_documents:
                split_docs = splitter.split_documents(documents)
            else:
                split_docs = documents

            return {
                "status": "success",
                "file_path": file_path,
                "original_docs": len(documents),
                "split_docs": split_docs
            }

        except Exception as e:
            return {"status": "error", "msg": f"处理文件 {file_path} 失败：{str(e)}"}

    async def load_process_files(self):
        """并发处理所有文件"""
        tasks = [self._load_and_split_single_file(path) for path in self.file_path_list]
        self.result = await asyncio.gather(*tasks)
        return self.result

    async def load_rag(self):
        """生成 Chroma 向量库"""
        docs = []
        for file_result in self.result:
            if file_result["status"] == "success":
                docs.extend(file_result["split_docs"])

        if not docs:
            print("没有可用文本生成索引")
            return

        # Chroma 自动持久化，无需 persist()
        vectorstore = Chroma.from_documents(
            docs,
            embedding=self.embedding,
            persist_directory="chroma_index"
        )

        print("Chroma 向量库已完成构建并持久化！")


# ---------------------- 使用示例 ----------------------
async def main():
    file_paths = [
        r"F:\多模态智能语音助手\my_assistant\param_knowledge.md",
    ]
    processor = LoadRag(file_paths, chunk_documents=True)  # chunk_documents=False 可整篇文章不分块
    results = await processor.load_process_files()
    await processor.load_rag()

    # 打印处理结果
    for res in results:
        if res["status"] == "success":
            print(f"文件 {res['file_path']} 处理成功：")
            print(f"  原始文档数：{res['original_docs']}")
            print(f"  分割后片段数：{len(res['split_docs'])}")
            print(f"  第一个片段预览：{res['split_docs'][0].page_content[:100]}...\n")
        else:
            print(f"处理失败：{res['msg']}\n")


if __name__ == '__main__':
    asyncio.run(main())
