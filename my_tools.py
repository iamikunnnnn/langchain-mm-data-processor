import base64
import io
import json
import re
import sqlite3
from datetime import datetime
import aiofiles
import akshare as ak
import joblib
import pandas as pd
from baidusearch.baidusearch import search
from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from data_preprocessing import DataPreprocessor
from stat_analyse import EnhancedDataAnalyzer
from param import get_ML_param, convert_to_json_string
from session_manager import session_store  # 导入刚才写的管理器
from utils import load_config

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# my_tools.py - 修改后
from threading import local

CONFIG = load_config()
thread_local = local()  # 线程局部存储


class columnInput(BaseModel):
    columns: str = Field(..., description="需要操作的列名")


def get_vector_db():
    """每个线程独立的向量数据库实例"""
    if not hasattr(thread_local, 'vector_db'):
        embedding = HuggingFaceEmbeddings(model_name=CONFIG["embedding"]["model"])
        thread_local.vector_db = Chroma(
            embedding_function=embedding,
            persist_directory="chroma_index"
        )
    return thread_local.vector_db
@tool("delete_memory",return_direct=True)
async def delete_memory(config:RunnableConfig) -> str:
    """直接删除所有 Agent 记忆"""
    conn = None
    thread_id = config["configurable"].get("thread_id")
    try:
        conn = sqlite3.connect("./checkpoints.sqlite")
        cursor = conn.cursor()
        cursor.execute("DELETE FROM writes WHERE thread_id=?", (thread_id,))
        cursor.execute("DELETE FROM checkpoints WHERE thread_id=?", (thread_id,))
        conn.commit()
        return "所有记忆已直接删除"
    except sqlite3.Error as e:
        return f"删除失败：{str(e)}"
    finally:
        if conn:
            conn.close()


@tool("get_stock_data_for_model",return_direct=True)
def get_stock_data_for_model(config:RunnableConfig,code: str, start_date: str = '20200101', end_date: str = None) :
    """
    获取股票数据，调用该工具的前置条件是先调用获取当前时间工具
    Args:
        code (str): 股票代码，如 '600036'。
        start_date (str): 开始日期 'YYYYMMDD',非必填，当用户提出上个月上周等事件信息时需要先获取当前时间，然后获取起始时间转换为YYYYMMDD格式再输入。
    Returns:DataFrame
    """

    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")


    try:
        # 1. 获取历史行情 (使用 stock_zh_a_hist 接口，它是目前akshare获取A股日线的主力接口)
        # adjust="qfq" 非常重要：前复权，保证价格连续性，适合建模
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")

        if df.empty:
            raise ValueError("未获取到数据")

        # 2. 数据清洗与重命名
        # akshare返回的列名通常是中文，建议转为英文方便模型处理
        df = df.rename(columns={
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'pct_chg',
            '换手率': 'turnover'
        })

        # 设置日期索引
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)

        # 只保留数值列
        numeric_cols = ['open', 'close', 'high', 'low', 'volume', 'amount', 'amplitude', 'pct_chg', 'turnover']
        df = df[numeric_cols]

        # -------------------------- 3. 特征工程 (Feature Engineering) --------------------------
        # 时间序列模型不仅需要原始价格，通常还需要技术指标作为特征

        # A. 移动平均线 (Moving Averages) - 捕捉趋势
        for window in [5, 10, 20, 60]:
            df[f'ma_{window}'] = df['close'].rolling(window=window).mean()

        # B. 相对强弱指标 (RSI) - 捕捉超买超卖
        # 简单的RSI计算实现
        def calculate_rsi(series, period=14):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        df['rsi_6'] = calculate_rsi(df['close'], 6)
        df['rsi_12'] = calculate_rsi(df['close'], 12)

        # C. MACD (平滑异同移动平均线) - 捕捉动量
        # EMA 计算
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd_dif'] = ema12 - ema26
        df['macd_dea'] = df['macd_dif'].ewm(span=9, adjust=False).mean()
        df['macd_bar'] = 2 * (df['macd_dif'] - df['macd_dea'])

        # D. 布林带 (Bollinger Bands) - 捕捉波动率
        df['boll_mid'] = df['close'].rolling(window=20).mean()
        df['boll_std'] = df['close'].rolling(window=20).std()
        df['boll_upper'] = df['boll_mid'] + 2 * df['boll_std']
        df['boll_lower'] = df['boll_mid'] - 2 * df['boll_std']

        # -------------------------- 4. 数据清理 --------------------------
        # 计算技术指标会产生 NaN (例如MA20的前19天是空的)，建模前需要删除
        df.dropna(inplace=True)

        print(f"数据处理完成，共 {len(df)} 条记录，特征数: {df.shape[1]}")
        df.to_csv(f"temp/stock_data.csv")
        return "已获取相关股票信息，保存至temp/stock_data.csv"

    except Exception as e:
        print(f"发生错误: {e}")

@tool("stock_data_model",return_direct=True)
def stock_data_model():
    """
    当用户需要训练股票模型时调用，只能用于训练预测股票的模型
    :return:
    """
    from stock_model_train import train_model
    stock_model_dict =train_model(f"temp/stock_data.csv")
    return stock_model_dict

@tool("search_knowledge")
def search_knowledge(query: str) -> str:
    """
    从本地知识库检索最相关的内容
    """
    vector_db = get_vector_db()  # 使用线程局部实例
    if not query:
        return "没有提供检索内容"

    # 从 Chroma 检索（与 FAISS 接口一致）
    docs_with_scores = vector_db.similarity_search_with_score(query, k=10)
    if not docs_with_scores:
        return "知识库中没有找到相关内容"

    # Chroma 的 score = 1 - cosine，越小越相关
    threshold = 0.6
    filtered_docs = [
        doc for doc, score in docs_with_scores if score <= threshold
    ]

    if not filtered_docs:
        # 如果一个都没有，就退而求其次返回前 3 条
        filtered_docs = [doc for doc, _ in docs_with_scores[:3]]

    return "\n\n".join(d.page_content for d in filtered_docs)

@tool("save_to_knowledge_base")
def save_to_knowledge_base(key_info: str) -> str:
    """
    用于存储对话中的关键信息到 Chroma 知识库，仅在满足条件时调用：
    1. 信息可复用且事实性；
    2. 信息未存储过或有更新；
    3. 后续对话/任务可能需要用到。
    """
    try:
        # 避免重复存储
        docs = vector_db.similarity_search(key_info, k=1)
        if docs and docs[0].page_content.strip() == key_info.strip():
            return f"无需重复存储：'{key_info}' 已存在于知识库"

        # 存入向量库，并可添加 metadata
        vector_db.add_texts(
            [key_info],
            metadatas=[{"source": "dialogue"}]  # 可自定义 metadata
        )

        return f"成功存储关键信息到知识库：'{key_info}'"
    except Exception as e:
        return f"存储失败：{str(e)}"


@tool("init_data")
def init_data(file_path: str, config: RunnableConfig) -> str:
    """
    初始化数据表。
    Args:
        file_path: 用户输入的文件路径
        config: LangChain 自动传入的配置（包含 thread_id），LLM 不需要生成这个参数
    """
    # 获取当前线程/用户的 ID
    thread_id = config["configurable"].get("thread_id")
    if not thread_id:
        return "系统错误：未检测到会话 ID (thread_id)"

    # 获取该用户的专属状态
    session = session_store.get_session(thread_id)

    # 操作专属状态
    try:
        if session.progressor is not None:
            return "数据表已初始化，请问需要删除现有数据表吗？"

        data = pd.read_csv(file_path)
        # 将 processor 存入用户的 session，而不是全局变量
        session.progressor = DataPreprocessor(data)
        session.enhanced_data_analyzer = EnhancedDataAnalyzer(data=data)
        return "已读取数据，请问您有什么分析需求呢?"
    except Exception as e:
        return f"初始化失败：{str(e)}"


@tool("clear_data", return_direct=True)
def clear_data(config: RunnableConfig, tmp=None) -> str:
    """删除已初始化的数据表实例"""
    thread_id = config["configurable"].get("thread_id")
    # 直接调用管理器清理
    session_store.clear_session(thread_id)
    return "数据表实例已成功删除，内存已释放"
@tool("get_current_time", return_direct=True)
def get_current_time(a=None):
    """
    获取当前系统时间，无需输入参数
    :return: 返回字符串，字符串为当前时间，格式为%Y年%m月%d日 %H时%M分%S秒
    """
    return f"当前时间：{datetime.now().strftime('%Y年%m月%d日 %H时%M分%S秒')}"


def clean_abstract(text):
    """清理摘要文本中的多余换行和空格"""
    # 替换换行符为空格
    text = text.replace('\n', ' ')
    # 替换连续多个空格为单个空格
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


@tool("web_search")
def web_search(query):
    """通过网络搜索获取信息，输入为搜索关键词"""
    # 直接搜索无需API密钥
    results = search(query, num_results=10)  # 找回十条

    for item in results:
        item['abstract'] = clean_abstract(item['abstract'])

    return results
#
# @tool("dataframe_analyse", return_direct=True)
# def dataframe_analyse(tmp=None):
#     """
#     对表格数据进行分析,当用户存在对数据的基础分析意图时调用,同样只需调用一次，无需输入任何参数
#     :return: {"形状","列名","空值数量","重复行数量"}
#     """
#     global ANALYZE_RESULT
#     from stat_analyse import filter_normal_stats
#     if isinstance(progressor, DataPreprocessor):
#         input_info = filter_normal_stats()
#         # 1. 初始化 (使用单个 LLM)
#         llm = ChatOpenAI(
#             base_url="https://api.siliconflow.cn/v1",
#             openai_api_key="sk-xtgeyahfmjxvrxjygvnwfywbezskstroipqofybruqldkgor",
#             model= "Qwen/Qwen3-VL-32B-Thinking"
#         )
#
#
#         # 3. Prompt 模板
#         prompt_template = PromptTemplate(
#             template="""
#             你是一名统计学专家，专门用于分析数据，你需要先完整复述输入的所有内容，然后在最后附上自己的分析判断
#             核心任务是：
#                     1. 首先最重要的，你要先完整复述内容。
#                     2. 整合信息并分析后认为统计结果。
#             提供指标：
#                 {input_info}
#             """,
#             input_variables =["input_info"]
#         )
#         # 4. 构建链条 (单次调用)
#         chain = prompt_template | llm | StrOutputParser()
#         # 5. 执行链条
#         llm_output = chain.invoke({
#             "input_info": input_info,
#
#         })
#         ANALYZE_RESULT = input_info
#
#         return llm_output

@tool("get_data_info")
def get_data_info(config:RunnableConfig):
    """
    获取数据的基本信息，如：形状、列名、空值数量、重复行数量、前二行数据
    :return:
    """
    # 获取当前线程/用户的 ID
    thread_id = config["configurable"].get("thread_id")
    if not thread_id:
        return "系统错误：未检测到会话 ID (thread_id)"

    # 获取该用户的专属状态
    session = session_store.get_session(thread_id)
    try:
        if isinstance(session.enhanced_data_analyzer,EnhancedDataAnalyzer):
            return session.data_info()
    except Exception as e:
        return "暂未读取到数据，需要我读取数据吗？"
@tool("get_multicollinearity_info")
def get_multicollinearity_info(config:RunnableConfig):
    """
    统计分析工具，获取各个列的VIF多重共线性检验结果
    :return:
    """

    try:
        # 获取当前线程/用户的 ID
        thread_id = config["configurable"].get("thread_id")
        if not thread_id:
            return "系统错误：未检测到会话 ID (thread_id)"

        # 获取该用户的专属状态
        session = session_store.get_session(thread_id)
        if isinstance(session.enhanced_data_analyzer,EnhancedDataAnalyzer):
            return session.enhanced_data_analyzer.multicollinearity_info()
    except Exception as e:
        "暂未读取到数据，需要我读取数据吗？"
@tool("get_homogeneity_info")
def get_homogeneity_info(config:RunnableConfig):
    """
    统计分析工具，获取各个列的方差齐性检验
    :return:
    """
    try:
        # 获取当前线程/用户的 ID
        thread_id = config["configurable"].get("thread_id")
        if not thread_id:
            return "系统错误：未检测到会话 ID (thread_id)"

        # 获取该用户的专属状态
        session = session_store.get_session(thread_id)
        if isinstance(session.enhanced_data_analyzer,EnhancedDataAnalyzer):
            return session.enhanced_data_analyzer.homogeneity_info()
    except Exception as e:
        "暂未读取到数据，需要我读取数据吗？"
@tool("get_correlation_info")
def get_correlation_info(config:RunnableConfig):
    """
    统计分析工具，获取各个列的相关性信息
    """
    try:
        # 获取当前线程/用户的 ID
        thread_id = config["configurable"].get("thread_id")
        if not thread_id:
            return "系统错误：未检测到会话 ID (thread_id)"

        # 获取该用户的专属状态
        session = session_store.get_session(thread_id)
        if isinstance(session.enhanced_data_analyzer,EnhancedDataAnalyzer):
            return session.enhanced_data_analyzer.correlation_analysis()
    except Exception as e:
        "暂未读取到数据，需要我读取数据吗？"
@tool("get_stats_info")
def get_stats_info(config:RunnableConfig):
    """
    统计分析工具，获取列统计信息，例如mean、median、std、min、max、25%_quantile、75%_quantile、skewness
    """
    try:
        # 获取当前线程/用户的 ID
        thread_id = config["configurable"].get("thread_id")
        if not thread_id:
            return "系统错误：未检测到会话 ID (thread_id)"

        # 获取该用户的专属状态
        session = session_store.get_session(thread_id)
        if isinstance(session.enhanced_data_analyzer,EnhancedDataAnalyzer):
            return session.enhanced_data_analyzer.stats_info()
    except Exception as e:
        "暂未读取到数据，需要我读取数据吗？"




@tool("get_dummy", args_schema=columnInput)
def get_dummy(columns,config:RunnableConfig):
    """
    对表格数据独热编码,当认为用户存在独热编码意图时调用
    :param columns: 需要进行独热编码的列
    """
    column_list = [c.strip().strip("'").strip('"') for c in columns.split(",")]
    # 获取当前线程/用户的 ID
    thread_id = config["configurable"].get("thread_id")
    if not thread_id:
        return "系统错误：未检测到会话 ID (thread_id)"

    # 获取该用户的专属状态
    session = session_store.get_session(thread_id)
    if isinstance(session.progressor, DataPreprocessor):
        session.progressor.get_dummy_data(column_list)
        session.progressor.save_data(rf"F:\多模态智能语音助手\my_assistant\temp\temp{thread_id}.csv")
        return rf"已独热编码,结果保存至F:\多模态智能语音助手\my_assistant\temp\temp{thread_id}.csv"


@tool("fill_null_with_mean", args_schema=columnInput)
def fill_null_with_mean(columns,config:RunnableConfig):
    """
    均值填充空值
    :param columns:需要填充空值的行:
    """
    column_list = [
        c.strip().strip("'").strip('"')
        for c in columns.replace(";", ",").split(",")
        if c.strip()
    ]
    try:
        # 获取当前线程/用户的 ID
        thread_id = config["configurable"].get("thread_id")
        if not thread_id:
            return "系统错误：未检测到会话 ID (thread_id)"

        # 获取该用户的专属状态
        session = session_store.get_session(thread_id)
        if isinstance(session.progressor, DataPreprocessor):
            session.progressor.fill_null_with_mean(columns=column_list)
            session.progressor.save_data(rf"F:\多模态智能语音助手\my_assistant\temp\temp{thread_id}.csv")
            return rf"已均值填充,结果保存至F:\多模态智能语音助手\my_assistant\temp\temp{thread_id}.csv"
    except ValueError as e:
        return f"错误：{e}"


@tool("drop_null_rows", args_schema=columnInput)
def drop_null_rows(columns,config:RunnableConfig):
    """
    直接删除包含空值的行
    :param columns:需要填充空值的行
    """
    column_list = [
        c.strip().strip("'").strip('"')
        for c in columns.replace(";", ",").split(",")
        if c.strip()
    ]
    try:
        # 获取当前线程/用户的 ID
        thread_id = config["configurable"].get("thread_id")
        if not thread_id:
            return "系统错误：未检测到会话 ID (thread_id)"

        # 获取该用户的专属状态
        session = session_store.get_session(thread_id)
        if isinstance(session.progressor, DataPreprocessor):
            session.progressor.drop_null_rows(columns=column_list)
            session.progressor.save_data(rf"F:\多模态智能语音助手\my_assistant\temp\temp{thread_id}.csv")
            return rf"已删除带空值的行,结果保存至F:\多模态智能语音助手\my_assistant\temp\temp{thread_id}.csv"
    except ValueError as e:
        return f"错误：{e}"



@tool("get_null_info", args_schema=columnInput)
def get_null_info(columns,config:RunnableConfig):
    """
    获取空值详细信息
    :param columns:需要获取空值详细信息的行
    """
    column_list = [
        c.strip().strip("'").strip('"')
        for c in columns.replace(";", ",").split(",")
        if c.strip()
    ]
    try:
        # 获取当前线程/用户的 ID
        thread_id = config["configurable"].get("thread_id")
        if not thread_id:
            return "系统错误：未检测到会话 ID (thread_id)"

        # 获取该用户的专属状态
        session = session_store.get_session(thread_id)
        if isinstance(session.progressor, DataPreprocessor):
            session.progressor.get_null_info(columns=column_list)
            session.progressor.save_data(rf"F:\多模态智能语音助手\my_assistant\temp\temp{thread_id}.csv")
            return rf"已删除空值,结果保存至F:\多模态智能语音助手\my_assistant\temp\temp{thread_id}.csv"
    except ValueError as e:
        return f"错误：{e}"


@tool("Label_Encoding", args_schema=columnInput)
def Label_Encoding(columns,config:RunnableConfig):
    """
    将指定列进行标签编码（Label Encoding），将类别转换为整数
    :param columns: 需要编码的列
    :return: 编码后的 DataFrame
    """
    column_list = [
        c.strip().strip("'").strip('"')
        for c in columns.replace(";", ",").split(",")
        if c.strip()
    ]
    try:
        # 获取当前线程/用户的 ID
        thread_id = config["configurable"].get("thread_id")
        if not thread_id:
            return "系统错误：未检测到会话 ID (thread_id)"

        # 获取该用户的专属状态
        session = session_store.get_session(thread_id)
        if isinstance(session.progressor, DataPreprocessor):
            session.progressor.Label_Encoding(columns=column_list)
            session.progressor.save_data(rf"F:\多模态智能语音助手\my_assistant\temp\temp{thread_id}.csv")
            return rf"已标签编码,结果保存至F:\多模态智能语音助手\my_assistant\temp\temp{thread_id}.csv"
    except ValueError as e:
        return f"错误：{e}"


@tool("Standard_Scaling", args_schema=columnInput)
def Standard_Scaling(columns,config:RunnableConfig):
    """
    对指定列进行标准化（StandardScaler），将数据缩放到均值为0、方差为1的分布
    :param columns: 需要标准化的列，可以是单列名或列名列表
    :return: 标准化后的 DataFrame
    """
    column_list = [
        c.strip().strip("'").strip('"')
        for c in columns.replace(";", ",").split(",")
        if c.strip()
    ]
    try:
        # 获取当前线程/用户的 ID
        thread_id = config["configurable"].get("thread_id")
        if not thread_id:
            return "系统错误：未检测到会话 ID (thread_id)"

        # 获取该用户的专属状态
        session = session_store.get_session(thread_id)
        if isinstance(session.progressor, DataPreprocessor):
            session.progressor.Standard_Scaling(columns=column_list)
            session.progressor.save_data(rf"F:\多模态智能语音助手\my_assistant\temp\temp{thread_id}.csv")
            return rf"标准化,结果保存至F:\多模态智能语音助手\my_assistant\temp\temp{thread_id}.csv"
    except ValueError as e:
        return f"错误：{e}"


@tool("drop_columns", args_schema=columnInput)
def drop_columns(columns,config:RunnableConfig):
    """
    删除输入的列名对应的列
    :param columns: 待删除的列名（单个字符串或列表）
    :return: 成功返回提示信息，失败返回具体错误原因
    """
    column_list = [
        c.strip().strip("'").strip('"')
        for c in columns.replace(";", ",").split(",")
        if c.strip()
    ]
    try:
        # 获取当前线程/用户的 ID
        thread_id = config["configurable"].get("thread_id")
        if not thread_id:
            return "系统错误：未检测到会话 ID (thread_id)"

        # 获取该用户的专属状态
        session = session_store.get_session(thread_id)
        if isinstance(session.progressor, DataPreprocessor):
            session.progressor.drop_columns(columns=column_list)
            session.progressor.save_data(rf"F:\多模态智能语音助手\my_assistant\temp\temp{thread_id}.csv")
            return rf"已删除列"
    except ValueError as e:
        return f"错误：{e}"


def fig_to_base64(fig):
    buf = io.BytesIO()
    # 直接用 fig.savefig()，支持 bbox_inches 参数
    fig.savefig(buf, format='png', bbox_inches='tight')  # 这里可以用 bbox_inches
    buf.seek(0)
    img_bytes = buf.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')
@tool("detect_outliers_IQR")
def detect_outliers_IQR(columns,config:RunnableConfig):
    """
    使用IQR方法检测异常值
    :param columns: 要检查的列，None表示检查所有数值列
    """
    try:
        column_list = [
            c.strip().strip("'").strip('"')
            for c in columns.replace(";", ",").split(",")
            if c.strip()
        ]
        # 获取当前线程/用户的 ID
        thread_id = config["configurable"].get("thread_id")
        if not thread_id:
            return "系统错误：未检测到会话 ID (thread_id)"

        # 获取该用户的专属状态
        session = session_store.get_session(thread_id)
        if isinstance(session.progressor, DataPreprocessor):
            session.progressor.detect_outliers_iqr(column_list)
            session.progressor.save_data(rf"F:\多模态智能语音助手\my_assistant\temp\temp{thread_id}.csv")
            return rf"已处理异常值,结果保存至F:\多模态智能语音助手\my_assistant\temp\temp{thread_id}.csv"
    except ValueError as e:
        return f"错误：{e}"
@tool("remove_outliers_IQR")
def remove_outliers_IQR(columns,config:RunnableConfig):
    """
    删除IQR方法检测到的异常值
    :param columns: 要处理的列，None表示处理所有数值列
    """
    try:
        column_list = [
            c.strip().strip("'").strip('"')
            for c in columns.replace(";", ",").split(",")
            if c.strip()
        ]
        # 获取当前线程/用户的 ID
        thread_id = config["configurable"].get("thread_id")
        if not thread_id:
            return "系统错误：未检测到会话 ID (thread_id)"

        # 获取该用户的专属状态
        session = session_store.get_session(thread_id)
        if isinstance(session.progressor, DataPreprocessor):
            session.progressor.remove_outliers_iqr(column_list)
            session.progressor.save_data(rf"F:\多模态智能语音助手\my_assistant\temp\temp{thread_id}.csv")
            return rf"已处理异常值,结果保存至F:\多模态智能语音助手\my_assistant\temp\temp{thread_id}.csv"
    except ValueError as e:
        return f"错误：{e}"

@tool("cap_outliers")
def cap_outliers(columns,config:RunnableConfig):
    """
    使用盖帽法处理异常值（截断到边界值）
    :param columns: 要处理的列

    """
    try:
        column_list = [
            c.strip().strip("'").strip('"')
            for c in columns.replace(";", ",").split(",")
            if c.strip()
        ]
        # 获取当前线程/用户的 ID
        thread_id = config["configurable"].get("thread_id")
        if not thread_id:
            return "系统错误：未检测到会话 ID (thread_id)"

        # 获取该用户的专属状态
        session = session_store.get_session(thread_id)
        if isinstance(session.progressor, DataPreprocessor):
            session.progressor.cap_outliers(column_list)
            session.progressor.save_data(rf"F:\多模态智能语音助手\my_assistant\temp\temp{thread_id}.csv")
            return rf"已处理异常值,结果保存至F:\多模态智能语音助手\my_assistant\temp\temp{thread_id}.csv"
    except ValueError as e:
        return f"错误：{e}"
@tool("normalize_minmax")
def normalize_minmax(columns,config:RunnableConfig):
    """
    对指定列进行Min-Max归一化
    :param columns: 需要归一化的列

    """
    try:
        column_list = [
            c.strip().strip("'").strip('"')
            for c in columns.replace(";", ",").split(",")
            if c.strip()
        ]
        # 获取当前线程/用户的 ID
        thread_id = config["configurable"].get("thread_id")
        if not thread_id:
            return "系统错误：未检测到会话 ID (thread_id)"

        # 获取该用户的专属状态
        session = session_store.get_session(thread_id)
        if isinstance(session.progressor, DataPreprocessor):
            session.progressor.normalize_minmax(column_list)
            session.progressor.save_data(rf"F:\多模态智能语音助手\my_assistant\temp\temp{thread_id}.csv")
            return rf"已创建滞后特征,结果保存至F:\多模态智能语音助手\my_assistant\temp\temp{thread_id}.csv"
    except ValueError as e:
        return f"错误：{e}"
@tool("create_lag_features")
def create_lag_features(columns,config:RunnableConfig):
    """
    创建滞后特征（用于时间序列）
    :param columns: 要创建滞后特征的列

    """
    try:
        column_list = [
            c.strip().strip("'").strip('"')
            for c in columns.replace(";", ",").split(",")
            if c.strip()
        ]
        # 获取当前线程/用户的 ID
        thread_id = config["configurable"].get("thread_id")
        if not thread_id:
            return "系统错误：未检测到会话 ID (thread_id)"

        # 获取该用户的专属状态
        session = session_store.get_session(thread_id)
        if isinstance(session.progressor, DataPreprocessor):
            session.progressor.create_lag_features(column_list)
            session.progressor.save_data(rf"F:\多模态智能语音助手\my_assistant\temp\temp{thread_id}.csv")
            return rf"已创建滞后特征,结果保存至F:\多模态智能语音助手\my_assistant\temp\temp{thread_id}.csv"
    except ValueError as e:
        return f"错误：{e}"

@tool("create_rolling_features")
def create_rolling_features(columns,window,agg_funcs,config:RunnableConfig):
    """
    创建滚动窗口特征
    :param columns: 要处理的列
    :param window: 窗口大小
    :param agg_funcs: 聚合函数列表，如['mean', 'std', 'min', 'max']
    :return: self
    """
    try:
        column_list = [
            c.strip().strip("'").strip('"')
            for c in columns.replace(";", ",").split(",")
            if c.strip()
        ]
        # 获取当前线程/用户的 ID
        thread_id = config["configurable"].get("thread_id")
        if not thread_id:
            return "系统错误：未检测到会话 ID (thread_id)"

        # 获取该用户的专属状态
        session = session_store.get_session(thread_id)
        if isinstance(session.progressor, DataPreprocessor):
            session.progressor.create_rolling_features(column_list,window,agg_funcs)
            session.progressor.save_data(rf"F:\多模态智能语音助手\my_assistant\temp\temp{thread_id}.csv")
            return rf"已创建滚动窗口特征,结果保存至F:\多模态智能语音助手\my_assistant\temp\temp{thread_id}.csv"
    except ValueError as e:
        return f"错误：{e}"
@tool("bin_data")
def bin_data(columns,bins,labels,config:RunnableConfig):
    """
    对连续变量进行分箱处理
    :param columns: 要分箱的列
    :param bins: 分箱数量
    :param labels: 分箱标签
    :return: self
    """
    try:
        column_list = [
            c.strip().strip("'").strip('"')
            for c in columns.replace(";", ",").split(",")
            if c.strip()
        ]
        # 获取当前线程/用户的 ID
        thread_id = config["configurable"].get("thread_id")
        if not thread_id:
            return "系统错误：未检测到会话 ID (thread_id)"

        # 获取该用户的专属状态
        session = session_store.get_session(thread_id)
        if isinstance(session.progressor, DataPreprocessor):
            session.progressor.bin_data(column_list,bins,labels)
            session.progressor.save_data(rf"F:\多模态智能语音助手\my_assistant\temp\temp{thread_id}.csv")
            return rf"分箱处理完成,结果保存至F:\多模态智能语音助手\my_assistant\temp\temp{thread_id}.csv"
    except ValueError as e:
        return f"错误：{e}"
@tool("filter_by_condition")
def filter_by_condition(condition,config:RunnableConfig):
    """
    根据条件表达式筛选数据
    :param condition: 条件表达式，如 "age > 18 and income < 50000"
    """
    try:
        # 获取当前线程/用户的 ID
        thread_id = config["configurable"].get("thread_id")
        if not thread_id:
            return "系统错误：未检测到会话 ID (thread_id)"

        # 获取该用户的专属状态
        session = session_store.get_session(thread_id)
        if isinstance(session.progressor, DataPreprocessor):
            session.progressor.filter_by_condition(condition)
            session.progressor.save_data(rf"F:\多模态智能语音助手\my_assistant\temp\temp{thread_id}.csv")
            return rf"筛选完成,结果保存至F:\多模态智能语音助手\my_assistant\temp\temp{thread_id}.csv"
    except ValueError as e:
        return f"错误：{e}"

@tool("merge_rare_categories")
def merge_rare_categories(columns,threshold,new_category,config:RunnableConfig):
    """
    合并低频类别
    :param columns: 要处理的分类列
    :param threshold: 频率阈值，低于此值的类别会被合并
    :param new_category: 合并后的新类别名称
    """
    try:
        column_list = [
            c.strip().strip("'").strip('"')
            for c in columns.replace(";", ",").split(",")
            if c.strip()
        ]
        # 获取当前线程/用户的 ID
        thread_id = config["configurable"].get("thread_id")
        if not thread_id:
            return "系统错误：未检测到会话 ID (thread_id)"

        # 获取该用户的专属状态
        session = session_store.get_session(thread_id)
        if isinstance(session.progressor, DataPreprocessor):
            session.progressor.merge_rare_categories(column_list,threshold,new_category)
            session.progressor.save_data(rf"F:\多模态智能语音助手\my_assistant\temp\temp{thread_id}.csv")
            return rf"合并低频类别,结果保存至F:\多模态智能语音助手\my_assistant\temp\temp{thread_id}.csv"
    except ValueError as e:
        return f"错误：{e}"

@tool("log_transform")
def log_transform(columns,config:RunnableConfig):
    """
    对数变换（用于处理偏态分布）
    :param columns: 要变换的列
    :return: self
    """
    try:
        column_list = [
            c.strip().strip("'").strip('"')
            for c in columns.replace(";", ",").split(",")
            if c.strip()
        ]
        # 获取当前线程/用户的 ID
        thread_id = config["configurable"].get("thread_id")
        if not thread_id:
            return "系统错误：未检测到会话 ID (thread_id)"

        # 获取该用户的专属状态
        session = session_store.get_session(thread_id)
        if isinstance(session.progressor, DataPreprocessor):
            session.progressor.log_transform(column_list)
            session.progressor.save_data(rf"F:\多模态智能语音助手\my_assistant\temp\temp{thread_id}.csv")
            return rf"对数变换,结果保存至F:\多模态智能语音助手\my_assistant\temp\temp{thread_id}.csv"
    except ValueError as e:
        return f"错误：{e}"

@tool("fill_null_with_knn")
def fill_null_with_knn(columns,n_neighbors,config:RunnableConfig):
    """
    使用KNN算法填充空值
    :param columns: 要处理的列，None表示处理所有数值列
    :param n_neighbors: KNN的邻居数量，默认5
    """
    try:
        column_list = [
            c.strip().strip("'").strip('"')
            for c in columns.replace(";", ",").split(",")
            if c.strip()
        ]
        # 获取当前线程/用户的 ID
        thread_id = config["configurable"].get("thread_id")
        if not thread_id:
            return "系统错误：未检测到会话 ID (thread_id)"

        # 获取该用户的专属状态
        session = session_store.get_session(thread_id)
        if isinstance(session.progressor, DataPreprocessor):
            session.progressor.log_transform(column_list,n_neighbors)
            session.progressor.save_data(rf"F:\多模态智能语音助手\my_assistant\temp\temp{thread_id}.csv")
            return rf"KNN填充空值,结果保存至F:\多模态智能语音助手\my_assistant\temp\temp{thread_id}.csv"
    except ValueError as e:
        return f"错误：{e}"


@tool("machine_learning_train",return_direct=True)
def machine_learning_train(query: str,config:RunnableConfig):
    """
    1. 当需要训练模型时，且参数明确时，调用该工具，固定参数的单次模型训练。输入只有一个{query}，需要结合用户需求完成query输入，query的内容需包含：model_name（模型名称）、X_columns（特征列名，用逗号分隔）、y_column（目标列名）、mode（'分类'或'回归'）、model_param(模型对应需要的参数)等信息。不可与machine_learning_train_BayesSearch同时调用。"
    2. 该工具和machine_learning_train_BayesSearch只能调用一个，不可一起调用
    :param query: 用户关于模型训练的查询（例如："用随机森林做回归，特征列是Age,Score，目标列是Salary，参数n_estimators=100"）
    """
    # 获取当前线程/用户的 ID
    thread_id = config["configurable"].get("thread_id")
    if not thread_id:
        return "系统错误：未检测到会话 ID (thread_id)"

    # 获取该用户的专属状态
    session = session_store.get_session(thread_id)

    data = session.progressor.data
    param_list = get_ML_param(query)
    param_dict = convert_to_json_string(param_list)
    param_dict = json.loads(param_dict)
    model_name = param_dict["model_name"]
    X_columns = param_dict["X_columns"].split(',')  # 拆成列表
    y_column = param_dict["y_column"]
    mode = param_dict["mode"]
    model_param = param_dict["model_param"]
    model_param = model_param.replace("\\", "")  # 注意：这里要写两个反斜杠表示一个实际的反斜杠
    model_param = json.loads(model_param)
    from machine_learning_model import model_choice
    model = model_choice(data=data, model_name=model_name, X_columns=X_columns, y_column=y_column, mode=mode,
                         model_param=model_param)

    model.init_model()
    model.my_train_test_split()
    model.train_bayesian_search()
    model.download_model("./machine_learning_model")
    model_name, dict = model.evaluate()

    return f"{model_name}模型训练完毕，已保存至./machine_learning_model{thread_id}。模型评估结果为\n{dict}"

@tool(
    "machine_learning_train_BayesSearch",
    return_direct=True)
def machine_learning_train_BayesSearch(query: str,config:RunnableConfig):
    """
    贝叶斯搜索调参工具，用于参数不明确时的模型训练。固定参数的单次模型训练。输入只有一个{query}，需要结合用户需求完成query输入，query的内容需包含：model_name（模型名称）、X_columns（特征列名，用逗号分隔）、y_column（目标列名）、mode（'分类'或'回归'）等信息。不可与machine_learning_train_BayesSearch同时调用。"
    输入示例："用随机森林做回归，特征列是Age,Score，目标列是Salary"
    约束：与machine_learning_train互斥，只能调用其中一个。
    """
    # 获取当前线程/用户的 ID
    thread_id = config["configurable"].get("thread_id")
    if not thread_id:
        return "系统错误：未检测到会话 ID (thread_id)"
    # 获取该用户的专属状态
    session = session_store.get_session(thread_id)

    data = session.progressor.data
    param_list = get_ML_param(query)
    param_dict = convert_to_json_string(param_list)
    param_dict = json.loads(param_dict)
    model_name = param_dict["model_name"]
    X_columns = param_dict["X_columns"].split(',')  # 拆成列表
    y_column = param_dict["y_column"]
    mode = param_dict["mode"]
    model_param = param_dict["model_param"]
    model_param = model_param.replace("\\", "")  # 注意：这里要写两个反斜杠表示一个实际的反斜杠
    model_param = json.loads(model_param)
    from machine_learning_model import model_choice
    model = model_choice(data=data, model_name=model_name, X_columns=X_columns, y_column=y_column, mode=mode,
                         model_param=model_param)

    model.init_model()
    model.my_train_test_split()
    best_params = model.train_bayesian_search()
    model.download_model(f"./machine_learning_model_{thread_id}")
    model_name, dict = model.evaluate()

    return f"{model_name}模型训练完毕，已保存至./machine_learning_model。模型评估结果为\n{dict}，最优参数为{best_params}"

def visualization():
    """
    暂未想到快速干净的可视化途径，故在此处直接复用之前global的model类
    :return:
    """
    model = joblib.load("random_forest_model.pkl")
    if model is None:
        return False,None,None,None
    else:
        binary_fig,_ = model.plot_boundary()
        fit_true_fig ,_ =model.plot_fit_true()
        predict_true_fig ,_ =model.plot_true_pred()
        binary_fig = fig_to_base64(binary_fig)
        fit_true_fig = fig_to_base64(fit_true_fig)
        predict_true_fig = fig_to_base64(predict_true_fig)
        return True,binary_fig,fit_true_fig,predict_true_fig


def get_tools():
    return [
            get_current_time,
            web_search,
            # dataframe_analyse,
            get_dummy,
            init_data,
            clear_data,
            fill_null_with_mean,
            search_knowledge,
            drop_null_rows,
            get_null_info,
            Label_Encoding,
            Standard_Scaling,
            drop_columns,
            machine_learning_train,
            machine_learning_train_BayesSearch,
            get_stock_data_for_model,
            stock_data_model,
            save_to_knowledge_base,
            delete_memory,
            fill_null_with_knn,
            log_transform,
            merge_rare_categories,
            filter_by_condition,
            bin_data,
            create_rolling_features,
            create_lag_features,
            normalize_minmax,
            cap_outliers,
            remove_outliers_IQR,
            detect_outliers_IQR,
            ]
