import base64
import io
import json
import re
from datetime import datetime
from typing import Optional

import akshare as ak
import pandas as pd
import requests
from baidusearch.baidusearch import search
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from data_preprocessing import DataPreprocessor
from machine_learning_model import model_choice
from param import get_ML_param, convert_to_json_string
from utils import load_config

config = load_config()

# 声明模块级变量，用于存储模型实例和预处理实例
progressor = None
trained_model: Optional[model_choice] = None  # 类型注解：可选的Model实例

class columnInput(BaseModel):
    columns: str = Field(..., description="需要填充空值的列名")


# 加载FAISS知识库
embedding = HuggingFaceEmbeddings(model_name=config["embedding"]["model"])

vector_db = FAISS.load_local("faiss_index", embedding,
                             allow_dangerous_deserialization=True)  # 加载已存储的索引


@tool("get_stock_data_for_model",description="获取股票数据",return_direct=True)
def get_stock_data_for_model(code: str, start_date: str = '20200101', end_date: str = None) :
    """
    获取股票数据
    Args:
        code (str): 股票代码，如 '600036'。
        start_date (str): 开始日期 'YYYYMMDD',非必填。
    Returns:DataFrame
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    print(f"正在获取 {code} 的历史数据...")

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

@tool("search_knowledge", description="必须用于从本地知识库获取答案,该工具必须优先使用")
def search_knowledge(query: str) -> str:
    """
    从本地知识库检索最相关的内容，该工具必须优先使用
    输入：用户查询文本。
    输出：检索出的知识条目文本。
    """
    if not query:
        return "没有提供检索内容"

    docs_with_scores = vector_db.similarity_search_with_score(query, k=20)
    if not docs_with_scores:
        return "知识库中没有找到相关内容"
    threshold = 0.5
    # 2. 根据阈值筛选结果（分数 < 阈值 表示相似度较高）
    filtered_docs = [doc for doc, score in docs_with_scores if score < threshold]

    return "\n\n".join(d.page_content for d in filtered_docs)


@tool("init_data", return_direct=True)
def init_data(file_path):
    """
    每当用户传入临时文件路径时自动输入路径尝试初始化数据表.
    :param file_path:用户输入的路径
    """

    global progressor
    try:
        if progressor is not None:
            return "数据表已初始化，该函数只能执行一次，请问需要删除现有数据表吗？"
        else:
            data = pd.read_csv(file_path)
            progressor = DataPreprocessor(data)
        return "已读取数据，请问您有什么分析需求呢?需要让我先初步分析一下吗（我会看一下这个数据的一些基本信息和前五行数据）"
    except Exception as e:
        return f"初始化失败：{str(e)}"


@tool("clear_data", return_direct=True)
def clear_data(tmp=None) -> str:
    """当用户需要切换处理的数据时，手动删除已初始化的数据表实例，无需输入任何参数"""
    global progressor
    if progressor is None:
        return "当前没有已初始化的数据表"
    # 主动删除全局变量并释放内存
    del progressor
    # 可选：删除后重置为 None（方便后续判断状态）
    progressor = None
    return "数据表实例已成功删除"

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


def get_coordinate(address):
    """
    获取地点坐标
    :param address: 需要提取坐标的地点
    :return: 起始地点对应的坐标
    """
    api_key = config["api"]["gaode_api_key"]
    url = "https://restapi.amap.com/v3/geocode/geo"
    params = {
        "key": api_key,
        "address": address
    }
    response = requests.get(url, params=params)

    data = response.json()
    if data["status"] == "1" and data["count"] != "0":
        location = data["geocodes"][0]["location"]
        return f"{location}"
    else:
        return None

@tool("dataframe_analyse", return_direct=True)
def dataframe_analyse(tmp=None):
    """
    对表格数据进行分析,当用户存在对数据的基础分析意图时调用,同样只需调用一次，无需输入任何参数
    :return: {"形状","列名","空值数量","重复行数量"}
    """
    if isinstance(progressor, DataPreprocessor):
        input_info = progressor.get_basic_info()
        # 1. 初始化 (使用单个 LLM)
        llm = ChatOpenAI(
            base_url="https://api.siliconflow.cn/v1",
            openai_api_key="sk-xtgeyahfmjxvrxjygvnwfywbezskstroipqofybruqldkgor",
            model= "Qwen/Qwen3-VL-32B-Thinking"
        )


        # 3. Prompt 模板
        prompt_template = PromptTemplate(
            template="""
            你是一名统计学专家，专门用于分析数据，你需要判断数据的各个指标的意义，保留有意义的信息，不需要分析，而是做信息过滤。
            核心任务是：
                    1. 首先最重要的，你要先完整复述：all_cols、前二行数据、形状
                    2. 整合信息并展示你分析后认为有用的信息。
                    4. 剔除冗余 / 无意义信息;

            核心目标：
                1.你传递出去的信息足以让别人在看不到数据的情况下，能够完成接下来的统计学建模或者机器学习建模。

            提供指标：
                {input_info}    
            """,
            input_variables =["input_info"]
        )
        # 4. 构建链条 (单次调用)
        chain = prompt_template | llm | StrOutputParser()
        # 5. 执行链条
        llm_output = chain.invoke({
            "input_info": input_info,

        })
        return llm_output





@tool("get_dummy", args_schema=columnInput)
def get_dummy(columns):
    """
    对表格数据独热编码,当认为用户存在独热编码意图时调用
    :param columns: 需要进行独热编码的列
    """
    column_list = [c.strip().strip("'").strip('"') for c in columns.split(",")]
    global progressor
    if isinstance(progressor, DataPreprocessor):
        progressor.get_dummy_data(column_list)
        progressor.save_data(r"F:\多模态智能语音助手\my_assistant\temp\temp.csv")

        return r"独热编码完毕,结果保存至F:\多模态智能语音助手\my_assistant\temp\temp.csv"


@tool("fill_null_with_mean", args_schema=columnInput)
def fill_null_with_mean(columns):
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
        if isinstance(progressor, DataPreprocessor):
            progressor.fill_null_with_mean(columns=column_list)
            progressor.save_data(r"F:\多模态智能语音助手\my_assistant\temp\temp.csv")
            return r"均值填充完毕,结果保存至F:\多模态智能语音助手\my_assistant\temp\temp.csv"
    except ValueError as e:
        return f"错误：{e}"


@tool("drop_null_rows", args_schema=columnInput)
def drop_null_rows(columns):
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
        if isinstance(progressor, DataPreprocessor):
            progressor.drop_null_rows(columns=column_list)
            progressor.save_data(r"F:\多模态智能语音助手\my_assistant\temp\temp.csv")
            return r"已删除包含空值的行,结果保存至F:\多模态智能语音助手\my_assistant\temp\temp.csv"
    except ValueError as e:
        return f"错误：{e}"


@tool("get_null_info", args_schema=columnInput)
def get_null_info(columns):
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
        if isinstance(progressor, DataPreprocessor):
            result = progressor.get_null_info(columns=column_list)
            progressor.save_data(r"F:\多模态智能语音助手\my_assistant\temp\temp.csv")
            return f"已获取空值信息\n {result}"
    except ValueError as e:
        return f"错误：{e}"


@tool("get_null_info", args_schema=columnInput)
def get_null_info(columns):
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
        if isinstance(progressor, DataPreprocessor):
            progressor.get_null_info(columns=column_list)
            progressor.save_data(r"F:\多模态智能语音助手\my_assistant\temp\temp.csv")
            return r"空值删除完毕,结果保存至F:\多模态智能语音助手\my_assistant\temp\temp.csv"
    except ValueError as e:
        return f"错误：{e}"


@tool("Label_Encoding", args_schema=columnInput)
def Label_Encoding(columns):
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
        if isinstance(progressor, DataPreprocessor):
            progressor.Label_Encoding(columns=column_list)
            progressor.save_data(r"F:\多模态智能语音助手\my_assistant\temp\temp.csv")
            return r"标签编码完毕,结果保存至F:\多模态智能语音助手\my_assistant\temp\temp.csv"
    except ValueError as e:
        return f"错误：{e}"


@tool("Standard_Scaling", args_schema=columnInput)
def Standard_Scaling(columns):
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
        if isinstance(progressor, DataPreprocessor):
            progressor.Standard_Scaling(columns=column_list)
            progressor.save_data(r"F:\多模态智能语音助手\my_assistant\temp\temp.csv")
            return r"标准化完毕,结果保存至F:\多模态智能语音助手\my_assistant\temp\temp.csv"
    except ValueError as e:
        return f"错误：{e}"


@tool("drop_columns", args_schema=columnInput)
def drop_columns(columns):
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
        if isinstance(progressor, DataPreprocessor):
            progressor.drop_columns(columns=column_list)
            progressor.save_data(r"F:\多模态智能语音助手\my_assistant\temp\temp.csv")
            return "删除列完毕"
    except ValueError as e:
        return f"错误：{e}"


@tool("machine_learning_train",return_direct=True)
def machine_learning_train(query: str):
    """
    1. 当需要训练模型时，且参数明确时，调用该工具，固定参数的单次模型训练。输入只有一个{query}，需要结合用户需求完成query输入，query的内容需包含：model_name（模型名称）、X_columns（特征列名，用逗号分隔）、y_column（目标列名）、mode（'分类'或'回归'）、model_param(模型对应需要的参数)等信息。不可与machine_learning_train_BayesSearch同时调用。"
    2. 该工具和machine_learning_train_BayesSearch只能调用一个，不可一起调用
    :param query: 用户关于模型训练的查询（例如："用随机森林做回归，特征列是Age,Score，目标列是Salary，参数n_estimators=100"）
    """
    global model  # 声明全局变量
    data = pd.read_csv("./temp/temp.csv")
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

    return f"{model_name}模型训练完毕，已保存至./machine_learning_model。模型评估结果为\n{dict}"

@tool(
    "machine_learning_train_BayesSearch",
    return_direct=True)
def machine_learning_train_BayesSearch(query: str,):
    """
    贝叶斯搜索调参工具，用于参数不明确时的模型训练。固定参数的单次模型训练。输入只有一个{query}，需要结合用户需求完成query输入，query的内容需包含：model_name（模型名称）、X_columns（特征列名，用逗号分隔）、y_column（目标列名）、mode（'分类'或'回归'）等信息。不可与machine_learning_train_BayesSearch同时调用。"
    输入示例："用随机森林做回归，特征列是Age,Score，目标列是Salary"
    约束：与machine_learning_train互斥，只能调用其中一个。
    """
    global model  # 声明全局变量
    data = pd.read_csv("./temp/temp.csv")
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
    model.download_model("./machine_learning_model")
    model_name, dict = model.evaluate()

    return f"{model_name}模型训练完毕，已保存至./machine_learning_model。模型评估结果为\n{dict}，最优参数为{best_params}"

def fig_to_base64(fig):
    buf = io.BytesIO()
    # 直接用 fig.savefig()，支持 bbox_inches 参数
    fig.savefig(buf, format='png', bbox_inches='tight')  # 这里可以用 bbox_inches
    buf.seek(0)
    img_bytes = buf.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')
def visualization():
    """
    暂未想到快速干净的可视化途径，故在此处直接复用之前global的model类
    :return:
    """
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
            dataframe_analyse,
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
            stock_data_model
            ]
