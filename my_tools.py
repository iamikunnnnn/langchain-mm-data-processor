import base64
import io
import json
import re
from datetime import datetime
from typing import Optional

import pandas as pd
import requests
from baidusearch.baidusearch import search
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pydantic import BaseModel, Field
from sympy.parsing.latex import parse_latex

from data_preprocessing import DataPreprocessor
from utils import load_config
from machine_learning_model import model_choice
from param import get_ML_param, convert_to_json_string

config = load_config()

# 声明模块级变量，用于存储模型实例和预处理实例
progressor = None
trained_model: Optional[model_choice] = None  # 类型注解：可选的Model实例

class columnInput(BaseModel):
    columns: str = Field(..., description="需要填充空值的列名")


# 1. 定义多参数的 schema（Pydantic 模型）
class MachineLearningTrainInput(BaseModel):
    model_name: str = Field(
        ...,  # 表示必填参数
        description="model_name，必须是以下之一：['KNN', '线性回归', '决策树', '随机森林', '梯度提升树', '支持向量机']"
    )
    X_columns: str = Field(
        ...,
        description="X_columns，多个列名用逗号分隔（如 'age,income'）"
    )
    y_column: str = Field(
        ...,
        description="y_column，单个列名（如 'label'）"
    )
    mode: str = Field(
        ...,
        description="mode，必须是 '回归' 或 '分类'"
    )
    # param_dict: str = Field(
    #     description="param_dict，例如：'{\"n_estimators\": 100, \"max_depth\": 5}' 或 '{param: num, param2: true}'"
    # )


# 加载FAISS知识库
embedding = HuggingFaceEmbeddings(model_name=config["embedding"]["model"])

vector_db = FAISS.load_local("faiss_index", embedding,
                             allow_dangerous_deserialization=True)  # 加载已存储的索引


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


@tool("get_weather")
def get_weather(city):
    """
    查询指定城市的实时天气信息
    :param city: 城市名称
    :return: 返回一个字符串，指定城市的当前温度和天气
    """
    try:
        url = f"http://wttr.in/{city}?format=j1"
        response = requests.get(url)
        data = response.json()
        if "current_condition" in data:
            temp = data["current_condition"][0]["temp_C"]
            weather = data["current_condition"][0]["weatherDesc"][0]["value"]
            print(f"{city}当前温度：{temp}°C，天气：{weather}")
            return f"{city}当前温度：{temp}°C，天气：{weather}"
        else:
            return f"无法获取{city}的天气信息"
    except Exception as e:
        return f"查询天气时出错: {str(e)}"


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


@tool("navigation")
def navigation(origin, destination):
    """
    导航工具，输入起点和目标地点
    :param origin: 起始地点
    :param destination: 目标地点
    :return: 导航总距离、预计时间、下一步操作、下一步操作的行进距离
    """
    print(origin)
    print(destination)
    origin = get_coordinate(origin)
    destination = get_coordinate(destination)
    api_key = config["api"]["gaode_api_key"]
    params = {
        "key": api_key,
        "origin": origin,
        "destination": destination
    }
    url = f"https://restapi.amap.com/v3/direction/driving"
    response = requests.get(url, params=params)
    data = response.json()
    if data["status"] == "1" and data["count"] != "0":
        # print(data)
        # 获取速度最快的路线（已明确唯一，直接取第一条）
        fastest_route = data["route"]["paths"][0]

        # 解析总距离、总时间
        total_distance = int(fastest_route["distance"])
        total_time_min = int(fastest_route["duration"]) // 60
        distance_str = f"{total_distance}米（约{total_distance / 1000:.1f}公里）"

        # 解析下一步导航指令（第一步）
        first_step = fastest_route["steps"][0]
        next_instruction = first_step["instruction"]
        next_distance = first_step["distance"]  # 第一步距离
        return f"distance_str:{distance_str}, total_time_min:{total_time_min}, next_instruction:{next_instruction},next_distance: {next_distance}"
    else:
        print("导航失败")
        return


@tool("calculate", return_direct=True)
def calculate(latex_expr: str) -> str:
    """
    需要计算数学式子时使用该工具，输入{input}，用于计算任意 LaTeX 数学表达式并返回数值字符串。
    支持 /sin、/cos、/log、矩阵、积分、求和等所有 SymPy 能识别的 LaTeX 语法。
    :param latex_expr: LaTeX 数学表达式
    :return: 计算结果

    """
    try:
        # LaTeX -> SymPy
        expr = parse_latex(latex_expr)
        # 求值
        value = expr.evalf()
        # 去掉浮点末尾的 .0
        if value == int(value):
            return str(int(value))
        return str(float(value))
    except Exception as e:
        return latex_expr


@tool("dataframe_analyse", return_direct=True)
def dataframe_analyse(tmp=None):
    """
    对表格数据进行分析,当用户存在对数据的基础分析意图时调用,同样只需调用一次，无需输入任何参数
    :return: {"形状","列名","空值数量","重复行数量"}
    """
    if isinstance(progressor, DataPreprocessor):
        return progressor.get_basic_info()


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


@tool("machine_learning_train")
def machine_learning_train(query: str):
    """
    当需要训练模型时调用该工具，输入包含训练模型需要的model_name、X_columns、y_column、mode、model_param等信息的查询文本
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
    model.train()
    model.download_model("./machine_learning_model")
    model_name, dict = model.evaluate()

    return f"{model_name}模型训练完毕，已保存至./machine_learning_model。模型评估结果为\n{dict}"


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
    return [get_weather,
            get_current_time,
            web_search, navigation,
            calculate,
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
            machine_learning_train
            ]
