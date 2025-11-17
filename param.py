
# 加载你的参数定义
import json
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
def param_types():
    """
    定义模型参数对应的数据类型（完整版本，去重处理）
    """
    return {
        # 通用参数
        # 'random_state': 'int',
        # 'n_jobs': 'int',
        "随机森林":{
        # 随机森林 RandomForestRegressor / RandomForestClassifier
        'n_estimators': 'int',
        'criterion': 'select',
        'max_depth': 'int',
        'min_samples_split': 'int',
        'min_samples_leaf': 'int',
        'min_weight_fraction_leaf': 'float',
        'max_features': 'select',
        'max_leaf_nodes': 'int',
        'min_impurity_decrease': 'float',
        'bootstrap': 'bool',
        'oob_score': 'bool',
        'ccp_alpha': 'float',
        'max_samples': 'int',
        },
        # 梯度提升树 GradientBoostingRegressor / GradientBoostingClassifier
        "梯度提升树":{
        'learning_rate': 'float',
        'subsample': 'float',
        'validation_fraction': 'float',
        'n_iter_no_change': 'int',
        'tol': 'float',
        'init': 'select',  # estimator 或 None
        'warm_start': 'bool',
        },
        # 线性回归 LinearRegression
        "线性回归":{
        'fit_intercept': 'bool',
        'normalize': 'bool',  # 已弃用
        'copy_X': 'bool',
        'positive': 'bool',
        },
        # Logistic回归 LogisticRegression
        "逻辑回归":{
        'penalty': 'select',
        'dual': 'bool',
        'C': 'float',
        'intercept_scaling': 'float',
        'class_weight': 'select',
        'solver': 'select',
        'multi_class': 'select',
        'l1_ratio': 'float',  # elasticnet
        },
        # 支持向量机 SVR / SVC
        "支持向量机":{
        'kernel': 'select',
        'degree': 'int',
        'gamma': 'select',
        'coef0': 'float',
        'shrinking': 'bool',
        'cache_size': 'int',
        'verbose': 'bool',
        'max_iter': 'int',
        'epsilon': 'float',  # SVR
        'probability': 'bool',  # SVC
        },
        # 决策树 DecisionTreeRegressor / DecisionTreeClassifier
        'splitter': 'select',

        "决策树":{
        # KNN KNeighborsRegressor / KNeighborsClassifier
        'n_neighbors': 'int',
        'weights': 'select',
        'algorithm': 'select',
        'leaf_size': 'int',
        'p': 'int',
        'metric': 'select',
        'metric_params': 'dict',
    }

    }
def param_options_map():
    """
    定义枚举类型参数的可选值（完整版本，去重处理）
    """
    return {
        #
        "随机森林":{
        'criterion': ['mse', 'friedman_mse', 'mae', 'poisson', 'gini', 'entropy'],
        'max_features': ['auto', 'sqrt', 'log2', None],
        },
        "决策树":{
        # 决策树
        'criterion': ['mse', 'friedman_mse', 'mae', 'poisson', 'gini', 'entropy'],
        'max_features': ['auto', 'sqrt', 'log2', None],
        'splitter': ['best', 'random'],
        },
        # Logistic回归
        "逻辑回归":{
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'solver': ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'],
        'multi_class': ['auto', 'ovr', 'multinomial'],
        'class_weight': ['balanced', None],
        },
        # 支持向量机
        "支持向量机":{
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
        'gamma': ['scale', 'auto'],
        },
        # KNN
        "KNN":{
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'metric': ['minkowski', 'euclidean', 'manhattan', 'chebyshev', 'precomputed'],
        },
        # GradientBoosting init
        "梯度提升树":{
        'init': [None],  # 可以是 estimator, 这里简单处理
    }
    }



param_types = param_types()
param_options_map = param_options_map()


def cast_params(model_name, params_dict, param_types):
    """
    根据 param_types 的类型定义对参数进行类型转换
    (此函数功能完善，无需修改)
    """
    casted = {}
    type_map = {
        'int': int,
        'float': float,
        'bool': lambda x: str(x).lower() in ["true", "1"],  # 稍稍增强 bool 的判断
        'select': str,
        'dict': dict
    }
    for k, v in params_dict.items():
        if k in param_types.get(model_name, {}):
            typ_str = param_types[model_name][k]
            typ_func = type_map.get(typ_str, str)
            try:
                casted[k] = typ_func(v)
            except Exception as e:
                print(f"Warning: 参数 {k}={v} 类型转换失败 ({typ_str})，已忽略，使用默认值")
        else:
            print(f"Warning: 参数 {k} 对模型 {model_name} 不存在类型定义，忽略。")
    return casted


def validate_enum_params(model_name, params_dict, param_options_map):
    """
    过滤掉不合法的枚举参数
    (此函数功能完善，无需修改)
    """
    valid_params = {}
    for k, v in params_dict.items():
        if k in param_options_map.get(model_name, {}):
            if v in param_options_map[model_name][k]:
                valid_params[k] = v
            else:
                print(f"Warning: 参数 {k}={v} 对模型 {model_name} 不合法，已忽略，使用默认值")
        else:
            valid_params[k] = v
    return valid_params


def get_ML_param(query: str):
    """
    优化后的机器学习训练任务解析器 (已修复 KeyError)
    """

    # 1. 初始化 (使用单个 LLM)
    llm = ChatOpenAI(
        base_url="https://api.siliconflow.cn/v1",
        openai_api_key="sk-xtgeyahfmjxvrxjygvnwfywbezskstroipqofybruqldkgor",
        model="Qwen/Qwen2.5-7B-Instruct"
    )

    # 2. 将参数定义转换为字符串，注入到 Prompt
    valid_models_str = json.dumps(list(param_types.keys()))
    param_types_str = json.dumps(param_types)
    param_options_str = json.dumps(param_options_map)

    # 3. 优化后的 Prompt 模板
    prompt_template = PromptTemplate(
        template="""你是一名机器学习任务解析器。请将用户的任务描述严格转换为 JSON 格式。
# 任务
从用户的 `{query}` 中提取以下5个字段：

1.  `model_name`: 必须从以下列表中选择：{valid_models}
2.  `X_columns`: 特征列 (必须是字符串列表)
3.  `y_columns`: 目标列 (必须是单个字符串)
4.  `mode`: 任务模式 (必须是 "分类" 或 "回归")
5.  `model_param`: 参数字典 (必须是 `{{key: value}}` 字典格式)

# 参数约束 (重要)
你必须参考以下定义来生成 `model_param`。只包含用户提到、且在定义中存在的参数。
-   **所有模型的类型定义**: {param_types}
-   **枚举参数的合法值**: {param_options}

# 输出要求
-   **必须**只输出一个合法的 JSON 对象，不要包含任何 markdown 标记或解释性文字。
-   如果用户未提供 `model_param`，请输出 `{{}}`。  <-- ⭐️⭐️⭐️ 已修复 ⭐️⭐️⭐️

# 用户任务
{query}
""",
        input_variables=["query", "valid_models", "param_types", "param_options"]
    )

    # 4. 构建链条 (单次调用)
    chain = prompt_template | llm | StrOutputParser()

    # 5. 执行链条
    llm_output = chain.invoke({
        "query": query,
        "valid_models": valid_models_str,
        "param_types": param_types_str,
        "param_options": param_options_str
    })

    # 6. 安全解析 JSON
    try:
        # 增加鲁棒性：从 LLM 可能的输出中提取 JSON 块
        match = re.search(r"\{.*\}", llm_output, re.DOTALL)
        if not match:
            raise json.JSONDecodeError("No JSON object found", llm_output, 0)

        data = json.loads(match.group(0))

        # 按顺序提取
        model_name = data.get("model_name")
        X_cols = data.get("X_columns", [])
        y_col = data.get("y_columns")
        mode = data.get("mode")
        model_param = data.get("model_param", {})

    except Exception as e:
        print(f"Error: LLM 输出解析失败: {e}\nRaw output: {llm_output}")
        # 兜底
        model_name, X_cols, y_col, mode, model_param = None, [], None, None, {}

    # 7. 类型转换与验证 (复用原函数)
    if not isinstance(model_param, dict):
        model_param = {}

    model_param = cast_params(model_name, model_param, param_types)
    model_param = validate_enum_params(model_name, model_param, param_options_map)

    # 8. 返回最终结果
    return [
        model_name,
        X_cols,
        y_col,
        mode,
        model_param
    ]


import json


def convert_to_json_string(result_list: list) -> str:
    """
    将 get_ML_param 的列表输出转换为目标 JSON 字符串格式。

    输入 (result_list):
    [
        model_name,  # (str) e.g., "线性回归"
        X_cols,      # (list) e.g., ["Feat1", "Feat2"]
        y_col,       # (str) e.g., "Price"
        mode,        # (str) e.g., "回归"
        model_param   # (dict) e.g., {'fit_intercept': True}
    ]

    输出 (str):
    一个 JSON 字符串，格式如下:
    {
      "model_name": "线性回归",
      "X_columns": "Feat1,Feat2",
      "y_column": "Price",
      "mode": "回归",
      "model_param": "{\"fit_intercept\": true}"
    }
    """

    # 1. 拆包
    model_name, X_cols, y_col, mode, model_param = result_list

    # 2. 转换 X_columns：从 list 转换为逗号分隔的字符串
    # 确保 X_cols 是一个列表，以防解析出错
    if isinstance(X_cols, list):
        x_columns_str = ",".join(X_cols)
    else:
        x_columns_str = ""  # 或者 str(X_cols)

    # 3. 转换 model_param：从 dict 转换为 JSON 字符串
    # ensure_ascii=False 确保中文等字符不会被转义
    model_param_str = json.dumps(model_param, ensure_ascii=False)

    # 4. 构建目标字典
    # 注意：我们将 y_col 映射到您要求的 y_column
    output_dict = {
        "model_name": str(model_name) if model_name is not None else "",
        "X_columns": x_columns_str,
        "y_column": str(y_col) if y_col is not None else "",
        "mode": str(mode) if mode is not None else "",
        "model_param": model_param_str
    }

    # 5. 将整个字典转换为 JSON 字符串
    # indent=2 是为了打印时更美观，如果用于传输，可以去掉
    final_json_string = json.dumps(output_dict, ensure_ascii=False, indent=2)

    return final_json_string

if __name__ == '__main__':
    # --- 测试 (与之前相同) ---

    test_query2 = "我需要训练一个线性回归模型，用于预测房价，特征列是SquareFootage,YearBuilt,Rooms，目标列是Price，参数设置fit_intercept为True，normalize为False"
    test_query5 = "用决策树模型做分类，特征列是A,B,C，目标列是Result，不需要特别指定参数，用默认值即可"
    test_query3 = "用支持向量机做二分类，特征列是Feature1,Feature2,Feature3，目标列是Label，参数C设置为1.0，kernel用'rbf'，gamma取'scale'"


    # --- 运行测试 1 ---
    print("--- 测试 Query 2 (线性回归) ---")

    # 步骤 1: 获取列表结果 (调用一次)
    list_result = get_ML_param(test_query2)

    print("\n[步骤 1] 原始 List 输出:")
    print(list_result)

    # 步骤 2: 转换为目标 JSON 字符串 (调用一次)
    final_json = convert_to_json_string(list_result)

    print("\n[步骤 2] 目标 JSON 字符串输出:")
    print(final_json)


    # --- 运行测试 2 ---
    print("\n" + "---" * 15 + "\n")
    print("--- 测试 Query 3 (支持向量机) ---")

    # 步骤 1: 获取列表结果
    list_result_2 = get_ML_param(test_query3)

    print("\n[步骤 1] 原始 List 输出:")
    print(list_result_2)

    # 步骤 2: 转换为目标 JSON 字符串
    final_json_2 = convert_to_json_string(list_result_2)

    print("\n[步骤 2] 目标 JSON 字符串输出:")
    print(final_json_2)
    print(type(final_json_2))