from typing import List, Union

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

# 全局加载BGE模型（保持不变）
embedding_model = SentenceTransformer(r"F:\MuXueAI\MuXueAI\bge_small_zh").eval()
for p in embedding_model.parameters():
    p.requires_grad = False

class StockDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, seq_len=10, target_col='close', scaler=None, mode='train'):
        """
        Args:
            dataframe: akshare 获取并清洗后的 DataFrame
            seq_len: 时间步长 (用过去多少天预测)
            target_col: 要预测的目标列 (默认收盘价)
            scaler: 归一化器 (训练集需要拟合，验证/测试集复用训练集的scaler)
            mode: 'train' 会拟合scaler, 'test' 仅使用scaler
        """
        super().__init__()
        self.seq_len = seq_len
        self.target_col = target_col
        self.mode = mode

        # 1. 数据准备：只保留数值列 (股票数据通常全是数值)
        self.df = dataframe.reset_index(drop=True)

        # 提取所有用于输入的特征列 (排除日期等非数值列)
        self.feature_cols = [c for c in self.df.columns if c in
                             ['open', 'close', 'high', 'low', 'volume', 'amount',
                              'ma_5', 'ma_10', 'ma_20', 'rsi_6', 'macd_dif', 'turnover']]

        # 2. 归一化 (至关重要！)
        # 我们需要将数据缩放到 [0, 1] 之间
        if scaler is None:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        else:
            self.scaler = scaler

        # 获取原始数据矩阵
        data_matrix = self.df[self.feature_cols].values

        if self.mode == 'train':
            # 训练模式：计算均值方差并转换
            self.data_normalized = self.scaler.fit_transform(data_matrix)
        else:
            # 测试模式：使用训练集的参数转换 (防止未来数据泄露)
            self.data_normalized = self.scaler.transform(data_matrix)

        # 找到目标列在特征矩阵中的索引，用于提取 Label
        self.target_idx = self.feature_cols.index(target_col)

    def __len__(self):
        return len(self.df) - self.seq_len

    def __getitem__(self, idx):
        """
        返回:
            x: [seq_len, feature_dim] (过去N天的特征)
            y: [1] (第N+1天的预测目标)
        """
        # 获取输入序列 x (索引 idx 到 idx+seq_len)
        # 过去 seq_len 天的数据
        x = self.data_normalized[idx: idx + self.seq_len]

        # 获取标签 y (索引 idx + seq_len)
        # 紧接着的"下一天"的真实值
        # 注意：这里我们预测的是"归一化后"的值，模型输出后需要反归一化还原
        y = self.data_normalized[idx + self.seq_len, self.target_idx]

        # 转为 Tensor, float32 用于回归
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def get_scaler(self):
        return self.scaler

class DataPreprocessor:
    """
    数据预处理
    """
    def __init__(self, data: pd.DataFrame = None):
        """
        初始化数据预处理器
        :param: data
        """
        # 创建时直接初始化data
        self.data = data.copy() if data is not None else None

    def get_data_null(self):
        return self.data.isna().sum()

    # 空值处理 - 3个常用方法

    def drop_null_columns(self, columns):
        self.data = self.data.drop(columns=columns)
        return self.data

    def drop_null_rows(self, columns: Union[str, List[str], None] = None):
        """
        删除包含空值的行
        :param:columns: 要检查的列，None表示检查所有列
        :return:self: 返回自身，支持链式调用
        """
        if self.data is None:
            raise ValueError("没有加载数据")

        if columns:
            # 如果columns的类型确实是str
            if isinstance(columns, str):
                columns = [columns]

            # 检查列是否存在
            missing_cols = [col for col in columns if col not in self.data.columns]
            if missing_cols:
                raise ValueError(f"列不存在: {missing_cols}")

            self.data = self.data.dropna(subset=columns)
        else:
            self.data = self.data.dropna()

        return self

    def drop_columns(self, columns):
        """
        删除输入的列名对应的列

        :param columns: 待删除的列名（单个字符串或列表）
        :return: 成功返回提示信息，失败返回具体错误原因
        """
        try:
            # 检查数据是否加载
            if self.data is None:
                return "删除失败：未加载数据，请先加载数据"  # 更具体的提示

            # 处理空输入（如 columns 为空列表或空字符串）
            if not columns:
                return "删除失败：未指定需要删除的列"

            # 统一将输入转换为列表（支持单个列名或列表）
            if isinstance(columns, str):
                columns = [columns.strip()]  # 去除可能的空格
            else:
                # 确保列表中的列名去除空格
                columns = [col.strip() for col in columns]

            # 检查列是否存在
            missing_cols = [col for col in columns if col not in self.data.columns]
            if missing_cols:
                return f"删除失败：以下列不存在：{missing_cols}"

            # 执行删除
            self.data = self.data.drop(columns=columns)  # 显式指定 columns 参数，避免歧义
            return f"成功删除列：{columns}"  # 明确返回成功信息

        except Exception as e:
            # 捕获其他意外错误（如列名格式错误、数据被锁定等）
            return f"删除失败：发生意外错误 - {str(e)}"


    def fill_null_with_mean(self, columns: Union[str, List[str], None] = None) -> 'DataPreprocessor':
        """
        用均值填充空值（仅适用于数值列）
        :param:columns: 要处理的列，None表示处理所有数值列
        :return:self: 返回自身，支持链式调用
        """
        if self.data is None:
            raise ValueError("没有加载数据")

        if columns is None:
            # 只处理数值列
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                self.data[col].fillna(self.data[col].mean(), inplace=True)
        else:
            if isinstance(columns, str):
                columns = [columns]

            for col in columns:
                if col not in self.data.columns:
                    raise ValueError(f"列 '{col}' 不存在")
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    self.data[col].fillna(self.data[col].mean(), inplace=True)
                else:
                    print(f"警告: 列 '{col}' 不是数值类型，跳过处理")

        return self

    # 重复值处理 - 2个常用方法
    def remove_duplicates(self, columns: Union[str, List[str], None] = None) -> 'DataPreprocessor':
        """
        删除重复行
        Args:
            columns: 检查重复的列，None表示检查所有列
        Returns:
            self: 返回自身，支持链式调用
        """
        if self.data is None:
            raise ValueError("没有加载数据")
        if columns:
            if isinstance(columns, str):
                columns = [columns]
            self.data = self.data.drop_duplicates(subset=columns)
        else:
            self.data = self.data.drop_duplicates()
        return self

    # 数据信息查看 - 2个常用方法

    import pandas as pd

    def get_basic_info(self):
        """
        获取数据基本信息，根据数据类型（定类/定量）选择性计算统计量
        :return: dict: 包含数据基本信息
        """
        if self.data is None:
            return {"error": "没有加载数据"}
        return {
            "形状": self.data.shape,
            "列名": self.data.columns,
            "空值数量": dict(self.data.isna().sum()),
            "重复行数量": self.data.duplicated().sum(),
            "前二行数据": self.data.head(2).to_dict(orient="records"),
        }


    def get_null_info(self, columns: Union[str, List[str], None] = None) -> dict:
        """
        获取空值详细信息
        :param:columns: 指定列，None表示所有列
        :return:dict: 空值信息
        """
        if self.data is None:
            return {"error": "没有加载数据"}

        if columns is not None:
            if isinstance(columns, str):
                columns = [columns]

            # 检查列是否存在
            missing_cols = [col for col in columns if col not in self.data.columns]
            if missing_cols:
                raise ValueError(f"列不存在: {missing_cols}")

            data_subset = self.data[columns]
        else:
            data_subset = self.data

        null_counts = data_subset.isnull().sum()
        null_percentage = (null_counts / len(data_subset) * 100).round(2)

        return {
            "空值数量": dict(null_counts),
            "空值百分比": dict(null_percentage),
            "总空值数": null_counts.sum()
        }

    def get_data(self) -> pd.DataFrame:
        """
        获取当前数据
        :return:pd.DataFrame: 当前处理后的数据
        """
        if self.data is None:
            raise ValueError("没有加载数据")
        return self.data.copy()

    def save_data(self, filepath: str) -> None:
        """
        保存数据到CSV文件
        :param:filepath: 文件路径
        """
        if self.data is None:
            raise ValueError("没有数据可保存")

        self.data.to_csv(filepath, index=False)

    def get_dummy_data(self, columns) -> pd.DataFrame:
        """
        :param columns: 需要独热编码的列
        :return: 处理后的DataFrame（去除原分类列，添加独热编码列）
        """
        try:
            # 步骤1：验证并筛选存在的列
            if not columns:  # 避免空列表
                return self.data

            # 获取存在的列和缺失的列
            existing_columns = [col for col in columns if col in self.data.columns]
            missing_columns = list(set(columns) - set(existing_columns))

            # 提示缺失列（不中断程序）
            if missing_columns:
                print(f"警告：以下列不存在，已跳过：{missing_columns}")
            if not existing_columns:  # 所有列都不存在时直接返回
                return self.data

            # 步骤2：处理存在的列（替换原判断逻辑，更严谨）
            # 提取需要处理的子DataFrame
            data_subset = self.data[existing_columns]

            # 情况1：仅单列需要编码（返回Series）
            if len(existing_columns) == 1:
                # 直接用pd.get_dummies处理
                dummies = pd.get_dummies(data_subset, prefix=existing_columns[0])
                # 合并回原数据（删除原列，添加编码列）
                self.data = pd.concat([self.data.drop(columns=existing_columns), dummies], axis=1)

            # 情况2：多列需要编码（返回DataFrame）
            else:
                # 筛选出object类型的分类列（避免对数值列编码）
                categorical_columns = data_subset.select_dtypes(include=['object']).columns
                if not len(categorical_columns):
                    return self.data  # 无分类列可编码

                # 过滤基数过高的列（避免生成过多编码列）
                columns_to_encode = [
                    col for col in categorical_columns
                    if len(self.data[col].unique()) <= 300
                ]
                if not columns_to_encode:
                    return self.data  # 所有分类列基数过高，跳过

                # 用DictVectorizer生成独热编码（稀疏矩阵节省内存）
                transfer = DictVectorizer(sparse=True)
                dummies = transfer.fit_transform(
                    self.data[columns_to_encode].to_dict(orient='records')
                )

                # 转换为稀疏DataFrame并合并
                dummies_df = pd.DataFrame.sparse.from_spmatrix(
                    dummies,
                    columns=transfer.get_feature_names_out(),
                    index=self.data.index
                ).astype(int)  # 转为int类型

                # 合并回原数据（删除原分类列，添加编码列）
                self.data = pd.concat(
                    [self.data.drop(columns=columns_to_encode), dummies_df],
                    axis=1
                )

            return self.data
        except Exception as e:
            print(e)
    def Label_Encoding(self, columns):
        """
        将指定列进行标签编码（Label Encoding），将类别转换为整数
        :param columns: 需要编码的列
        :return: 编码后的 DataFrame
        """
        # 如果是series不需要那么操作
        if isinstance(self.data[columns], pd.Series):
            le = LabelEncoder()
            self.data[columns] = le.fit_transform(self.data[columns].astype(str))  # 转成 str 以防有 NaN
        else:

            # 先筛选出 object 类型的列
            categorical_columns = self.data[columns].select_dtypes(include=['object']).columns

            # 可选：限制类别数量，避免高基数编码
            columns_to_encode = [col for col in categorical_columns if len(self.data[col].unique()) <= 10]

            if not columns_to_encode:
                return self.data

            # 对每列进行 Label Encoding
            for col in columns_to_encode:
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col].astype(str))  # 转成 str 以防有 NaN

        return self.data

    def Standard_Scaling(self, columns):
        """
        对指定列进行标准化（StandardScaler），将数据缩放到均值为0、方差为1的分布
        :param columns: 需要标准化的列，可以是单列名或列名列表
        :return: 标准化后的 DataFrame
        """
        scaler = StandardScaler()

        # 如果是单列（Series）
        if isinstance(self.data[columns], pd.Series):
            self.data[columns] = scaler.fit_transform(self.data[columns].values.reshape(-1, 1))
        else:
            # 先筛选出数值型的列
            numeric_columns = self.data[columns].select_dtypes(include=['int64', 'float64']).columns

            if not len(numeric_columns):
                return self.data

            # 批量标准化
            self.data[numeric_columns] = scaler.fit_transform(self.data[numeric_columns])

        return self.data

    def __repr__(self) -> str:
        """字符串表示"""
        if self.data is None:
            return "DataPreprocessor(无数据)"
        else:
            return f"DataPreprocessor(数据形状: {self.data.shape})"
    #------------------------------------------------新增功能--------------
    def detect_outliers_iqr(self, columns: Union[str, List[str], None] = None, threshold: float = 1.5) -> dict:
        """
        使用IQR方法检测异常值
        :param columns: 要检查的列，None表示检查所有数值列
        :param threshold: IQR倍数阈值，默认1.5
        :return: dict: 异常值信息
        """
        if self.data is None:
            raise ValueError("没有加载数据")

        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        elif isinstance(columns, str):
            columns = [columns]

        outlier_info = {}
        for col in columns:
            if col not in self.data.columns:
                continue
            if not pd.api.types.is_numeric_dtype(self.data[col]):
                continue

            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)]
            outlier_info[col] = {
                "异常值数量": len(outliers),
                "异常值占比": f"{len(outliers) / len(self.data) * 100:.2f}%",
                "下界": lower_bound,
                "上界": upper_bound
            }

        return outlier_info

    def remove_outliers_iqr(self, columns: Union[str, List[str], None] = None,
                            threshold: float = 1.5) -> 'DataPreprocessor':
        """
        删除IQR方法检测到的异常值
        :param columns: 要处理的列，None表示处理所有数值列
        :param threshold: IQR倍数阈值
        :return: self
        """
        if self.data is None:
            raise ValueError("没有加载数据")

        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        elif isinstance(columns, str):
            columns = [columns]

        for col in columns:
            if col not in self.data.columns or not pd.api.types.is_numeric_dtype(self.data[col]):
                continue

            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            self.data = self.data[(self.data[col] >= lower_bound) & (self.data[col] <= upper_bound)]

        self.data = self.data.reset_index(drop=True)
        return self

    def cap_outliers(self, columns: Union[str, List[str], None] = None, threshold: float = 1.5) -> 'DataPreprocessor':
        """
        使用盖帽法处理异常值（截断到边界值）
        :param columns: 要处理的列
        :param threshold: IQR倍数阈值
        :return: self
        """
        if self.data is None:
            raise ValueError("没有加载数据")

        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        elif isinstance(columns, str):
            columns = [columns]

        for col in columns:
            if col not in self.data.columns or not pd.api.types.is_numeric_dtype(self.data[col]):
                continue

            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            self.data[col] = self.data[col].clip(lower=lower_bound, upper=upper_bound)

        return self

    def normalize_minmax(self, columns: Union[str, List[str], None] = None,
                         feature_range: tuple = (0, 1)) -> 'DataPreprocessor':
        """
        对指定列进行Min-Max归一化
        :param columns: 需要归一化的列
        :param feature_range: 归一化范围，默认(0,1)
        :return: self
        """
        if self.data is None:
            raise ValueError("没有加载数据")

        scaler = MinMaxScaler(feature_range=feature_range)

        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        elif isinstance(columns, str):
            columns = [columns]

        numeric_columns = [col for col in columns if col in self.data.columns
                           and pd.api.types.is_numeric_dtype(self.data[col])]

        if numeric_columns:
            self.data[numeric_columns] = scaler.fit_transform(self.data[numeric_columns])

        return self

    def create_lag_features(self, columns: Union[str, List[str]],
                            lags: Union[int, List[int]] = 1) -> 'DataPreprocessor':
        """
        创建滞后特征（用于时间序列）
        :param columns: 要创建滞后特征的列
        :param lags: 滞后期数，可以是单个整数或列表
        :return: self
        """
        if self.data is None:
            raise ValueError("没有加载数据")

        if isinstance(columns, str):
            columns = [columns]
        if isinstance(lags, int):
            lags = [lags]

        for col in columns:
            if col not in self.data.columns:
                continue
            for lag in lags:
                self.data[f'{col}_lag_{lag}'] = self.data[col].shift(lag)

        return self

    def create_rolling_features(self, columns: Union[str, List[str]], window: int,
                                agg_funcs: List[str] = ['mean']) -> 'DataPreprocessor':
        """
        创建滚动窗口特征
        :param columns: 要处理的列
        :param window: 窗口大小
        :param agg_funcs: 聚合函数列表，如['mean', 'std', 'min', 'max']
        :return: self
        """
        if self.data is None:
            raise ValueError("没有加载数据")

        if isinstance(columns, str):
            columns = [columns]

        for col in columns:
            if col not in self.data.columns or not pd.api.types.is_numeric_dtype(self.data[col]):
                continue

            for func in agg_funcs:
                if func == 'mean':
                    self.data[f'{col}_rolling_{window}_{func}'] = self.data[col].rolling(window=window).mean()
                elif func == 'std':
                    self.data[f'{col}_rolling_{window}_{func}'] = self.data[col].rolling(window=window).std()
                elif func == 'min':
                    self.data[f'{col}_rolling_{window}_{func}'] = self.data[col].rolling(window=window).min()
                elif func == 'max':
                    self.data[f'{col}_rolling_{window}_{func}'] = self.data[col].rolling(window=window).max()
                elif func == 'sum':
                    self.data[f'{col}_rolling_{window}_{func}'] = self.data[col].rolling(window=window).sum()

        return self

    def bin_data(self, columns: Union[str, List[str]], bins: int = 5,
                 labels: List[str] = None) -> 'DataPreprocessor':
        """
        对连续变量进行分箱处理
        :param columns: 要分箱的列
        :param bins: 分箱数量
        :param labels: 分箱标签
        :return: self
        """
        if self.data is None:
            raise ValueError("没有加载数据")

        if isinstance(columns, str):
            columns = [columns]

        for col in columns:
            if col not in self.data.columns or not pd.api.types.is_numeric_dtype(self.data[col]):
                continue

            self.data[f'{col}_binned'] = pd.cut(self.data[col], bins=bins, labels=labels)

        return self

    def filter_by_condition(self, condition: str) -> 'DataPreprocessor':
        """
        根据条件表达式筛选数据
        :param condition: 条件表达式，如 "age > 18 and income < 50000"
        :return: self
        """
        if self.data is None:
            raise ValueError("没有加载数据")

        self.data = self.data.query(condition).reset_index(drop=True)
        return self

    def sample_data(self, n: int = None, frac: float = None, random_state: int = None) -> 'DataPreprocessor':
        """
        随机采样数据
        :param n: 采样数量
        :param frac: 采样比例（0-1之间）
        :param random_state: 随机种子
        :return: self
        """
        if self.data is None:
            raise ValueError("没有加载数据")

        self.data = self.data.sample(n=n, frac=frac, random_state=random_state).reset_index(drop=True)
        return self

    def merge_rare_categories(self, columns: Union[str, List[str]], threshold: float = 0.05,
                              new_category: str = 'Other') -> 'DataPreprocessor':
        """
        合并低频类别
        :param columns: 要处理的分类列
        :param threshold: 频率阈值，低于此值的类别会被合并
        :param new_category: 合并后的新类别名称
        :return: self
        """
        if self.data is None:
            raise ValueError("没有加载数据")

        if isinstance(columns, str):
            columns = [columns]

        for col in columns:
            if col not in self.data.columns:
                continue

            freq = self.data[col].value_counts(normalize=True)
            rare_categories = freq[freq < threshold].index
            self.data[col] = self.data[col].replace(rare_categories, new_category)

        return self

    def log_transform(self, columns: Union[str, List[str]], add_constant: float = 1) -> 'DataPreprocessor':
        """
        对数变换（用于处理偏态分布）
        :param columns: 要变换的列
        :param add_constant: 添加常数避免log(0)
        :return: self
        """
        if self.data is None:
            raise ValueError("没有加载数据")

        if isinstance(columns, str):
            columns = [columns]

        for col in columns:
            if col not in self.data.columns or not pd.api.types.is_numeric_dtype(self.data[col]):
                continue

            self.data[f'{col}_log'] = np.log(self.data[col] + add_constant)

        return self


    def fill_null_with_knn(self, columns: Union[str, List[str], None] = None, n_neighbors: int = 5) -> 'DataPreprocessor':
        """
        使用KNN算法填充空值
        :param columns: 要处理的列，None表示处理所有数值列
        :param n_neighbors: KNN的邻居数量，默认5
        :return: self
        """
        from sklearn.impute import KNNImputer

        if self.data is None:
            raise ValueError("没有加载数据")

        if columns is None:
            # 只处理数值列
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            if isinstance(columns, str):
                columns = [columns]
            # 验证列存在且为数值类型
            numeric_cols = [col for col in columns if col in self.data.columns
                            and pd.api.types.is_numeric_dtype(self.data[col])]

            if not numeric_cols:
                print("警告: 没有数值列需要处理")
                return self

        # 创建KNN填充器
        imputer = KNNImputer(n_neighbors=n_neighbors)

        # 对指定的数值列进行KNN填充
        self.data[numeric_cols] = imputer.fit_transform(self.data[numeric_cols])

        return self