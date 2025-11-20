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

        # 1. 计算相关性>0.5的列对（仅针对定量数据）
        # 筛选数值型列用于相关性分析
        numeric_cols = self.data.select_dtypes(include=['number']).columns.tolist()
        high_corr_pairs = []
        if len(numeric_cols) >= 2:
            corr_matrix = self.data[numeric_cols].corr().abs()
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    if corr_matrix.iloc[i, j] > 0.5:
                        high_corr_pairs.append([numeric_cols[i], numeric_cols[j]])

        # 2. 按数据类型计算统计信息
        stats_info = {}
        all_cols = self.data.columns.tolist()

        for col in all_cols:
            col_data = self.data[col].dropna()
            if col_data.empty:
                stats_info[col] = {"error": "该列全为空值", "data_type": "unknown"}
                continue

            # 判断数据类型（定类/定量）
            if pd.api.types.is_numeric_dtype(col_data):
                data_type = "quantitative"  # 定量数据（数值型）
            else:
                # 检查是否为定类数据（字符串或类别型）
                if pd.api.types.is_object_dtype(col_data) or pd.api.types.is_categorical_dtype(col_data):
                    data_type = "categorical"  # 定类数据
                else:
                    data_type = "other"  # 其他类型（如时间等）

            # 初始化统计信息字典
            col_stats = {
                "data_type": data_type,
                "count": len(col_data),
                "missing_count": self.data[col].isna().sum()
            }

            # 定量数据：计算数值型统计量
            if data_type == "quantitative":
                # 集中趋势
                col_stats["mean"] = round(col_data.mean(), 2)
                col_stats["median"] = round(col_data.median(), 2)

                # 离散程度
                col_stats["std"] = round(col_data.std(), 2)
                col_stats["variance"] = round(col_data.var(), 2)
                col_stats["min"] = round(col_data.min(), 2)
                col_stats["max"] = round(col_data.max(), 2)

                # 分位数
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                col_stats["25%_quantile"] = round(q1, 2)
                col_stats["75%_quantile"] = round(q3, 2)
                col_stats["iqr"] = round(q3 - q1, 2)

                # 分布形态
                col_stats["skewness"] = round(col_data.skew(), 2)

                # 其他衍生指标
                col_stats["sum"] = round(col_data.sum(), 2)
                if col_data.mean() != 0:
                    col_stats["cv"] = round(col_data.std() / col_data.mean(), 2)

                # 正态性检验
                try:
                    if len(col_data) >= 3:
                        stat, p_value = stats.shapiro(col_data)
                        col_stats["normality"] = {
                            "statistic": round(stat, 4),
                            "p_value": round(p_value, 4),
                            "is_normal": p_value > 0.05
                        }
                    else:
                        col_stats["normality"] = {"error": "样本量不足（需≥3）"}
                except:
                    col_stats["normality"] = {"error": "检验失败"}

            # 定类数据：计算分类相关统计量
            elif data_type == "categorical":
                # 众数（最频繁出现的值）
                mode = col_data.mode()
                col_stats["mode"] = mode.iloc[0] if not mode.empty else None

                # 类别频数（前5个，避免类别过多时冗余）
                top_categories = col_data.value_counts().head(5).to_dict()
                col_stats["top_categories"] = {str(k): v for k, v in top_categories.items()}

                # 唯一值数量
                col_stats["unique_count"] = col_data.nunique()

            # 其他类型数据：仅保留基础信息
            else:
                col_stats["basic_info"] = f"非定量/定类数据（{col_data.dtype}）"

            stats_info[col] = col_stats

        return {
            "形状": self.data.shape,
            "列名": all_cols,
            "空值数量": dict(self.data.isna().sum()),
            "重复行数量": self.data.duplicated().sum(),
            "前二行数据": self.data.head(2).to_dict(orient="records"),
            "高相关性列对(>0.5)": high_corr_pairs,  # 仅包含定量列
            "列统计信息": stats_info
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
