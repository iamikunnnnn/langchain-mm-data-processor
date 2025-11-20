import os
from datetime import datetime

import joblib
import matplotlib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, \
    GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report, confusion_matrix
import inspect
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
from sklearn import tree
# 设置字体，支持中文显示
matplotlib.rcParams["font.family"] = ["KaiTi"]
# 解决负号显示问题（可选）
matplotlib.rcParams["axes.unicode_minus"] = False  # 正确显示负号
class model_choice:
    def __init__(self, model_name, data, X_columns, y_column, test_size=0.2, random_state=42, model_param=None,
                 mode="回归"):
        self.data = data
        self.X_columns = X_columns
        self.y_column = y_column
        self.test_size = test_size
        self.random_state = random_state
        self.model_name = model_name
        self.model_param = model_param
        self.mode = mode
        # 模型字典（这里先构造，不带参数）
        if self.mode == '回归':
            self.model_dict = {
                "KNN": KNeighborsRegressor,
                "线性回归": LinearRegression,
                "决策树": DecisionTreeRegressor,
                "随机森林": RandomForestRegressor,
                "梯度提升树": GradientBoostingRegressor,
                "支持向量机": SVR,
            }
        else :
            self.model_dict = {
                "KNN": KNeighborsClassifier,
                "逻辑回归": LogisticRegression,
                "决策树": DecisionTreeClassifier,
                "随机森林": RandomForestClassifier,
                "梯度提升树": GradientBoostingClassifier,
                "支持向量机": SVC,
            }

    def init_model(self):
        """
        查模型字典：确认模型是否在支持列表里,然后加载参数。
        参数过滤：只取合法的参数。
        **实例化模型(核心功能，用户不一定每个参数都会传)**：有参数就带参数实例化，否则用默认参数。
        :return:
        """
        # 检查模型名是否存在
        if self.model_name not in self.model_dict:
            return f"模型 {self.model_name} 不在支持列表里: {list(self.model_dict.keys())}"

        self.model_name = self.model_name
        model_class = self.model_dict[self.model_name]

        # 接收参数
        if self.model_param is not None:
            self.valid_keys = inspect.signature(model_class).parameters.keys()
            filtered_params = {k: v for k, v in self.model_param.items() if k in self.valid_keys}
            # 实例化模型
            self.model = model_class(**filtered_params)
        else:
            # 没有传参数就用默认
            self.model = model_class()

    def get_params(self):
        """
        用于返回参数,用于UI初步获取参数列表
        :return:
        """
        model_class = self.model_dict[self.model_name]  # 临时性的实例化模型来获取参数
        valid_keys = list(inspect.signature(model_class).parameters.keys())
        return valid_keys

    def update_params(self, model_param):
        """
        更新参数
        :param model_param:
        :return:
        """
        self.model_param = model_param

    def my_train_test_split(self):
        # 检查原始数据是否有空值
        if self.data.isna().sum().sum() > 0:
            return "数据中存在空值，请先处理缺失值"

        # 提取特征和目标
        X = self.data[self.X_columns].copy()  # 使用 .copy() 避免视图问题
        y = self.data[self.y_column].copy()

        # ===== 关键修复：确保 y 是 Series =====
        if isinstance(y, pd.DataFrame):
            if len(self.y_column) == 1:
                y = y.iloc[:, 0]  # 转为 Series
            else:
                # 如果有多个目标列，某些模型可能不支持
                pass

        # 过滤类别（只对分类任务需要）
        if self.mode == "分类":
            counts = y.value_counts()
            valid_classes = counts[counts > 1].index

            # 创建布尔掩码
            mask = y.isin(valid_classes)
            X_filtered = X[mask].copy()
            y_filtered = y[mask].copy()

            # ===== 关键修复：重置索引 =====
            X_filtered = X_filtered.reset_index(drop=True)
            y_filtered = y_filtered.reset_index(drop=True)

            # 检查过滤后是否还有数据
            if len(X_filtered) == 0:
                return "过滤后没有有效数据"

            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                X_filtered, y_filtered,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y_filtered
            )
        else:
            # 回归任务
            X_filtered = X.reset_index(drop=True)
            y_filtered = y.reset_index(drop=True)

            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                X_filtered, y_filtered,
                test_size=self.test_size,
                random_state=self.random_state
            )

        # return self.x_train, self.x_test, self.y_train, self.y_test

    def train(self):
        # self._my_train_test_split()
        self.model.fit(self.x_train, self.y_train)

    def predict(self, X=None):
        if X is None:
            X = self.x_test
            self.pred = self.model.predict(X)
        return self.pred

    def evaluate(self):
        self.y_pred = self.predict()
        if self.mode == "回归":
            mse = mean_squared_error(self.y_test, self.y_pred)
            r2 = r2_score(self.y_test, self.y_pred)
            dict = {
                "mse": mse,
                "r2": r2,
            }
            return self.model_name, dict
        else:
            acc = accuracy_score(self.y_test, self.y_pred)
            labels = sorted(self.y_test.unique())  # 实际类别
            classification = classification_report(self.y_test, self.y_pred,labels=labels, target_names=[str(l) for l in labels] )
            dict = {
                "classification": classification_report
            }
            return self.model_name, {"classification": classification}

    def download_model(self, save_dir="saved_models", filename=None):
        """
        保存训练好的模型到本地（.joblib 格式）

        :param save_dir: 保存目录（默认 "saved_models"）
        :param filename: 自定义文件名（默认自动生成，包含模型名和时间戳）
        :return: 保存的文件路径
        """

        # 检查模型是否已训练
        if not hasattr(self, "model") or not hasattr(self.model, "fit"):
            return "模型未初始化或未训练，请先调用 train() 方法"

        # 创建保存目录（不存在则自动创建）
        os.makedirs(save_dir, exist_ok=True)

        # 自动生成文件名（模型名 + 时间戳，避免重复）
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.model_name}_{timestamp}.joblib"

        # 完整保存路径
        save_path = os.path.join(save_dir, filename)

        # 序列化并保存模型
        joblib.dump(self.model, save_path)
        print(f"模型已成功保存到：{save_path}")
        return save_path
    def train_bayesian_search(self):
        """
        贝叶斯搜索
        :return:
        """
        from skopt import BayesSearchCV
        from skopt.space import Real, Integer, Categorical  # 定义参数搜索空间
        from param import get_model_param_spaces
        param_space = get_model_param_spaces()[self.mode][self.model_name]
        # 2. 定义交叉验证
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # 3. 初始化BayesSearchCV（注意参数名是search_spaces，不是param_grid）
        bayes_search = BayesSearchCV(
            estimator=self.model,
            search_spaces=param_space,  # 这里用search_spaces，而非param_grid
            cv=cv,
            n_iter=5,  # 搜索5次（刚好覆盖所有可能的奇数，实际可根据需要增加）
            random_state=42
        )
        bayes_search.fit(self.x_train,self.y_train)

        self.model = bayes_search.best_estimator_
        return bayes_search.best_params_
    def load_model(self,save_path="./machien_learning_model/梯度提升树_20251116_001219.joblib"):
        self.model = joblib.load(save_path)

    # 绘制决策边界
    def plot_boundary(self, pca=None):
        """
        绘制决策边界和数据点
        :param y: 数据标签
        :param pca: PCA对象，用于降维（在函数外已经使用了pca的情况，防止两个pca不一样，输入pca保持降维结果一致）
        :return: None
        """
        data = np.array(self.x_test)
        if data.ndim == 1 or data.shape[1] == 1:
            print("维度过低")
            return

            # PCA 降维（如果传入pca则直接使用传入的pca）
        if data.shape[1] > 2:
            if pca is None:
                pca = PCA(n_components=2)
                data_reduced = pca.fit_transform(data)  # 使用不同的变量名
            else:
                data_reduced = pca.transform(data)
            need_inv = True  # 用于标记是否经历过降维（如果降维过在后面需要重新升维，防止与estimator的训练维度不统一）
        else:
            data_reduced = data
            need_inv = False  # 用于标记是否经历过降维（如果降维过在后面需要重新升维，防止与estimator的训练维度不统一）

        # 基于降维后的数据创建网格
        min_x, max_x = data_reduced[:, 0].min() - 1, data_reduced[:, 0].max() + 1
        min_y, max_y = data_reduced[:, 1].min() - 1, data_reduced[:, 1].max() + 1

        xx, yy = np.meshgrid(
            np.arange(min_x, max_x, 0.3),
            np.arange(min_y, max_y, 0.3)
        )

        # 转为一维且合并的网格点，每行代表一个网格点，用于后续预测
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # 如果降过维，把网格点升回原始空间再预测
        if need_inv:
            grid_original = pca.inverse_transform(grid_points)
        else:
            grid_original = grid_points

        # 预测网格点的类别
        Z = self.model.predict(grid_original)
        # 将预测好的类别reshape为xx的形状，用于绘制是对应每个网格点对应的类别
        Z = Z.reshape(xx.shape)
        # 绘图
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.contourf(xx, yy, Z, alpha=0.3)
        ax.scatter(data_reduced[:, 0], data_reduced[:, 1], c=self.y_test, edgecolors='k', s=50)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        return fig,ax


    def plot_true_pred(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(self.y_test, self.y_pred, alpha=0.6)
        ax.plot([self.y_test.min(), self.y_test.max()],
                [self.y_test.min(), self.y_test.max()], 'r--')  # 45°参考线
        ax.set_xlabel('真实值')
        ax.set_ylabel('预测值')
        ax.set_title('真实值 vs 预测值')
        return fig,ax

    def plot_tree(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        # 4. 在指定 ax 上绘制决策树
        plot_tree(self.model, ax=ax, feature_names=self.X_columns,
                  class_names=sorted(set(self.y_column)), filled=True, rounded=True)

    def plot_fit_true(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(range(len(self.y_test)),self.y_test,color='b',label="真实值")
        ax.plot(range(len(self.y_pred)),self.y_pred, color='red',label="预测值")
        ax.legend(loc='upper right')
        ax.grid(True)
        return fig,ax