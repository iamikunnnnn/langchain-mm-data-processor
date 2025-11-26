import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import levene, bartlett, shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from itertools import combinations
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.linear_model import LinearRegression
from itertools import combinations

class EnhancedDataAnalyzer:
    def __init__(self, data):
        self.data = data

    def get_basic_info(self):
        """
        获取数据基本信息，根据数据类型（定类/定量）选择性计算统计量
        :return: dict: 包含数据基本信息
        """
        if self.data is None:
            return {"error": "没有加载数据"}

        # 筛选数值型列
        numeric_cols = self.data.select_dtypes(include=['number']).columns.tolist()

        # 1. 多重共线性检验 (VIF)
        multicollinearity_info = self._check_multicollinearity(numeric_cols)

        # 2. 方差齐性检验
        homogeneity_info = self._check_homogeneity()

        # 3. 异方差性检验
        # heteroscedasticity_info = self._check_heteroscedasticity(numeric_cols)

        # 4. 相关性分析
        correlation_analysis = self._correlation_analysis()

        # 5. 按数据类型计算统计信息
        stats_info = self._calculate_column_statistics()

        return {
            "形状": self.data.shape,
            "列名": self.data.columns.tolist(),
            "空值数量": dict(self.data.isna().sum()),
            "重复行数量": self.data.duplicated().sum(),
            "前二行数据": self.data.head(2).to_dict(orient="records"),
            "多重共线性检验(VIF)": multicollinearity_info,
            "方差齐性检验": homogeneity_info,
            # "异方差性检验": heteroscedasticity_info,
            "相关性分析": correlation_analysis,
            "列统计信息": stats_info
        }

    def data_info(self):
        return {
            "形状": self.data.shape,
            "列名": self.data.columns.tolist(),
            "空值数量": dict(self.data.isna().sum()),
            "重复行数量": self.data.duplicated().sum(),
            "前二行数据": self.data.head(2).to_dict(orient="records"),
        }
    def multicollinearity_info(self):
        multicollinearity_info = self._check_multicollinearity(self.numeric_cols)
        return multicollinearity_info
    def homogeneity_info(self):
        homogeneity_info = self._check_homogeneity()
        return homogeneity_info
    def correlation_analysis(self):
        correlation_analysis = self._correlation_analysis()
        return correlation_analysis
    def stats_info(self):
        stats_info = self._calculate_column_statistics()
        return stats_info
    def _check_multicollinearity(self, numeric_cols):
        """
        多重共线性检验 - VIF (Variance Inflation Factor)
        VIF > 10: 严重共线性
        VIF > 5: 中度共线性
        VIF < 5: 可接受
        """
        if len(numeric_cols) < 2:
            return {"message": "数值列少于2列，无法检验"}

        try:
            data_clean = self.data[numeric_cols].dropna()
            if len(data_clean) < len(numeric_cols) + 1:
                return {"error": "样本量不足"}

            vif_data = []
            for i, col in enumerate(numeric_cols):
                try:
                    vif = variance_inflation_factor(data_clean.values, i)
                    vif_data.append({
                        "变量": col,
                        "VIF": round(vif, 2),
                        "共线性等级": "S" if vif > 10 else "中度" if vif > 5 else "可接受"
                    })
                except:
                    vif_data.append({"变量": col, "VIF": "计算失败"})

            return vif_data
        except Exception as e:
            return {"error": f"VIF检验失败: {str(e)}"}

    def _check_homogeneity(self):
        """
        方差齐性检验（针对分类变量vs数值变量）
        使用 Levene 检验
        """
        results = []
        numeric_cols = self.data.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()

        if not numeric_cols or not categorical_cols:
            return {"message": "需要同时存在数值列和分类列"}

        for cat_col in categorical_cols:
            for num_col in numeric_cols:
                try:
                    groups = [group[num_col].dropna() for name, group in self.data.groupby(cat_col)]
                    groups = [g for g in groups if len(g) >= 2]

                    if len(groups) < 2:
                        continue

                    stat, p_value = levene(*groups)
                    results.append({
                        "分类变量": cat_col,
                        "数值变量": num_col,
                        # "Levene统计量": round(stat, 4),
                        # "p值": round(p_value, 4),
                        "方差齐性": "Y" if p_value > 0.05 else "否"
                    })
                except:
                    continue

        return results if results else {"message": "无可用的分类-数值列对"}

    def _check_heteroscedasticity(self, numeric_cols):
        """
        异方差性检验 - Breusch-Pagan 检验
        针对回归模型的残差
        """
        if len(numeric_cols) < 2:
            return {"message": "数值列少于2列，无法检验"}



        results = []
        for x_col, y_col in combinations(numeric_cols, 2):

            data_clean = self.data[[x_col, y_col]].dropna()
            if len(data_clean) < 10:
                continue

            X = data_clean[[x_col]].values
            y = data_clean[y_col].values

            # 简单线性回归
            model = LinearRegression()
            model.fit(X, y)
            residuals = y - model.predict(X)

            # BP检验必须添加常数项
            X_bp = sm.add_constant(X)

            try:
                _, p_value, _, _ = het_breuschpagan(residuals, X_bp)
            except Exception as e:
                p_value = None

            results.append({
                "自变量": x_col,
                "因变量": y_col,
                # "BP_p值": None if p_value is None else round(p_value, 4),
                "异方差性": "无法判断" if p_value is None else ("Y" if p_value < 0.05 else "不存在")
            })

        return results

    def _correlation_analysis(self):
        """
        完整的相关性分析链路
        """
        results = {
            "数值_vs_数值": [],
            "分类_vs_数值": [],
            "分类_vs_分类": []
        }

        numeric_cols = self.data.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()

        # ① 数值 vs 数值
        results["数值_vs_数值"] = self._numeric_vs_numeric(numeric_cols)

        # ② 分类 vs 数值
        results["分类_vs_数值"] = self._categorical_vs_numeric(categorical_cols, numeric_cols)

        # ③ 分类 vs 分类
        results["分类_vs_分类"] = self._categorical_vs_categorical(categorical_cols)

        return results

    def _numeric_vs_numeric(self, numeric_cols):
        """数值 vs 数值相关性分析"""
        if len(numeric_cols) < 2:
            return {"message": "数值列少于2列"}

        results = []
        for col1, col2 in combinations(numeric_cols, 2):  # 限制前5列
            try:
                data_pair = self.data[[col1, col2]].dropna()
                if len(data_pair) < 3:
                    continue

                # 正态性检验
                _, p1 = shapiro(data_pair[col1]) if len(data_pair) <= 5000 else (None, 0)
                _, p2 = shapiro(data_pair[col2]) if len(data_pair) <= 5000 else (None, 0)
                is_normal = (p1 > 0.05 and p2 > 0.05) if p1 is not None else False

                # 选择相关性方法
                if is_normal and len(data_pair) >= 50:
                    # Pearson
                    r, p = stats.pearsonr(data_pair[col1], data_pair[col2])
                    method = "Pearson"
                elif len(data_pair) < 50:
                    # Kendall
                    r, p = stats.kendalltau(data_pair[col1], data_pair[col2])
                    method = "Kendall"
                else:
                    # Spearman
                    r, p = stats.spearmanr(data_pair[col1], data_pair[col2])
                    method = "Spearman"

                results.append({
                    "变量1": col1,
                    "变量2": col2,
                    "方法": method,
                    # "相关系数": round(r, 4),
                    # "p值": round(p, 4),
                    "显著性": "Y" if p < 0.05 else "不显著",
                    # "正态性": "Y" if is_normal else "N"
                })
            except:
                continue

        return results

    def _categorical_vs_numeric(self, categorical_cols, numeric_cols):
        """分类 vs 数值分析（均值差异）"""
        if not categorical_cols or not numeric_cols:
            return {"message": "缺少分类列或数值列"}

        results = []
        for cat_col in categorical_cols[:3]:
            for num_col in numeric_cols[:3]:
                try:
                    groups = [group[num_col].dropna() for name, group in self.data.groupby(cat_col)]
                    groups = [g for g in groups if len(g) >= 2]
                    n_groups = len(groups)

                    if n_groups < 2:
                        continue

                    # 正态性检验
                    normal_tests = [shapiro(g)[1] > 0.05 if len(g) <= 5000 else False for g in groups]
                    all_normal = all(normal_tests)

                    if n_groups == 2:
                        # t检验
                        if all_normal:
                            # 方差齐性
                            _, p_levene = levene(*groups)
                            equal_var = p_levene > 0.05
                            stat, p = stats.ttest_ind(groups[0], groups[1], equal_var=equal_var)
                            method = "t-test (equal_var)" if equal_var else "Welch t-test"
                        else:
                            # Mann-Whitney U
                            stat, p = stats.mannwhitneyu(groups[0], groups[1])
                            method = "Mann-Whitney U"
                    else:
                        # ANOVA或Kruskal-Wallis
                        if all_normal:
                            _, p_levene = levene(*groups)
                            if p_levene > 0.05:
                                stat, p = stats.f_oneway(*groups)
                                method = "ANOVA"
                            else:
                                stat, p = stats.kruskal(*groups)
                                method = "Kruskal-Wallis"
                        else:
                            stat, p = stats.kruskal(*groups)
                            method = "Kruskal-Wallis"

                    results.append({
                        "分类变量": cat_col,
                        "数值变量": num_col,
                        # "组数": n_groups,
                        "方法": method,
                        # "统计量": round(stat, 4),
                        # "p值": round(p, 4),
                        "显著性": "Y" if p < 0.05 else "不显著"
                    })
                except:
                    continue

        return results

    def _categorical_vs_categorical(self, categorical_cols):
        """分类 vs 分类分析（独立性检验）"""
        if len(categorical_cols) < 2:
            return {"message": "分类列少于2列"}

        results = []
        for col1, col2 in combinations(categorical_cols[:4], 2):
            try:
                contingency = pd.crosstab(self.data[col1], self.data[col2])

                # 检查期望频数
                chi2, p, dof, expected = stats.chi2_contingency(contingency)
                min_expected = expected.min()

                if min_expected >= 5:
                    # 卡方检验
                    method = "Chi-square"
                    stat, p_value = chi2, p
                else:
                    # Fisher精确检验（仅支持2x2）
                    if contingency.shape == (2, 2):
                        _, p_value = stats.fisher_exact(contingency)
                        method = "Fisher exact"
                        stat = None
                    else:
                        method = "Chi-square (警告:期望频数<5)"
                        stat, p_value = chi2, p

                results.append({
                    "变量1": col1,
                    "变量2": col2,
                    "方法": method,
                    # "统计量": round(stat, 4) if stat is not None else "N/A",
                    # "p值": round(p_value, 4),
                    "显著性": "Y" if p_value < 0.05 else "无显著关联",
                    "最小期望频数": round(min_expected, 2)
                })
            except:
                continue

        return results

    def _calculate_column_statistics(self):
        """计算列统计信息"""
        stats_info = {}
        all_cols = self.data.columns.tolist()

        for col in all_cols:
            col_data = self.data[col].dropna()
            if col_data.empty:
                stats_info[col] = {"error": "该列全为空值", "data_type": "unknown"}
                continue

            # 判断数据类型
            if pd.api.types.is_numeric_dtype(col_data):
                data_type = "quantitative"
            elif pd.api.types.is_object_dtype(col_data) or pd.api.types.is_categorical_dtype(col_data):
                data_type = "categorical"
            else:
                data_type = "other"

            col_stats = {
                "data_type": data_type,
                "count": len(col_data),
                "missing_count": self.data[col].isna().sum()
            }

            # 定量数据统计
            if data_type == "quantitative":
                col_stats.update({
                    "mean": round(col_data.mean(), 2),
                    "median": round(col_data.median(), 2),
                    "std": round(col_data.std(), 2),
                    # "variance": round(col_data.var(), 2),
                    "min": round(col_data.min(), 2),
                    "max": round(col_data.max(), 2),
                    "25%_quantile": round(col_data.quantile(0.25), 2),
                    "75%_quantile": round(col_data.quantile(0.75), 2),
                    # "iqr": round(col_data.quantile(0.75) - col_data.quantile(0.25), 2),
                    "skewness": round(col_data.skew(), 2),
                    # "sum": round(col_data.sum(), 2)
                })

                if col_data.mean() != 0:
                    col_stats["cv"] = round(col_data.std() / col_data.mean(), 2)

                # 正态性检验
                try:
                    if len(col_data) >= 3:
                        stat, p_value = stats.shapiro(col_data) if len(col_data) <= 5000 else (None, None)
                        if stat is not None:
                            col_stats["normality"] = {
                                "statistic": round(stat, 4),
                                "p_value": round(p_value, 4),
                                "is_normal": p_value > 0.05
                            }
                except:
                    col_stats["normality"] = {"error": "检验失败"}

            # 定类数据统计
            elif data_type == "categorical":
                mode = col_data.mode()
                col_stats["mode"] = mode.iloc[0] if not mode.empty else None
                col_stats["top_categories"] = {str(k): v for k, v in col_data.value_counts().head(5).to_dict().items()}
                col_stats["unique_count"] = col_data.nunique()

            stats_info[col] = col_stats

        return stats_info


def filter_normal_stats(path="test_data/train_2.csv"):
    """
    对统计信息进行过滤，只保留有异常、显著性、问题的内容。
    不改变原结构，只过滤掉正常项。
    """


    data = pd.read_csv(path)

    analyzer = EnhancedDataAnalyzer(data)
    result = analyzer.get_basic_info()

    # 深拷贝
    vif_list = result.get("多重共线性检验(VIF)", [])

    for idx in range(len(vif_list) - 1, -1, -1):
        # 判断当前元素的共线性等级是否≠可接受，是则原地删除
        if vif_list[idx].get("共线性等级") == "可接受" or vif_list[idx].get("共线性等级") == "重度":
            vif_list.pop(idx)  # 按索引删除，直接修改原列表
    homogeneity_list = result.get("方差齐性检验", [])

    for idx in range(len(homogeneity_list) - 1, -1, -1):
        # 判断当前元素的共线性等级是否≠可接受，是则原地删除
        if homogeneity_list[idx].get("方差齐性") == "否":
            homogeneity_list.pop(idx)  # 按索引删除，直接修改原列表

    # heteroscedasticity_list = result.get("异方差性检验", [])
    # for idx in range(len(heteroscedasticity_list) - 1, -1, -1):
    #     # 判断当前元素的共线性等级是否≠可接受，是则原地删除
    #     if heteroscedasticity_list[idx].get("异方差性") == "不存在":
    #         heteroscedasticity_list.pop(idx)  # 按索引删除，直接修改原列表

    correlation_dict = result.get("相关性分析", {})
    corr_type_list = ['数值_vs_数值', '分类_vs_数值', '分类_vs_分类']

    for corr_type in corr_type_list:
        # 获取当前类型的相关性列表（如数值_vs_数值的列表）
        corr_list = correlation_dict.get(corr_type, [])
        # 反向遍历索引，原地删除“不显著”的记录
        for idx in range(len(corr_list) - 1, -1, -1):
            # 不同类型的“显著性”字段值不同，需分别判断
            significance = corr_list[idx].get("显著性")
            # 数值_vs_数值/分类_vs_数值：显著值为“显著”；分类_vs_分类：显著值为“显著关联”
            if significance in ("不显著", "无显著关联"):
                corr_list.pop(idx)  # 原地删除不显著的记录


    return result

r=filter_normal_stats()
print(r)