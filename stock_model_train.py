import torch
import torch.nn as nn
import math
import pandas as pd
from torch.utils.data import DataLoader

from data_preprocessing import StockDataset
from my_assistant.my_tools import get_stock_data_for_model

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    """
    注入时间位置信息，因为 Transformer 本身不具备时序感知能力
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 注册为 buffer，不参与反向传播更新
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        # 加上位置编码
        x = x + self.pe[:, :x.size(1), :]
        return x


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        """
        Args:
            input_size: 输入特征维度 (例如 open, close, vol, ma5 等共 N 个特征)
            d_model: Transformer 内部的特征维度 (通常要比 input_size 大)
            nhead: 多头注意力的头数
            num_layers: Encoder 层数
        """
        super(TimeSeriesTransformer, self).__init__()

        # 1. 特征映射层: 将原始特征维度映射到 Transformer 的 d_model 维度
        self.input_net = nn.Linear(input_size, d_model)

        # 2. 位置编码
        self.pos_encoder = PositionalEncoding(d_model)

        # 3. Transformer Encoder
        # batch_first=True 意味着输入格式为 [batch_size, seq_len, feature]
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=d_model * 4,
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. 输出层: 回归预测，输出维度为 1 (预测价格)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, src):
        # src shape: [batch_size, seq_len, input_size]

        # 映射特征
        src = self.input_net(src)  # -> [batch_size, seq_len, d_model]

        # 加上位置编码
        src = self.pos_encoder(src)

        # Transformer 计算
        # output shape: [batch_size, seq_len, d_model]
        output = self.transformer_encoder(src)

        # 取最后一个时间步的输出作为预测依据 (Many-to-One 模式)
        # 类似于 LSTM 取 hidden_state[-1]
        last_output = output[:, -1, :]  # -> [batch_size, d_model]

        # 全连接层输出结果
        prediction = self.decoder(last_output)  # -> [batch_size, 1]

        return prediction


class TransformerTrainer:
    def __init__(self, dataframe, seq_len=30,
                 d_model=64, nhead=4, num_layers=2,
                 batch_size=32, learning_rate=0.001, epochs=50):
        """
        dataframe: 必须是 akshare 获取并清洗后的 pd.DataFrame
        """
        self.epochs = epochs
        self.batch_size = batch_size

        # 1. 准备数据集 (使用上一轮我们定义的 StockDataset)
        # 划分 80% 训练, 20% 测试
        train_size = int(len(dataframe) * 0.8)
        train_df = dataframe.iloc[:train_size]
        val_df = dataframe.iloc[train_size:]

        # 训练集 Dataset
        self.train_dataset = StockDataset(train_df, seq_len=seq_len, mode='train')
        # 验证集 Dataset (复用训练集的 scaler)
        scaler = self.train_dataset.get_scaler()
        self.val_dataset = StockDataset(val_df, seq_len=seq_len, scaler=scaler, mode='test')

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)

        # 2. 自动获取输入特征维度
        sample_x, _ = self.train_dataset[0]
        input_size = sample_x.shape[1]  # 特征数量

        # 3. 初始化 Transformer 模型
        self.model = TimeSeriesTransformer(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        ).to(device)

        # 4. 优化器与损失函数
        # 回归任务使用 MSELoss (均方误差)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)

    def train(self):
        print(f"开始训练... 设备: {device}")
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0

            for x_batch, y_batch in self.train_loader:
                x_batch = x_batch.to(device)  # [batch, seq_len, features]
                y_batch = y_batch.to(device)  # [batch]

                # 前向传播
                outputs = self.model(x_batch)
                # outputs 形状是 [batch, 1]，y_batch 是 [batch]
                # 需要调整形状一致
                loss = self.criterion(outputs.squeeze(), y_batch)

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            self.scheduler.step()

            # 验证集评估
            val_loss = self.evaluate()

            if (epoch + 1) % 5 == 0:
                avg_train_loss = train_loss / len(self.train_loader)
                print(
                    f"Epoch [{epoch + 1}/{self.epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")

        print("训练完成")
        return self.model, self.train_dataset.get_scaler()

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in self.val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = self.model(x_batch)
                loss = self.criterion(outputs.squeeze(), y_batch)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)


import pandas as pd
import numpy as np
import torch


def train_model(stock_data_path):
    """
    训练模型并返回训练结果、模型对象和评估数据。
    """
    df = pd.read_csv(stock_data_path)

    if df.empty:
        print("数据为空，无法训练")
        return None

    # 2. 初始化训练器
    trainer = TransformerTrainer(
        dataframe=df,
        seq_len=20,  # 用过去20天
        d_model=32,  # 隐层维度
        nhead=2,  # 注意力头数
        num_layers=1,
        batch_size=16,
        epochs=30
    )

    # 3. 训练
    # model: 训练好的网络
    # scaler: 训练好的归一化器 (非常重要，未来预测必须用它)
    model, scaler = trainer.train()

    # 4. 简单评估 (取验证集最后一条数据)
    model.eval()
    test_dataset = trainer.val_dataset

    # 以此为例：取验证集最后一个窗口的数据
    x, y_true = test_dataset[len(test_dataset) - 1]
    x_tensor = x.unsqueeze(0).to(device)  # 增加 batch 维度

    with torch.no_grad():
        y_pred_norm = model(x_tensor).item()

    # --- 反归一化逻辑 ---
    # 构建一个形状为 (1, 特征数) 的 dummy 数组，全填0
    dummy_row = np.zeros((1, len(trainer.train_dataset.feature_cols)))

    # 获取目标列（close）在特征列表中的索引
    target_idx = trainer.train_dataset.target_idx

    # A. 计算预测股价 (Real Prediction)
    dummy_row[0, target_idx] = y_pred_norm
    y_pred_real = scaler.inverse_transform(dummy_row)[0, target_idx]

    # B. 计算真实股价 (Real Truth)
    dummy_row[0, target_idx] = y_true.item()
    y_true_real = scaler.inverse_transform(dummy_row)[0, target_idx]

    # 5. 构造返回结果字典
    result = {
        "model": model,  # 训练好的模型对象
        "scaler": scaler,  # 训练好的归一化器
        "feature_cols": trainer.train_dataset.feature_cols,  # 特征列名
        "metrics": {  # 数值结果
            "pred_norm": y_pred_norm,  # 归一化预测值
            "true_norm": y_true.item(),  # 归一化真实值
            "pred_price": round(y_pred_real, 2),  # 真实预测股价
            "true_price": round(y_true_real, 2),  # 真实实际股价
            "diff": round(y_pred_real - y_true_real, 2)  # 误差
        }
    }

    return result

