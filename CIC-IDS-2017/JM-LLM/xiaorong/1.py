import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import warnings
warnings.filterwarnings('ignore')

# ===================== 全局参数配置 =====================
# 随机种子（保证可复现）
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 训练超参数
BATCH_SIZE = 1024
EPOCHS = 20
LR = 0.001
THRESHOLD = 0.1
# 数据集路径（按需修改）
DATA_PATH = r"D:\code\lunwen\data\merged.csv"

# ===================== 1. 数据预处理模块 =====================
def load_and_preprocess_data(data_path):
    """加载并预处理网络入侵数据集，适配消融实验"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据集不存在: {data_path}")
    # 读取数据
    data = pd.read_csv(data_path)
    # 定位标签列
    label_col = next((col for col in ['Label', ' Label', 'label', 'Class'] if col in data.columns), None)
    if not label_col:
        raise ValueError("未找到标签列")
    # 特征标签分离
    X = data.drop([label_col], axis=1)
    y = (data[label_col] != 'BENIGN').astype(int).values.ravel()
    # 预处理：编码、缺失值、异常值、标准化
    X = pd.get_dummies(X, columns=X.select_dtypes(include=['object']).columns)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())
    # 异常值截断
    for col in X.select_dtypes(include=[np.number]).columns:
        Q1, Q3 = X[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        X[col] = X[col].clip(Q1-10*IQR, Q3+10*IQR)
    # 标准化
    X = StandardScaler().fit_transform(X)
    # 分层划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    # 转换张量
    X_train = torch.FloatTensor(X_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_train = torch.LongTensor(y_train).to(device)
    y_test = torch.LongTensor(y_test).to(device)
    return X_train, X_test, y_train, y_test

# ===================== 2. 消融实验模型定义 =====================
# ------------ Baseline: 纯数据驱动基线模型（无任何JM模块）------------
class BaselineModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
        self.optimizer = optim.Adam(self.parameters(), lr=LR)
        self.loss_fn = nn.BCELoss()

    def forward(self, x):
        return self.model(x)

# ------------ Ablation-1: 移除JM约旦分解模块（仅普通特征融合）------------
class Ablation1_NoJordanDecomp(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # 无约旦分解，仅普通全连接层
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
        self.optimizer = optim.Adam(self.parameters(), lr=LR)
        self.loss_fn = nn.BCELoss()

    def forward(self, x):
        return self.fc(x)

# ------------ Ablation-2: 移除JM先验约束（有分解、无注意力引导）------------
class Ablation2_NoJMPrior(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # 保留约旦分解结构，剔除先验约束引导
        self.jordan_feat = nn.Linear(input_dim, 64)
        self.head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        self.optimizer = optim.Adam(self.parameters(), lr=LR)
        self.loss_fn = nn.BCELoss()

    def jordan_decomp(self, x):
        # 基础约旦分解，无先验权重约束
        return self.jordan_feat(x)

    def forward(self, x):
        feat = torch.tanh(self.jordan_decomp(x))
        return self.head(feat)

# ------------ Ablation-3: 移除高阶约旦块（仅一阶约旦块）------------
class Ablation3_NoHighOrderBlock(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # 仅保留一阶约旦块，无高阶级联传播刻画
        self.first_order = nn.Linear(input_dim + 1, 64)  # 一阶反馈
        self.head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        self.optimizer = optim.Adam(self.parameters(), lr=LR)
        self.loss_fn = nn.BCELoss()

    def forward(self, x):
        batch_size = x.size(0)
        prev_out = torch.zeros(batch_size, 1, device=device)
        # 仅一阶约旦块，无高阶耦合
        combined = torch.cat([x, prev_out], dim=1)
        hidden = torch.tanh(self.first_order(combined))
        return self.head(hidden)

# ------------ 完整JM融合模型（对照组）------------
class FullJMModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # 完整约旦矩阵：分解+先验约束+高阶块
        self.hidden_dim = 64
        self.jordan_encoder = nn.Linear(input_dim + 1, self.hidden_dim)
        # 高阶约旦块（刻画级联传播）
        self.high_order_block = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.attention = nn.Linear(self.hidden_dim, self.hidden_dim)  # JM先验注意力
        self.head = nn.Sequential(nn.Linear(self.hidden_dim, 1), nn.Sigmoid())
        self.optimizer = optim.Adam(self.parameters(), lr=LR)
        self.loss_fn = nn.BCELoss()

    def forward(self, x):
        batch_size = x.size(0)
        prev_out = torch.zeros(batch_size, 1, device=device)
        # 约旦分解
        combined = torch.cat([x, prev_out], dim=1)
        jordan_feat = torch.tanh(self.jordan_encoder(combined))
        # 高阶约旦块+先验注意力约束
        high_feat = torch.tanh(self.high_order_block(jordan_feat))
        attn_weight = torch.sigmoid(self.attention(high_feat))
        feat = high_feat * attn_weight
        return self.head(feat)

# ===================== 3. 训练与评估工具 =====================
def train_model(model, X_train, y_train):
    """统一训练逻辑"""
    model.train()
    loader = DataLoader(TensorDataset(X_train, y_train), BATCH_SIZE, shuffle=True)
    for _ in range(EPOCHS):
        for x, y in loader:
            model.optimizer.zero_grad()
            pred = model(x)
            loss = model.loss_fn(pred, y.float().view(-1,1))
            loss.backward()
            model.optimizer.step()
    return model

def evaluate_model(model, X_test, y_test):
    """统一评估指标：准确率、F1、MSE"""
    model.eval()
    with torch.no_grad():
        pred = model(X_test).cpu().numpy().ravel()
        y_true = y_test.cpu().numpy()
        y_pred = (pred > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mse = mean_squared_error(y_true, pred)
    return round(acc,4), round(f1,4), round(mse,4)

# ===================== 4. 消融实验主流程 =====================
if __name__ == "__main__":
    # 加载数据
    print("========== 开始预处理数据集 ==========")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(DATA_PATH)
    input_dim = X_train.shape[1]

    # 构建消融分组模型
    ablation_groups = {
        "Baseline (纯数据驱动)": BaselineModel(input_dim).to(device),
        "Ablation-1 (移除约旦分解)": Ablation1_NoJordanDecomp(input_dim).to(device),
        "Ablation-2 (移除JM先验约束)": Ablation2_NoJMPrior(input_dim).to(device),
        "Ablation-3 (移除高阶约旦块)": Ablation3_NoHighOrderBlock(input_dim).to(device),
        "Full-Model (完整JM模型)": FullJMModel(input_dim).to(device)
    }

    # 执行消融实验
    print("\n========== 开始消融实验 ==========")
    results = {}
    for name, model in ablation_groups.items():
        print(f"\n训练模型: {name}")
        trained_model = train_model(model, X_train, y_train)
        acc, f1, mse = evaluate_model(trained_model, X_test, y_test)
        results[name] = {"TestAcc": acc, "F1": f1, "MSE": mse}
        print(f"评估结果 - Acc: {acc}, F1: {f1}, MSE: {mse}")

    # 输出消融结果汇总
    print("\n========== 消融实验结果汇总 ==========")
    for name, metric in results.items():
        print(f"{name}: {metric}")
