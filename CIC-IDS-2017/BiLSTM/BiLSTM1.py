import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------------
# 1. CIC-IDS-2017数据集处理
# --------------------------
class CICIDS2017Dataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)  # 多分类标签

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_cic_ids_2017(file_path):
    """加载并预处理CIC-IDS-2017数据集"""
    # 读取数据
    df = pd.read_csv(file_path)

    # 处理分类特征
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # 特征与标签分离
    X = df.drop([' Label'], axis=1).values
    y = df[' Label'].values  # 标签

    # 处理无穷大值和缺失值
    X = np.nan_to_num(X, nan=np.nan, posinf=np.nan, neginf=np.nan)
    valid_indices = ~np.isnan(X).any(axis=1)
    X = X[valid_indices]
    y = y[valid_indices]

    # 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分训练集、验证集、测试集 (70%/10%/20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.125, random_state=42
    )

    # 调整输入形状以适配LSTM [样本数, 时间步, 特征数]
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    print(f"数据集信息:")
    print(f"  总样本数: {len(X)}")
    print(f"  特征维度: {X.shape[1]}")
    print(f"  训练集: {len(X_train)}")
    print(f"  验证集: {len(X_val)}")
    print(f"  测试集: {len(X_test)}")
    print(f"  攻击样本比例: {np.mean(y):.4f}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# --------------------------
# 2. BiLSTM模型定义
# --------------------------
class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=2, dropout=0.3):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            bidirectional=True,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            self.dropout,
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # BiLSTM输出: [batch_size, seq_len, hidden_dim*2]
        lstm_out, _ = self.bilstm(x)

        # 取最后一个时间步的特征
        features = lstm_out[:, -1, :]

        # 分类层
        logits = self.fc(features)
        return logits


# --------------------------
# 3. 模型训练与评估
# --------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20, patience=5):
    """训练模型并应用早停策略"""
    model.to(device)
    best_val_loss = float('inf')
    early_stop_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        # 训练模式
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # 前向传播
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        # 计算训练指标
        train_loss /= len(train_loader)
        train_acc = 100. * correct / total

        # 验证模式
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        # 计算验证指标
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 打印进度
        print(f'Epoch {epoch+1}/{epochs} | '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')

        # 早停策略
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), 'best_bilstm_model.pth')
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break

    # 加载最佳模型
    model.load_state_dict(torch.load('best_bilstm_model.pth'))
    return model, history


def evaluate_model(model, test_loader):
    """评估模型性能"""
    model.to(device)
    model.eval()
    y_true = []
    y_pred = []
    y_scores = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            scores = torch.softmax(output, dim=1)
            _, predicted = output.max(1)

            y_true.extend(target.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            # 多分类场景下建议保存所有类别的概率
            y_scores.extend(scores.cpu().numpy())  # 修改：保存所有类别概率

    # 计算评估指标（显式指定average='weighted'）
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }

    print("\n模型评估报告:")
    print(classification_report(y_true, y_pred))

    return metrics, y_true, y_pred, y_scores


def plot_training_history(history):
    """可视化训练历史"""
    plt.figure(figsize=(12, 4))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# --------------------------
# 4. 主函数
# --------------------------
if __name__ == "__main__":
    # 设置随机种子，确保结果可复现
    torch.manual_seed(42)
    np.random.seed(42)

    # 加载CIC-IDS-2017数据
    print("加载CIC-IDS-2017数据集...")
    file_path = r'D:\code\lunwen\data\merged.csv'
    X_train, X_val, X_test, y_train, y_val, y_test = load_cic_ids_2017(file_path)

    # 创建数据加载器
    train_dataset = CICIDS2017Dataset(X_train, y_train)
    val_dataset = CICIDS2017Dataset(X_val, y_val)
    test_dataset = CICIDS2017Dataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # 初始化模型（注意num_classes自动获取）
    input_dim = X_train.shape[2]  # 特征维度
    num_classes = len(np.unique(y_train))  # 分类类别数

    print(f"\n创建BiLSTM模型 - 输入维度: {input_dim}, 分类类别: {num_classes}")
    model = BiLSTMModel(
        input_dim=input_dim,
        hidden_dim=64,
        num_classes=num_classes,
        dropout=0.3
    )

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    print("\n开始训练模型...")
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        epochs=20, patience=5
    )

    # 评估模型
    print("\n在测试集上评估模型...")
    metrics, y_true, y_pred, y_scores = evaluate_model(model, test_loader)

    # 可视化训练历史
    plot_training_history(history)

    # 打印最终性能指标
    print("\n最终性能指标:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")