import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# 加载UNSW-NB15数据集
def load_unsw_nb15(data_path):
    # 读取CSV文件
    df = pd.read_csv(data_path)
    print(f"原始数据集大小: {df.shape}")

    # 数据清洗
    df = df.dropna()  # 移除缺失值
    df = df.drop_duplicates()  # 移除重复记录
    print(f"清洗后数据集大小: {df.shape}")

    # 特征选择（根据需要调整）
    numeric_features = [
        'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate',
        'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss',
        'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb',
        'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean',
        'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src',
        'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
        'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd',
        'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports'
    ]

    categorical_features = [
        'proto', 'service', 'state'
    ]

    # 处理数值特征
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[numeric_features])

    # 处理分类特征
    X_cat = pd.get_dummies(df[categorical_features])

    # 合并特征
    X = np.hstack([X_num, X_cat.values])

    # 处理标签
    le = LabelEncoder()
    y = le.fit_transform(df['label'])  # 0=正常，1=攻击

    return X, y


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# 训练函数
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


# 评估函数
def evaluate(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    pre = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    return {
        'acc': acc,
        'pre': pre,
        'rec': rec,
        'f1': f1,
        'auc': auc
    }


# 主函数
def main():
    train_X, train_y = load_unsw_nb15(r'D:\code\github\kaggle_data\mrwellsdavid\unsw-nb15\versions\1\UNSW_NB15_training-set.csv')
    test_X, test_y = load_unsw_nb15(r'D:\code\github\kaggle_data\mrwellsdavid\unsw-nb15\versions\1\UNSW_NB15_training-set.csv')

    train_dataset = CustomDataset(train_X, train_y)
    test_dataset = CustomDataset(test_X, test_y)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BiLSTMModel(
        input_dim=train_X.shape[1],
        hidden_dim=128,
        output_dim=2,
        num_layers=2,
        dropout=0.5
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    best_f1 = 0
    patience = 10
    counter = 0
    best_loss = float('inf')

    for epoch in range(1, 201):
        loss = train(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, test_loader, device)

        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Acc: {metrics["acc"]:.4f}, '
              f'Pre: {metrics["pre"]:.4f}, Rec: {metrics["rec"]:.4f}, F1: {metrics["f1"]:.4f}, AUC: {metrics["auc"]:.4f}')

        # 早停策略
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_loss = loss
            torch.save(model.state_dict(), 'best_model.pt')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print('Early stopping!')
                break

    # 加载最佳模型并评估
    model.load_state_dict(torch.load('best_model.pt'))
    final_metrics = evaluate(model, test_loader, device)
    print(f'Final Acc: {final_metrics["acc"]:.4f}, Pre: {final_metrics["pre"]:.4f}, '
          f'Rec: {final_metrics["rec"]:.4f}, F1: {final_metrics["f1"]:.4f}, AUC: {final_metrics["auc"]:.4f}')


if __name__ == "__main__":
    main()