import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool

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
    
    return X, y, df

# 构建图数据
def build_graph_data(X, y, df, k_neighbors=3):
    # 使用KNN算法构建图连接
    from sklearn.neighbors import kneighbors_graph
    adj_matrix = kneighbors_graph(X, k_neighbors, mode='distance', include_self=False)
    
    # 转换为稀疏矩阵格式
    edge_index = []
    edge_weight = []
    
    for i, j in zip(*adj_matrix.nonzero()):
        edge_index.append([i, j])
        edge_weight.append(adj_matrix[i, j])
    
    # 对边权重进行归一化
    edge_weight = np.exp(-np.array(edge_weight))  # 使用高斯核
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    
    # 节点特征和标签
    x = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)
    
    # 创建图数据对象
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
    
    # 划分训练集、验证集和测试集
    indices = torch.arange(data.num_nodes)
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2
    
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    
    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True
    
    return data

class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, dropout=0.5, gnn_type='GCN'):
        super().__init__()
        self.dropout = dropout
        self.layers = torch.nn.ModuleList()
        
        # 输入层
        if gnn_type == 'GCN':
            self.layers.append(GCNConv(input_dim, hidden_dim))
        elif gnn_type == 'GAT':
            self.layers.append(GATConv(input_dim, hidden_dim))
        elif gnn_type == 'GraphSAGE':
            self.layers.append(SAGEConv(input_dim, hidden_dim))
        
        # 隐藏层
        for _ in range(num_layers - 2):
            if gnn_type == 'GCN':
                self.layers.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == 'GAT':
                self.layers.append(GATConv(hidden_dim, hidden_dim))
            elif gnn_type == 'GraphSAGE':
                self.layers.append(SAGEConv(hidden_dim, hidden_dim))
        
        # 输出层
        self.layers.append(GCNConv(hidden_dim, output_dim))
    
    def forward(self, x, edge_index, edge_weight=None):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.layers[-1](x, edge_index, edge_weight)
        return x

# 训练函数
def train(model, data, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(data.x, data.edge_index, data.edge_attr)
    loss = criterion(outputs[data.train_mask], data.y[data.train_mask])
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

# 评估函数
def evaluate(model, data, device):
    model.eval()
    
    outputs = model(data.x, data.edge_index, data.edge_attr)
    pred = outputs.argmax(dim=1)  # 使用最大概率作为预测结果
    
    # 计算准确率
    train_acc = pred[data.train_mask].eq(data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()
    val_acc = pred[data.val_mask].eq(data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
    test_acc = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    
    # 计算RPA（相对传播准确率）
    from sklearn.metrics import precision_recall_fscore_support
    
    rpa = precision_recall_fscore_support(
        data.y[data.test_mask], 
        pred[data.test_mask], 
        average='macro'
    )[2]  # F1分数作为RPA的近似
    
    return {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'rpa': rpa
    }

# 主函数
def main():
    X, y, df = load_unsw_nb15(r'D:\code\github\kaggle_data\mrwellsdavid\unsw-nb15\versions\1\UNSW_NB15_training-set.csv')
    
    data = build_graph_data(X, y, df, k_neighbors=3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
        
    gnn = GNNModel(
            input_dim=data.x.size(1),
            hidden_dim=128,
            output_dim=2,
            num_layers=4
    ).to(device)

    # 优化器和损失函数
    optimizer = torch.optim.Adam(gnn.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # 训练循环
    best_val_acc = 0
    patience = 10
    counter = 0
    best_loss = float('inf')

    for epoch in range(1, 201):
        loss = train(gnn, data, optimizer, criterion, device)
        metrics = evaluate(gnn, data, device)

        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {metrics["train_acc"]:.4f}, '
              f'Val: {metrics["val_acc"]:.4f}, Test: {metrics["test_acc"]:.4f}, RPA: {metrics["rpa"]:.4f}')
        
        # 早停策略
        if metrics['val_acc'] > best_val_acc:
            best_val_acc = metrics['val_acc']
            best_loss = loss
            torch.save(gnn.state_dict(), 'best_model.pt')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print('Early stopping!')
                break
    
    # 加载最佳模型并评估
    gnn.load_state_dict(torch.load('best_model.pt'))
    final_metrics = evaluate(gnn, data, device)
    print(f'Final Test Acc: {final_metrics["test_acc"]:.4f}, RPA: {final_metrics["rpa"]:.4f}')

if __name__ == "__main__":
    main()