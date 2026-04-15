import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.metrics import precision_recall_fscore_support, f1_score
import warnings
import logging

warnings.filterwarnings('ignore')

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_unsw_nb15(data_path):
    """加载并预处理UNSW-NB15数据集"""
    # 加载数据
    df = pd.read_csv(data_path)
    
    # 数据清洗
    df = df.dropna()  # 移除缺失值
    df = df.drop_duplicates()  # 移除重复记录
    
    # 特征选择
    numerical_features = ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 
                         'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 
                         'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 
                         'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 
                         'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 
                         'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 
                         'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports']
    
    categorical_features = ['proto', 'service', 'state']
    
    # 提取特征和标签
    X_num = df[numerical_features].values
    X_cat = df[categorical_features].values
    
    # 标准化数值特征
    scaler = StandardScaler()
    X_num = scaler.fit_transform(X_num)
    
    # 编码分类特征
    X_cat_encoded = []
    for i in range(X_cat.shape[1]):
        le = LabelEncoder()
        X_cat_encoded.append(le.fit_transform(X_cat[:, i]))
    
    # 转换为NumPy数组并确保为数值类型
    X_cat_encoded = np.array(X_cat_encoded).T
    
    # 合并特征
    X = np.hstack([X_num, X_cat_encoded]).astype(np.float32)  # 确保为float32类型
    
    # 处理标签
    y = df['label'].values  # 二分类标签(0=正常,1=攻击)
    attack_cat = df['attack_cat'].values  # 攻击类型
    
    return X, y, attack_cat, df

def build_graph_data(X, y, attack_cat, df, k_neighbors=5):
    """构建图数据"""
    # 使用KNN算法构建图的连接
    adj = kneighbors_graph(X, k_neighbors, mode='distance', include_self=False)
    
    # 将距离转换为相似度（权重）
    adj.data = np.exp(-adj.data)  # 使用高斯核转换
    
    # 确保图是无向的
    adj = 0.5 * (adj + adj.T)
    
    # 获取边索引和权重
    edge_index = []
    edge_attr = []
    
    for i, j in zip(*adj.nonzero()):
        edge_index.append([i, j])
        edge_attr.append(adj[i, j])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    # 节点特征和标签
    x = torch.tensor(X, dtype=torch.float)  # 此时X应为float类型
    y = torch.tensor(y, dtype=torch.long)
    
    # 创建图数据对象
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    # 划分训练集、验证集和测试集
    num_nodes = x.size(0)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)
    
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    return data

class MultiModalGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5, gnn_type='GCN'):
        super(MultiModalGNN, self).__init__()
        
        self.gnn_layers = torch.nn.ModuleList()
        self.bn_layers = torch.nn.ModuleList()
        
        # 输入层
        if gnn_type == 'GCN':
            self.gnn_layers.append(GCNConv(input_dim, hidden_dim))
        elif gnn_type == 'GAT':
            self.gnn_layers.append(GATConv(input_dim, hidden_dim))
        elif gnn_type == 'GraphSAGE':
            self.gnn_layers.append(SAGEConv(input_dim, hidden_dim))
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
        
        self.bn_layers.append(torch.nn.BatchNorm1d(hidden_dim))
        
        # 隐藏层
        for _ in range(num_layers - 1):
            if gnn_type == 'GCN':
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == 'GAT':
                self.gnn_layers.append(GATConv(hidden_dim, hidden_dim))
            elif gnn_type == 'GraphSAGE':
                self.gnn_layers.append(SAGEConv(hidden_dim, hidden_dim))
            
            self.bn_layers.append(torch.nn.BatchNorm1d(hidden_dim))
        
        # 节点分类器
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # 关键节点预测器（用于传播预测）
        self.key_node_predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, 1)
        )
        
        self.dropout = dropout
        self.node_embedding = None  # 用于存储节点嵌入
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # GNN层传播
        for i, (gnn, bn) in enumerate(zip(self.gnn_layers, self.bn_layers)):
            x = gnn(x, edge_index, edge_attr) if edge_attr is not None else gnn(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 保存节点嵌入
        self.node_embedding = x
        
        # 节点分类预测
        class_pred = self.classifier(x)
        
        # 关键节点预测（威胁传播源的可能性）
        key_node_scores = self.key_node_predictor(x)
        
        return {
            'class_pred': class_pred,
            'key_node_scores': key_node_scores,
            'node_embedding': self.node_embedding
        }

class ThreatPropagationPredictor:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.node_embedding = self._compute_node_embedding()
        self.num_nodes = data.x.size(0)
        
        # 构建邻接矩阵用于传播模拟
        self.adj_matrix = torch.zeros((self.num_nodes, self.num_nodes))
        edge_index = data.edge_index.cpu().numpy()
        edge_attr = data.edge_attr.cpu().numpy() if data.edge_attr is not None else np.ones(edge_index.shape[1])
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            # 将NumPy float32转换为PyTorch标量
            self.adj_matrix[src, dst] = torch.tensor(edge_attr[i], dtype=torch.float)
        
        # 归一化邻接矩阵
        row_sum = self.adj_matrix.sum(1, keepdim=True)
        self.adj_matrix = self.adj_matrix / (row_sum + 1e-10)
    
    def _compute_node_embedding(self):
        """计算节点嵌入"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.data)
            return outputs['node_embedding'].detach().cpu().numpy()
    
    def predict_propagation(self, initial_infected_nodes, steps=3, method='random_walk'):
        """
        预测威胁传播过程
        
        参数:
            initial_infected_nodes: 初始被感染节点的索引列表
            steps: 传播步数
            method: 传播方法，'random_walk' 或 'attention'
        
        返回:
            每一步的感染概率矩阵，形状为 (steps+1, num_nodes)
        """
        # 确保初始感染节点是一维数组
        initial_infected_nodes = np.array(initial_infected_nodes).ravel()
        
        # 初始化感染状态
        infection_state = np.zeros(self.num_nodes)
        infection_state[initial_infected_nodes] = 1.0
        
        # 记录感染历史
        infection_history = [infection_state.copy()]
        
        for step in range(steps):
            if method == 'random_walk':
                # 基于图结构的随机游走传播
                new_infection_state = self.adj_matrix @ infection_state
            else:
                # 基于节点嵌入相似度的传播
                # 计算当前感染节点与所有节点的相似度
                infected_embeddings = self.node_embedding[initial_infected_nodes]
                similarity = np.zeros((len(initial_infected_nodes), self.num_nodes))
                
                for i, infected_idx in enumerate(initial_infected_nodes):
                    # 使用余弦相似度，确保向量维度正确
                    infected_node_embedding = self.node_embedding[infected_idx].reshape(-1)  # 确保是一维向量
                    
                    # 计算分子（点积）
                    dot_product = np.dot(self.node_embedding, infected_node_embedding)
                    
                    # 计算分母（范数乘积）
                    norm_product = np.linalg.norm(self.node_embedding, axis=1) * np.linalg.norm(infected_node_embedding)
                    
                    # 避免除以零
                    norm_product = np.maximum(norm_product, 1e-10)
                    
                    # 计算相似度
                    sim = dot_product / norm_product
                    similarity[i] = sim
                
                # 加权相似度
                similarity = np.mean(similarity, axis=0)
                
                # 基于相似度传播
                new_infection_state = infection_state * 0.5 + similarity * 0.5
            
            # 更新感染状态
            infection_state = new_infection_state
            infection_history.append(infection_state.copy())
            
            # 更新感染节点列表（选择感染概率最高的节点）
            if step < steps - 1:
                threshold = np.percentile(infection_state, 80)  # 选择前20%的节点
                new_infected_nodes = np.where(infection_state >= threshold)[0]
                
                # 确保两个数组都是一维的再连接
                initial_infected_nodes = np.unique(np.concatenate([
                    initial_infected_nodes.ravel(), 
                    new_infected_nodes.ravel()
                ]))
        
        return np.array(infection_history)

def train(model, data, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(data)
    class_pred = outputs['class_pred']
    key_node_scores = outputs['key_node_scores']
    
    # 节点分类损失
    class_loss = criterion(class_pred[data.train_mask], data.y[data.train_mask])
    
    # 关键节点预测的辅助损失（使用标签的先验知识）
    # 这里简化处理，假设训练集中前10%的攻击样本是关键传播节点
    attack_indices = torch.where(data.y[data.train_mask] == 1)[0]
    top_k = max(1, int(len(attack_indices) * 0.1))
    key_node_mask = torch.zeros_like(data.y, dtype=torch.bool)
    
    if len(attack_indices) > 0:
        train_attack_indices = torch.where(data.train_mask)[0][attack_indices]
        # 随机选择一些攻击节点作为关键节点
        key_indices = np.random.choice(train_attack_indices.cpu().numpy(), top_k, replace=False)
        key_node_mask[key_indices] = True
    
    # 关键节点预测损失
    if torch.any(key_node_mask):
        key_node_loss = F.binary_cross_entropy_with_logits(
            key_node_scores[key_node_mask].squeeze(), 
            torch.ones(key_node_scores[key_node_mask].size(0), device=device)
        )
        
        # 非关键节点损失
        non_key_node_loss = F.binary_cross_entropy_with_logits(
            key_node_scores[~key_node_mask].squeeze(), 
            torch.zeros(key_node_scores[~key_node_mask].size(0), device=device)
        )
        
        propagation_loss = key_node_loss + non_key_node_loss
    else:
        propagation_loss = torch.tensor(0.0, device=device)
    
    # 总损失
    loss = class_loss + 0.1 * propagation_loss  # 给传播损失一个较小的权重
    loss.backward()
    optimizer.step()
    
    return loss.item()

def evaluate(model, data, device):
    model.eval()
    outputs = model(data)
    class_pred = outputs['class_pred'].argmax(dim=1)  # 使用概率最高的类别作为预测结果
    
    # 计算准确率
    train_acc = class_pred[data.train_mask].eq(data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()
    val_acc = class_pred[data.val_mask].eq(data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
    test_acc = class_pred[data.test_mask].eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    
    # 计算相对传播准确率(RPA) - 使用F1分数作为近似
    rpa = f1_score(data.y[data.test_mask].cpu().numpy(), class_pred[data.test_mask].cpu().numpy())
    
    return {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'rpa': rpa
    }

def run_training(model, data, optimizer, criterion, device, epochs=201, patience=10):
    best_val_acc = 0
    best_val_loss = float('inf')
    counter = 0
    
    for epoch in range(1, epochs + 1):
        loss = train(model, data, optimizer, criterion, device)
        metrics = evaluate(model, data, device)
        
        logging.info(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {metrics["train_acc"]:.4f}, '
                     f'Val: {metrics["val_acc"]:.4f}, Test: {metrics["test_acc"]:.4f}, RPA: {metrics["rpa"]:.4f}')
        
        # 早停策略
        if metrics['val_acc'] > best_val_acc or metrics['val_acc'] == best_val_acc and loss < best_val_loss:
            best_val_acc = metrics['val_acc']
            best_val_loss = loss
            torch.save(model.state_dict(), 'best_model.pt')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                logging.info('Early stopping!')
                break
    
    return best_val_acc

def main():
    # 加载数据
    X, y, attack_cat, df = load_unsw_nb15(r'D:\code\github\kaggle_data\mrwellsdavid\unsw-nb15\versions\1\UNSW_NB15_training-set.csv')

    # 构建图数据
    data = build_graph_data(X, y, attack_cat, df, k_neighbors=5)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    # 初始化模型
    model = MultiModalGNN(
        input_dim=data.x.size(1),
        hidden_dim=64,
        output_dim=2,  # 二分类：正常/攻击
        num_layers=3,
        dropout=0.5,
        gnn_type='GAT'  # 可以选择'GCN', 'GAT'或'GraphSAGE'
    ).to(device)
    
    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 训练模型
    best_val_acc = run_training(model, data, optimizer, criterion, device)
    
    # 加载最佳模型并评估
    model.load_state_dict(torch.load('best_model.pt'))
    final_metrics = evaluate(model, data, device)
    logging.info(f'Final Test Acc: {final_metrics["test_acc"]:.4f}, RPA: {final_metrics["rpa"]:.4f}')
    
    # 关键节点分析
    model.eval()
    outputs = model(data)
    key_node_scores = torch.sigmoid(outputs['key_node_scores']).detach().cpu().numpy()
    
    # 获取Top-K关键节点
    k = 10
    top_k_nodes = np.argsort(-key_node_scores)[:k]
    
    logging.info(f'Top {k} critical nodes: {top_k_nodes}')
    logging.info(f'Their scores: {key_node_scores[top_k_nodes]}')
    
    # 威胁传播预测
    propagation_predictor = ThreatPropagationPredictor(model, data)
    
    # 选择前3个关键节点作为初始感染源
    initial_infected_nodes = top_k_nodes[:3]
    
    # 预测5步的传播过程
    infection_history = propagation_predictor.predict_propagation(
        initial_infected_nodes, 
        steps=5, 
        method='attention'  # 可以选择'random_walk'或'attention'
    )
    
    # 找出可能被感染的前10个节点
    final_step = infection_history[-1]
    potential_victims = np.argsort(-final_step)[:10]
    
    logging.info(f'Potential victims after 5 steps: {potential_victims}')
    logging.info(f'Infection probabilities: {final_step[potential_victims]}')

if __name__ == '__main__':
    main()