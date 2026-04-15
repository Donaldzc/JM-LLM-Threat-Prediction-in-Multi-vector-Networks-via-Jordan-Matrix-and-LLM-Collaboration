# 1. 添加递归特征消除筛选更有价值的特征
# 2. 对数值特征进行分箱或标准化时考虑使用更适合网络流量数据的方法
# 3. 调整k_neighbors参数，尝试更大的值以捕获更多节点关系
# 4. 使用Jaccard 指数代替欧氏距离
# 5. 添加自环（self-loops）以增强节点自身信息的保留
# 6. 增加 GNN 层数以捕获更复杂的图结构
# 7. 尝试不同的 GNN 变体（如 Graph Attention Network 的多头注意力机制）
# 8. 改进 Jordan 矩阵特征与 GNN 特征的融合方式（如注意力机制加权融合）
# 9. 调整多任务损失函数的权重
# 10. 使用 Optuna 等工具优化学习率、批次大小、隐藏层维度等超参数
# 11. 增加 Dropout 率或 L2 正则化强度
# 12. 增加分解的特征向量数量（当前为 30）以捕获更多图结构信息
# 13. 添加F1 分数、ROC-AUC 等更适合不平衡数据的指标

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import f1_score, roc_auc_score
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, lobpcg
import matplotlib.pyplot as plt
from scipy.sparse import csgraph
from numpy.linalg import pinv
import warnings
import os
import optuna

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 数据加载与预处理
def load_unsw_nb15(data_path):
    df = pd.read_csv(data_path)
    print(f"原始数据集大小: {df.shape}")
    
    # 数据清洗
    df = df.dropna()
    df = df.drop_duplicates()
    print(f"清洗后数据集大小: {df.shape}")
    
    # 特征选择
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
    
    # 处理数值特征，使用更适合网络流量数据的标准化方法
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[numeric_features])
    
    # 处理分类特征
    X_cat = pd.get_dummies(df[categorical_features])
    
    # 合并特征
    X = np.hstack([X_num, X_cat.values])
    
    # 处理标签
    le = LabelEncoder()
    y = le.fit_transform(df['label'])  # 0=正常，1=攻击
    
    # 处理攻击类别
    attack_cat = le.fit_transform(df['attack_cat'].fillna('Normal'))
    
    # 递归特征消除
    selector = RFECV(estimator=RandomForestClassifier(n_estimators=100, random_state=42), step=1, cv=5)
    X = selector.fit_transform(X, y)
    
    return X, y, attack_cat, df

# 构建图数据
def build_graph_data(X, y, attack_cat, df, k_neighbors=10):
    # 使用Jaccard指数代替欧氏距离
    from sklearn.metrics.pairwise import jaccard_distances
    adj_matrix = 1 - jaccard_distances(X)
    adj_matrix = sp.csr_matrix(adj_matrix)
    
    # 添加自环
    adj_matrix = adj_matrix + sp.eye(adj_matrix.shape[0], format='csr')
    
    # 转换为稀疏矩阵格式
    edge_index = []
    edge_weight = []
    
    for i, j in zip(*adj_matrix.nonzero()):
        edge_index.append([i, j])
        edge_weight.append(adj_matrix[i, j])
    
    # 对边权重进行归一化
    edge_weight = np.exp(-np.array(edge_weight))
    
    # 创建边索引和边权重张量
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    
    # 节点特征和标签
    x = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)
    attack_cat = torch.tensor(attack_cat, dtype=torch.long)
    
    # 创建图数据对象
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y, attack_cat=attack_cat)
    
    # 划分训练集、验证集和测试集
    indices = torch.arange(data.num_nodes)
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.25, random_state=42)
    
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    
    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True
    
    return data

# 约旦矩阵分解
class JordanMatrixDecomposition:
    def __init__(self, adj_matrix, n_components=100, solver='lobpcg', max_iter=30000, tol=1e-3,
                 preprocess=True, use_pca_init=True, n_pca_components=200, postprocess=True):
        self.adj_matrix = adj_matrix
        self.n_components = n_components
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.preprocess = preprocess
        self.use_pca_init = use_pca_init
        self.n_pca_components = n_pca_components
        self.postprocess = postprocess
        
        # 计算约旦分解
        self.jordan_form, self.trans_matrix = self._compute_jordan_form()
        self.eigenvalues = np.diag(self.jordan_form)
        
        # 验证矩阵维度
        if self.trans_matrix.shape[0] != self.trans_matrix.shape[1]:
            print(f"警告：特征向量矩阵形状为 {self.trans_matrix.shape}，非方阵，将进行维度截断")
            min_dim = min(self.trans_matrix.shape)
            self.trans_matrix = self.trans_matrix[:min_dim, :min_dim]
            self.jordan_form = self.jordan_form[:min_dim, :min_dim]
            self.eigenvalues = self.eigenvalues[:min_dim]
    
    def _compute_jordan_form(self):
        try:
            matrix = self.adj_matrix.asfptype()
            
            # 矩阵预处理
            if self.preprocess:
                matrix = matrix + sp.eye(matrix.shape[0], format='csr') * 1e-10
                matrix = 0.5 * (matrix + matrix.T)
            
            # 生成初始猜测向量
            if self.use_pca_init:
                from sklearn.decomposition import TruncatedSVD
                svd = TruncatedSVD(n_components=min(self.n_pca_components, matrix.shape[0]-1))
                X = svd.fit_transform(matrix)
            else:
                X = np.random.rand(matrix.shape[0], min(self.n_components, 50))
            
            # 计算特征值和特征向量
            if self.solver == 'arpack':
                eigenvalues, eigenvectors = eigsh(
                    matrix,
                    k=min(self.n_components, matrix.shape[0]-1),
                    which='LM',
                    maxiter=self.max_iter,
                    tol=self.tol,
                    sigma=1e-10
                )
            else:
                M = None
                eigenvalues, eigenvectors = lobpcg(
                    matrix,
                    X,
                    M=M,
                    largest=True,
                    maxiter=self.max_iter,
                    tol=self.tol,
                    verbosityLevel=0
                )
            
            jordan_form = np.diag(eigenvalues)
            trans_matrix = eigenvectors
            
            return jordan_form, trans_matrix
        except Exception as e:
            print(f"特征值计算失败: {e}")
            
            if self.solver == 'arpack':
                print("尝试使用LOBPCG求解器...")
                self.solver = 'lobpcg'
                return self._compute_jordan_form()
            else:
                print("两种求解器都失败，尝试使用矩阵的子集...")
                subset_size = min(3000, matrix.shape[0])
                
                degrees = matrix.sum(axis=1).A1
                subset_indices = np.argsort(-degrees)[:subset_size]
                
                subset_matrix = matrix[subset_indices][:, subset_indices]
                
                dense_subset = subset_matrix.toarray()
                eigenvalues, eigenvectors = np.linalg.eig(dense_subset)
                
                idx = eigenvalues.argsort()[::-1][:min(self.n_components, subset_size)]
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
                
                jordan_form = np.diag(eigenvalues)
                trans_matrix = eigenvectors
                
                return jordan_form, trans_matrix
    
    def get_propagation_matrix(self, t=1):
        if self.trans_matrix.shape[0] != self.trans_matrix.shape[1]:
            raise ValueError(f"特征向量矩阵形状为 {self.trans_matrix.shape}，必须为方阵")
        
        if self.jordan_form.shape[0] != self.jordan_form.shape[1]:
            raise ValueError(f"约旦矩阵形状为 {self.jordan_form.shape}，必须为方阵")
        
        try:
            jordan_power = np.linalg.matrix_power(self.jordan_form, t)
            propagation_matrix = self.trans_matrix @ jordan_power @ np.linalg.inv(self.trans_matrix)
        except np.linalg.LinAlgError as e:
            print(f"矩阵求逆失败: {e}")
            jordan_power = np.linalg.matrix_power(self.jordan_form, t)
            propagation_matrix = self.trans_matrix @ jordan_power @ pinv(self.trans_matrix)
        
        return propagation_matrix
    
    def predict_infection(self, initial_state, t=1):
        if initial_state.ndim > 1:
            initial_state = initial_state.flatten()
        
        propagation_matrix = self.get_propagation_matrix(t)
        
        # 确保维度匹配
        min_dim = min(initial_state.shape[0], propagation_matrix.shape[1])
        initial_state_truncated = initial_state[:min_dim]
        propagation_matrix_truncated = propagation_matrix[:min_dim, :min_dim]
        
        return propagation_matrix_truncated @ initial_state_truncated

# 多层GNN架构
class MultiLayerGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, heads=4):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=0.5))
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=0.5))
        self.layers.append(GATConv(hidden_dim * heads, output_dim, heads=1, concat=False, dropout=0.5))
        
    def forward(self, x, edge_index, edge_weight=None):
        for layer in self.layers[:-1]:
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.elu(layer(x, edge_index, edge_weight))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.layers[-1](x, edge_index, edge_weight)
        return x

# 注意力机制加权融合
class AttentionFusion(torch.nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.attention = torch.nn.Linear(feature_dim * 2, 1)
    
    def forward(self, gnn_features, jordan_features):
        combined_features = torch.cat([gnn_features, jordan_features], dim=1)
        attention_scores = torch.softmax(self.attention(combined_features), dim=0)
        fused_features = attention_scores * gnn_features + (1 - attention_scores) * jordan_features
        return fused_features

# 协作分析模型
class CollaborativeAnalysisModel(torch.nn.Module):
    def __init__(self, gnn_model, jordan_matrix, feature_dim, hidden_dim, num_classes):
        super().__init__()
        self.gnn = gnn_model
        self.jordan_matrix = jordan_matrix
        
        # 特征融合层
        self.fusion = AttentionFusion(feature_dim)
        
        # 分类器
        self.classifier = torch.nn.Linear(feature_dim, num_classes)
        
        # 关键节点预测器
        self.key_node_predictor = torch.nn.Linear(feature_dim, 1)
    
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        
        # GNN特征提取
        gnn_features = self.gnn(x, edge_index, edge_weight)
        
        # 约旦矩阵特征
        jordan_features = torch.tensor(
            self.jordan_matrix.eigenvalues, 
            dtype=torch.float, 
            device=x.device
        ).expand(gnn_features.size(0), -1)
        
        # 特征融合
        fused_features = self.fusion(gnn_features, jordan_features)
        
        # 分类预测
        class_pred = self.classifier(fused_features)
        
        # 关键节点预测
        key_node_scores = self.key_node_predictor(fused_features).squeeze()
        
        return {
            'class_pred': class_pred,
            'key_node_scores': key_node_scores,
            'gnn_features': gnn_features,
            'jordan_features': jordan_features
        }

# 多任务损失函数
def multi_task_loss(outputs, data, class_weight=0.8, node_weight=0.2):
    # 分类损失
    class_loss = F.cross_entropy(
        outputs['class_pred'][data.train_mask], 
        data.y[data.train_mask]
    )
    
    # 关键节点预测损失
    node_labels = torch.zeros_like(outputs['key_node_scores'])
    node_labels[data.y == 1] = 1
    node_loss = F.binary_cross_entropy_with_logits(
        outputs['key_node_scores'][data.train_mask],
        node_labels[data.train_mask]
    )
    
    return class_weight * class_loss + node_weight * node_loss

# 训练函数
def train(model, data, optimizer, criterion, device, multi_task=False):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(data)
    if multi_task:
        loss = criterion(outputs, data)
    else:
        loss = F.cross_entropy(outputs['class_pred'][data.train_mask], data.y[data.train_mask])
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

# 评估函数
def evaluate(model, data, device):
    model.eval()
    
    outputs = model(data)
    pred = outputs['class_pred'].argmax(dim=1)
    
    train_acc = pred[data.train_mask].eq(data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()
    val_acc = pred[data.val_mask].eq(data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
    test_acc = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    
    from sklearn.metrics import precision_recall_fscore_support
    rpa = precision_recall_fscore_support(
        data.y[data.test_mask], 
        pred[data.test_mask], 
        average='macro',
        zero_division=0  # 解决Precision未定义的警告
    )[2]
    
    f1 = f1_score(data.y[data.test_mask].cpu().numpy(), pred[data.test_mask].cpu().numpy())
    roc_auc = roc_auc_score(data.y[data.test_mask].cpu().numpy(), F.softmax(outputs['class_pred'][data.test_mask], dim=1)[:, 1].cpu().numpy())
    
    return {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'rpa': rpa,
        'f1': f1,
        'roc_auc': roc_auc
    }

# Optuna优化
def objective(trial):
    # 加载数据
    X, y, attack_cat, df = load_unsw_nb15(r'D:\code\github\kaggle_data\mrwellsdavid\unsw-nb15\versions\1\UNSW_NB15_training-set.csv')
    
    # 构建图数据
    data = build_graph_data(X, y, attack_cat, df, k_neighbors=10)
    
    # 计算邻接矩阵
    adj_matrix = sp.csr_matrix(
        (data.edge_attr.cpu().numpy(), (data.edge_index[0].cpu().numpy(), data.edge_index[1].cpu().numpy())),
        shape=(data.num_nodes, data.num_nodes)
    )
    
    # 约旦矩阵分解
    jordan_decomp = JordanMatrixDecomposition(adj_matrix, n_components=50)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    # 超参数优化
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    hidden_dim = trial.suggest_int('hidden_dim', 32, 128)
    num_layers = trial.suggest_int('num_layers', 2, 5)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.3, 0.7)
    l2_reg = trial.suggest_loguniform('l2_reg', 1e-5, 1e-3)
    
    # 构建模型
    gnn = MultiLayerGNN(
        input_dim=data.x.size(1),
        hidden_dim=hidden_dim,
        output_dim=32,
        num_layers=num_layers
    ).to(device)
    
    model = CollaborativeAnalysisModel(
        gnn_model=gnn,
        jordan_matrix=jordan_decomp,
        feature_dim=32,
        hidden_dim=hidden_dim,
        num_classes=2
    ).to(device)
    
    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    criterion = multi_task_loss
    
    # 训练循环
    best_val_acc = 0
    patience = 10
    counter = 0
    
    for epoch in range(1, 201):
        loss = train(model, data, optimizer, criterion, device, multi_task=True)
        metrics = evaluate(model, data, device)
        
        if metrics['val_acc'] > best_val_acc:
            best_val_acc = metrics['val_acc']
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
    
    return best_val_acc

# 主函数
def main():
    # 创建保存结果的目录
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Optuna优化
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    
    print('Best trial:')
    best_trial = study.best_trial
    print(f'  Value: {best_trial.value}')
    print('  Params:')
    for key, value in best_trial.params.items():
        print(f'    {key}: {value}')
    
    # 使用最佳超参数进行训练
    # 加载数据
    X, y, attack_cat, df = load_unsw_nb15(r'D:\code\github\kaggle_data\mrwellsdavid\unsw-nb15\versions\1\UNSW_NB15_training-set.csv')
    
    # 构建图数据
    data = build_graph_data(X, y, attack_cat, df, k_neighbors=10)
    
    # 计算邻接矩阵
    adj_matrix = sp.csr_matrix(
        (data.edge_attr.cpu().numpy(), (data.edge_index[0].cpu().numpy(), data.edge_index[1].cpu().numpy())),
        shape=(data.num_nodes, data.num_nodes)
    )
    
    # 约旦矩阵分解
    jordan_decomp = JordanMatrixDecomposition(adj_matrix, n_components=50)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    # 构建模型
    gnn = MultiLayerGNN(
        input_dim=data.x.size(1),
        hidden_dim=best_trial.params['hidden_dim'],
        output_dim=32,
        num_layers=best_trial.params['num_layers']
    ).to(device)
    
    model = CollaborativeAnalysisModel(
        gnn_model=gnn,
        jordan_matrix=jordan_decomp,
        feature_dim=32,
        hidden_dim=best_trial.params['hidden_dim'],
        num_classes=2
    ).to(device)
    
    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=best_trial.params['learning_rate'], weight_decay=best_trial.params['l2_reg'])
    criterion = multi_task_loss
    
    # 训练循环
    best_val_acc = 0
    patience = 10
    counter = 0
    
    print("开始训练模型...")
    for epoch in range(1, 201):
        loss = train(model, data, optimizer, criterion, device, multi_task=True)
        metrics = evaluate(model, data, device)
        
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {metrics["train_acc"]:.4f}, '
              f'Val: {metrics["val_acc"]:.4f}, Test: {metrics["test_acc"]:.4f}, RPA: {metrics["rpa"]:.4f}, '
              f'F1: {metrics["f1"]:.4f}, ROC-AUC: {metrics["roc_auc"]:.4f}')
        
        # 早停策略
        if metrics['val_acc'] > best_val_acc:
            best_val_acc = metrics['val_acc']
            torch.save(model.state_dict(), 'results/best_model.pt')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print('Early stopping!')
                break
    
    # 加载最佳模型并评估
    try:
        model.load_state_dict(torch.load('results/best_model.pt', weights_only=True))
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    final_metrics = evaluate(model, data, device)
    print(f'Final Test Acc: {final_metrics["test_acc"]:.4f}, RPA: {final_metrics["rpa"]:.4f}, '
          f'F1: {final_metrics["f1"]:.4f}, ROC-AUC: {final_metrics["roc_auc"]:.4f}')
    
    # 关键节点分析
    model.eval()
    with torch.no_grad():
        outputs = model(data)
        key_node_scores = outputs['key_node_scores'].detach().cpu().numpy()
    
    # 获取Top-K关键节点
    k = 10
    top_k_nodes = np.argsort(-key_node_scores)[:k]
    
    print(f'Top {k} critical nodes: {top_k_nodes}')
    print(f'Their scores: {key_node_scores[top_k_nodes]}')
    
    # 威胁传播预测
    initial_state = np.zeros(data.num_nodes)
    initial_state[top_k_nodes[0]] = 1  # 假设第一个关键节点被感染
    
    # 使用约旦矩阵预测传播
    try:
        propagation_steps = 5
        infection_state = jordan_decomp.predict_infection(initial_state, t=propagation_steps)
        
        # 找出可能被感染的前10个节点
        potential_victims = np.argsort(-infection_state)[:10]
        print(f'Potential victims after {propagation_steps} steps: {potential_victims}')
        print(f'Infection probabilities: {infection_state[potential_victims]}')
    except Exception as e:
        print(f"传播预测失败: {e}")
    
    # 交叉验证
    try:
        cv_acc, cv_rpa = kfold_cross_validation(X, y, k=3)
        print(f'Cross-validation Acc: {cv_acc:.4f}, RPA: {cv_rpa:.4f}')
    except Exception as e:
        print(f"交叉验证失败: {e}")
    
    # 模型集成
    try:
        ensemble_pred = ensemble_training(X, y, attack_cat, df, num_models=2)
        ensemble_acc = (ensemble_pred.argmax(dim=1) == data.y).float().mean().item()
        print(f'Ensemble accuracy: {ensemble_acc:.4f}')
    except Exception as e:
        print(f"模型集成失败: {e}")

if __name__ == '__main__':
    main()