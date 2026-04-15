import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp
from sklearn.feature_selection import SelectKBest, f_classif

# 设置随机种子以确保结果可复现
np.random.seed(42)
torch.manual_seed(42)

# 1. 数据加载与预处理
def load_unsw_nb15(data_path):
    """加载并预处理UNSW-NB15数据集"""
    # 读取数据
    df = pd.read_csv(data_path)
    print(f"原始数据集大小: {df.shape}")
    
    # 数据清洗
    df = df.dropna()  # 移除缺失值
    df = df.drop_duplicates()  # 移除重复记录
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
    
    # 处理攻击类别（用于多分类）
    attack_cat = le.fit_transform(df['attack_cat'].fillna('Normal'))
    
    return X, y, attack_cat, df, X_num, X_cat, numeric_features, X_cat.columns

# 2. 特征重要性分析
def feature_importance_analysis(X, y, numeric_features, cat_features, top_k=10):
    """使用随机森林分析特征重要性"""
    # 合并特征名称
    all_features = numeric_features + list(cat_features)
    
    # 训练随机森林模型获取特征重要性
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # 获取特征重要性
    importances = rf.feature_importances_
    indices = np.argsort(importances)[-top_k:]
    
    # 绘制特征重要性图
    plt.figure(figsize=(12, 8))
    plt.title('Top {} Feature Importances'.format(top_k))
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [all_features[i] for i in indices])
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    return {all_features[i]: importances[i] for i in indices}

# 3. 传统机器学习模型
def train_traditional_models(X, y):
    """训练并评估传统机器学习模型"""
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # 评估
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        if y_prob is not None:
            roc_auc = roc_auc_score(y_test, y_prob)
        else:
            roc_auc = "N/A"
        
        results[name] = {
            "accuracy": accuracy,
            "report": report,
            "conf_matrix": conf_matrix,
            "roc_auc": roc_auc
        }
        
        print(f"{name} Accuracy: {accuracy:.4f}")
        if y_prob is not None:
            print(f"{name} ROC AUC: {roc_auc:.4f}")
        print(f"{name} Classification Report:\n{report}")
    
    return results

# 4. 构建图数据
def build_graph_data(X, y, k_neighbors=5):
    """使用KNN构建图数据"""
    from sklearn.neighbors import kneighbors_graph
    
    # 使用KNN算法构建图连接
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

# 5. 图神经网络模型
class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5, gnn_type='GCN'):
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

# 6. 训练GNN模型
def train_gnn(data, gnn_type='GCN', epochs=100, lr=0.001, weight_decay=5e-4):
    """训练图神经网络模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    model = GNNModel(
        input_dim=data.x.size(1),
        hidden_dim=64,
        output_dim=2,  # 二分类：正常/攻击
        num_layers=3,
        gnn_type=gnn_type
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_val_acc = 0
    patience = 10
    counter = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        model.eval()
        pred = out.argmax(dim=1)  # 使用最大概率作为预测结果
        train_acc = pred[data.train_mask].eq(data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()
        val_acc = pred[data.val_mask].eq(data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
        test_acc = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_{gnn_type}_model.pt')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
        
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
                  f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    
    # 加载最佳模型并评估
    model.load_state_dict(torch.load(f'best_{gnn_type}_model.pt'))
    model.eval()
    out = model(data.x, data.edge_index, data.edge_attr)
    pred = out.argmax(dim=1)
    
    test_acc = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    print(f'Final {gnn_type} Test Accuracy: {test_acc:.4f}')
    
    # 计算详细分类指标
    y_true = data.y[data.test_mask].cpu().numpy()
    y_pred = pred[data.test_mask].cpu().numpy()
    
    report = classification_report(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    return {
        "accuracy": test_acc,
        "report": report,
        "conf_matrix": conf_matrix,
        "model": model
    }

# 7. 模型比较与可视化
def compare_models(traditional_results, gnn_results):
    """比较不同模型的性能"""
    # 提取准确率
    accuracies = {}
    for name, result in traditional_results.items():
        accuracies[name] = result["accuracy"]
    
    for gnn_type, result in gnn_results.items():
        accuracies[f"{gnn_type} GNN"] = result["accuracy"]
    
    # 绘制准确率比较图
    plt.figure(figsize=(12, 6))
    plt.bar(accuracies.keys(), accuracies.values())
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.ylim(0.9, 1.0)  # 设置y轴范围以更好地显示差异
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()
    
    # 打印比较结果
    print("\n模型性能比较:")
    for name, acc in accuracies.items():
        print(f"{name}: {acc:.4f}")
    
    return accuracies

# 8. 主函数
def main():
    # 数据加载与预处理
    data_path = r'D:\code\github\kaggle_data\mrwellsdavid\unsw-nb15\versions\1\UNSW_NB15_training-set.csv'
    X, y, attack_cat, df, X_num, X_cat, numeric_features, cat_features = load_unsw_nb15(data_path)
    
    # 特征重要性分析
    print("\n进行特征重要性分析...")
    top_features = feature_importance_analysis(X, y, numeric_features, cat_features)
    print("Top特征重要性:")
    for feature, importance in sorted(top_features.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {importance:.4f}")
    
    # 传统机器学习模型
    print("\n训练传统机器学习模型...")
    traditional_results = train_traditional_models(X, y)
    
    # 构建图数据
    print("\n构建图数据...")
    graph_data = build_graph_data(X, y, k_neighbors=5)
    
    # 训练GNN模型
    print("\n训练图神经网络模型...")
    gnn_types = ['GCN', 'GAT']  # 可以添加GraphSAGE等其他GNN类型
    gnn_results = {}
    
    for gnn_type in gnn_types:
        print(f"\nTraining {gnn_type} GNN...")
        gnn_results[gnn_type] = train_gnn(graph_data, gnn_type=gnn_type, epochs=50)
    
    # 模型比较
    print("\n比较所有模型性能...")
    compare_models(traditional_results, gnn_results)
    
    # 保存结果
    with open('results.txt', 'w') as f:
        f.write("实验结果汇总:\n\n")
        
        f.write("特征重要性:\n")
        for feature, importance in sorted(top_features.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{feature}: {importance:.4f}\n")
        
        f.write("\n传统模型性能:\n")
        for name, result in traditional_results.items():
            f.write(f"\n{name}:\n")
            f.write(f"Accuracy: {result['accuracy']:.4f}\n")
            if result['roc_auc'] != "N/A":
                f.write(f"ROC AUC: {result['roc_auc']:.4f}\n")
            f.write(f"Classification Report:\n{result['report']}\n")
        
        f.write("\nGNN模型性能:\n")
        for gnn_type, result in gnn_results.items():
            f.write(f"\n{gnn_type} GNN:\n")
            f.write(f"Accuracy: {result['accuracy']:.4f}\n")
            f.write(f"Classification Report:\n{result['report']}\n")
    
    print("\n实验完成！结果已保存到'results.txt'")

if __name__ == "__main__":
    main()    