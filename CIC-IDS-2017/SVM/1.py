import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_fscore_support
import matplotlib.pyplot as plt

# 加载CIC - IDS - 2017数据集
def load_cic_ids_2017(data_path):
    # 读取CSV文件
    df = pd.read_csv(data_path)
    print(f"原始数据集大小: {df.shape}")

    # 数据清洗
    df = df.dropna()  # 移除缺失值
    df = df.drop_duplicates()  # 移除重复记录
    print(f"清洗后数据集大小: {df.shape}")

    # 特征选择（根据CIC-IDS-2017数据集调整）
    numeric_features = [
        ' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets',
        'Total Length of Fwd Packets', ' Total Length of Bwd Packets',
        ' Fwd Packet Length Mean', ' Bwd Packet Length Mean',
        'Flow Bytes/s', ' Flow Packets/s', ' Flow IAT Mean',
        ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min',
        'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std',
        ' Fwd IAT Max', ' Fwd IAT Min', 'Bwd IAT Total',
        ' Bwd IAT Mean', ' Bwd IAT Std', ' Bwd IAT Max',
        ' Bwd IAT Min', 'Fwd PSH Flags', ' Bwd PSH Flags',
        ' Fwd URG Flags', ' Bwd URG Flags', ' Fwd Header Length',
        ' Bwd Header Length', 'Fwd Packets/s', ' Bwd Packets/s',
        ' Min Packet Length', ' Max Packet Length', ' Packet Length Mean',
        ' Packet Length Std', ' Packet Length Variance',
        'FIN Flag Count', ' SYN Flag Count', ' RST Flag Count',
        ' PSH Flag Count', ' ACK Flag Count', ' URG Flag Count',
        ' CWE Flag Count', ' ECE Flag Count', ' Down/Up Ratio',
        ' Average Packet Size', ' Avg Fwd Segment Size',
        ' Avg Bwd Segment Size', ' Fwd Header Length.1',
        'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk',
        ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk',
        ' Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
        'Subflow Fwd Packets', ' Subflow Fwd Bytes',
        ' Subflow Bwd Packets', ' Subflow Bwd Bytes',
        'Init_Win_bytes_forward', ' Init_Win_bytes_backward',
        ' act_data_pkt_fwd', ' min_seg_size_forward',
        'Active Mean', ' Active Std', ' Active Max',
        ' Active Min', 'Idle Mean', ' Idle Std',
        ' Idle Max', ' Idle Min'
    ]

    categorical_features = []

    # 处理数值特征
    # 检查并处理无穷大值和过大的值
    df[numeric_features] = df[numeric_features].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=numeric_features)  # 移除包含NaN的行


    # 特征选择
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 根据实际列名修改标签列名（假设实际列名为'Label'）
    label_column = 'Label'  
    
    # 确保标签列不包含在特征列中
    if label_column in numeric_features:
        numeric_features.remove(label_column)

    categorical_features = []

    # 处理数值特征
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[numeric_features])

    # 处理分类特征
    X_cat = np.array([]).reshape(df.shape[0], 0)
    if categorical_features:
        X_cat = pd.get_dummies(df[categorical_features])

    # 合并特征
    X = np.hstack([X_num, X_cat])

    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[numeric_features])

    # 处理分类特征
    if categorical_features:
        X_cat = pd.get_dummies(df[categorical_features])
    else:
        X_cat = np.empty((X_num.shape[0], 0))

    # 合并特征
    X = np.hstack([X_num, X_cat])

    # 处理标签
    le = LabelEncoder()
    y = le.fit_transform(df[' Label'])  # 注意列名中的空格

    return X, y, df

# 构建图数据
def build_graph_data(X, y, k_neighbors=3):
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

class SVMBasedModel:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        # 初始化SVM模型
        self.svm = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=True,  # 启用概率估计
            random_state=42
        )
        self.is_trained = False

    def fit(self, X, y):
        # 训练SVM模型
        print(f"Training SVM with {X.shape[0]} samples and {X.shape[1]} features")
        self.svm.fit(X, y)
        self.is_trained = True

    def predict(self, X):
        # 预测类别
        if not self.is_trained:
            raise Exception("Model must be trained before prediction")
        return self.svm.predict(X)

    def predict_proba(self, X):
        # 预测类别概率
        if not self.is_trained:
            raise Exception("Model must be trained before prediction")
        return self.svm.predict_proba(X)

    def decision_function(self, X):
        # 决策函数值
        if not self.is_trained:
            raise Exception("Model must be trained before prediction")
        return self.svm.decision_function(X)

class CollaborativeAnalysisModel:
    def __init__(self, feature_dim, hidden_dim, num_classes):
        super().__init__()
        # 初始化SVM模型
        self.svm = SVMBasedModel(kernel='rbf', C=10, gamma='auto')

        # 特征变换层（可选）
        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.num_classes = num_classes
        self.is_trained = False

    def train(self, data):
        # 准备训练数据
        X_train = data.x[data.train_mask].cpu().numpy()
        y_train = data.y[data.train_mask].cpu().numpy()

        # 应用特征变换（可选）
        X_train_transformed = self.feature_transform(torch.tensor(X_train, dtype=torch.float)).detach().numpy()

        # 训练SVM
        self.svm.fit(X_train_transformed, y_train)
        self.is_trained = True

    def forward(self, data):
        if not self.is_trained:
            raise Exception("Model must be trained before inference")

        # 获取所有节点的特征
        X = data.x.cpu().numpy()

        # 应用特征变换
        X_transformed = self.feature_transform(torch.tensor(X, dtype=torch.float)).detach().numpy()

        # 预测类别概率
        class_probs = self.svm.predict_proba(X_transformed)

        # 决策函数值作为关键节点分数
        key_node_scores = self.svm.decision_function(X_transformed)

        # 转换为张量格式
        class_pred = torch.tensor(class_probs, dtype=torch.float)
        key_node_scores = torch.tensor(key_node_scores, dtype=torch.float)

        return {
            'class_pred': class_pred,
            'key_node_scores': key_node_scores,
            'svm_features': torch.tensor(X_transformed, dtype=torch.float)
        }

# 训练函数
def train(model, data):
    model.train(data)
    return 0.0  # SVM训练不需要返回损失值

# 评估函数
def evaluate(model, data):
    model_output = model.forward(data)

    # 获取预测结果
    pred = model_output['class_pred'].argmax(dim=1).cpu().numpy()
    true_labels = data.y.cpu().numpy()

    # 计算准确率
    train_acc = (pred[data.train_mask.cpu().numpy()] == true_labels[data.train_mask.cpu().numpy()]).mean()
    val_acc = (pred[data.val_mask.cpu().numpy()] == true_labels[data.val_mask.cpu().numpy()]).mean()
    test_acc = (pred[data.test_mask.cpu().numpy()] == true_labels[data.test_mask.cpu().numpy()]).mean()

    # 计算精确率、召回率和F1分数
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels[data.test_mask.cpu().numpy()],
        pred[data.test_mask.cpu().numpy()],
        average='binary'
    )

    # 计算混淆矩阵
    cm = confusion_matrix(true_labels[data.test_mask.cpu().numpy()], pred[data.test_mask.cpu().numpy()])

    # 计算ROC曲线和AUC值
    y_score = model_output['class_pred'][data.test_mask.cpu().numpy(), 1].cpu().numpy()
    fpr, tpr, _ = roc_curve(true_labels[data.test_mask.cpu().numpy()], y_score)
    roc_auc = auc(fpr, tpr)

    return {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'roc_auc': roc_auc
    }

# 主函数
def main():
    X, y, df = load_cic_ids_2017(r'D:\code\data\label_studio_data\TrafficLabelling_\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')

    data = build_graph_data(X, y, k_neighbors=3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    model = CollaborativeAnalysisModel(
        feature_dim=data.x.size(1),
        hidden_dim=64,
        num_classes=2  # 二分类：正常/攻击
    )

    # 训练模型
    train(model, data)

    # 评估模型
    metrics = evaluate(model, data)

    print(f'Train Acc: {metrics["train_acc"]:.4f}')
    print(f'Val Acc: {metrics["val_acc"]:.4f}')
    print(f'Test Acc: {metrics["test_acc"]:.4f}')
    print(f'Precision: {metrics["precision"]:.4f}')
    print(f'Recall: {metrics["recall"]:.4f}')
    print(f'F1 Score: {metrics["f1"]:.4f}')
    print(f'ROC AUC: {metrics["roc_auc"]:.4f}')
    print(f'Confusion Matrix:\n{metrics["confusion_matrix"]}')

    # 关键节点分析
    model_output = model.forward(data)
    key_node_scores = model_output['key_node_scores'].detach().cpu().numpy()

    # 获取Top - K关键节点
    k = 10
    top_k_nodes = np.argsort(-np.abs(key_node_scores))[:k]  # 使用绝对值，因为SVM决策函数可正可负

    print(f'Top {k} critical nodes: {top_k_nodes}')
    print(f'Their scores: {key_node_scores[top_k_nodes]}') 


if __name__ == "__main__":
    main()