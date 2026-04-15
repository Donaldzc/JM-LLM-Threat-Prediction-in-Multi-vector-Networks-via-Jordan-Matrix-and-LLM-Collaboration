import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, jaccard_score
import time
from tqdm import tqdm

# 模拟数据生成函数（实际应用中替换为真实数据加载）
def generate_simulated_data(n_users=1000, n_nodes=500, n_sets_per_user=10):
    """生成模拟网络拓扑与威胁传播数据"""
    # 生成邻接矩阵（模拟网络拓扑）
    adj = np.random.rand(n_nodes, n_nodes)
    adj = (adj > 0.1) * adj  # 稀疏化处理
    adj = adj / np.sum(adj, axis=1, keepdims=True)  # 行归一化
    
    # 生成用户-节点交互数据
    user_data = []
    for user in range(n_users):
        user_sets = []
        current_infected = np.random.choice(n_nodes, 1)
        for t in range(n_sets_per_user):
            # 模拟威胁传播
            new_infected = np.where(adj[current_infected].sum(axis=0) > 0.5)[0]
            current_infected = np.unique(np.concatenate([current_infected, new_infected]))
            user_sets.append(current_infected)
        user_data.append(user_sets)
    
    return adj, user_data

# 约旦矩阵模型实现
class JordanMatrixModel:
    def __init__(self, adj_matrix):
        self.adj_matrix = adj_matrix
        self.n_nodes = adj_matrix.shape[0]
        
    def decompose(self):
        """对邻接矩阵进行约旦分解"""
        try:
            # 使用scipy进行约旦分解（实际应用中可能需要更稳定的算法）
            from scipy.linalg import jordan_form
            J, P = jordan_form(self.adj_matrix)
            return J, P
        except:
            # 处理非对角化矩阵的近似分解
            eigenvalues, eigenvectors = np.linalg.eig(self.adj_matrix)
            return np.diag(eigenvalues), eigenvectors
    
    def predict_spread(self, initial_infected, steps=3):
        """预测威胁传播范围"""
        J, P = self.decompose()
        # 初始状态向量
        x0 = np.zeros(self.n_nodes)
        x0[initial_infected] = 1
        
        # 约旦矩阵幂运算模拟传播
        x = np.zeros((steps+1, self.n_nodes))
        x[0] = x0
        
        for t in range(1, steps+1):
            # 通过约旦分解计算A^t
            J_power = np.linalg.matrix_power(J, t)
            A_power = P @ J_power @ np.linalg.inv(P)
            x[t] = x0 @ A_power
        
        # 返回各时间步的感染节点（阈值0.5）
        infected_sets = [np.where(x[t] > 0.5)[0] for t in range(steps+1)]
        return infected_sets

# 大模型组件（基于Transformer的特征提取）
class TransformerFeatureExtractor(nn.Module):
    def __init__(self, node_emb_dim=64, n_heads=4, n_layers=2):
        super(TransformerFeatureExtractor, self).__init__()
        self.node_emb = nn.Embedding(1000, node_emb_dim)  # 节点嵌入层
        self.pos_emb = nn.Embedding(100, node_emb_dim)    # 位置嵌入层
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=node_emb_dim, nhead=n_heads, dim_feedforward=256
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        
    def forward(self, node_seq):
        """
        node_seq: [batch_size, seq_length] 节点序列
        """
        batch_size, seq_length = node_seq.shape
        pos = torch.arange(seq_length, device=node_seq.device).unsqueeze(0).repeat(batch_size, 1)
        
        # 嵌入层
        node_emb = self.node_emb(node_seq)
        pos_emb = self.pos_emb(pos)
        x = node_emb + pos_emb
        
        # Transformer编码
        x = self.transformer_encoder(x)
        return x  # [batch_size, seq_length, emb_dim]

# 图卷积网络组件（元素关系学习）
class GraphConvModule(nn.Module):
    def __init__(self, in_dim=64, out_dim=32):
        super(GraphConvModule, self).__init__()
        self.gcn1 = GCNConv(in_dim, 64)
        self.gcn2 = GCNConv(64, out_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        x: 节点特征 [n_nodes, in_dim]
        edge_index: 边索引 [2, n_edges]
        edge_weight: 边权重 [n_edges]
        """
        x = self.relu(self.gcn1(x, edge_index, edge_weight))
        x = self.gcn2(x, edge_index, edge_weight)
        return x

# 动态特征增强模块（注意力机制）
class AttentionEnhancement(nn.Module):
    def __init__(self, feature_dim=32, jordan_dim=10):
        super(AttentionEnhancement, self).__init__()
        self.jordan_proj = nn.Linear(jordan_dim, feature_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=4, batch_first=True
        )
        self.gate = nn.Sequential(
            nn.Linear(feature_dim*2, feature_dim),
            nn.Sigmoid()
        )
        
    def forward(self, feature_seq, jordan_features):
        """
        feature_seq: [batch_size, seq_length, feature_dim]
        jordan_features: [batch_size, jordan_dim] 约旦矩阵特征值
        """
        batch_size = feature_seq.shape[0]
        # 约旦特征投影到特征空间
        jordan_proj = self.jordan_proj(jordan_features).unsqueeze(1)  # [batch, 1, feature_dim]
        
        # 计算注意力权重
        attn_output, _ = self.attention(jordan_proj, feature_seq, feature_seq)
        attn_output = attn_output.squeeze(1)  # [batch, feature_dim]
        
        # 门控机制融合特征
        gate = self.gate(torch.cat([feature_seq[:, -1], attn_output], dim=1))
        enhanced_feature = gate * feature_seq[:, -1] + (1 - gate) * attn_output
        return enhanced_feature

# 协同决策引擎（闭环优化）
class CollaborativeDecisionEngine:
    def __init__(self, jordan_model, ml_model, graph_conv, attention_module):
        self.jordan_model = jordan_model
        self.ml_model = ml_model
        self.graph_conv = graph_conv
        self.attention_module = attention_module
        self.lr = 0.001
        self.optimizer = torch.optim.Adam(
            list(ml_model.parameters()) + 
            list(graph_conv.parameters()) + 
            list(attention_module.parameters()),
            lr=self.lr
        )
        
    def compute_errors(self, jordan_pred, ml_pred, ground_truth):
        """计算三维误差"""
        # 传播范围误差 (RPA)
        jaccard = jaccard_score(
            np.zeros(self.jordan_model.n_nodes), 
            np.zeros(self.jordan_model.n_nodes),
            labels=ground_truth,
            sample_weight=None
        )
        range_err = 1 - jaccard
        
        # 关键节点误差 (F1-CNI)
        jordan_key_nodes = self._identify_key_nodes(jordan_pred[-1])
        ml_key_nodes = self._identify_key_nodes(ml_pred)
        gt_key_nodes = self._identify_key_nodes(ground_truth)
        
        precision = len(set(jordan_key_nodes) & set(gt_key_nodes)) / max(1, len(jordan_key_nodes))
        recall = len(set(jordan_key_nodes) & set(gt_key_nodes)) / max(1, len(gt_key_nodes))
        node_err = 1 - 2 * precision * recall / max(1, precision + recall)
        
        # 动力学误差
        jordan_eigenvalues = np.abs(np.linalg.eigvals(self.jordan_model.adj_matrix))
        ml_eigenvalues = self._infer_eigenvalues(ml_pred)
        dynamic_err = np.mean(np.abs(jordan_eigenvalues - ml_eigenvalues) / (jordan_eigenvalues + 1e-8))
        
        return range_err, node_err, dynamic_err
    
    def _identify_key_nodes(self, infected_nodes, top_k=20):
        """识别关键节点（简化版）"""
        return infected_nodes[:top_k] if len(infected_nodes) > top_k else infected_nodes
    
    def _infer_eigenvalues(self, ml_pred):
        """从大模型预测中反演特征值（简化版）"""
        # 实际应用中需通过传播模式反推特征值
        return np.random.rand(5)  # 模拟反演结果
    
    def close_loop_optimization(self, initial_infected, ground_truth, max_iter=100, threshold=0.15):
        """闭环优化迭代"""
        jordan_pred = self.jordan_model.predict_spread(initial_infected)
        total_err = 1.0
        iter_count = 0
        
        while total_err > threshold and iter_count < max_iter:
            # 1. 大模型预测
            ml_input = self._prepare_ml_input(jordan_pred)
            ml_pred = self.ml_model(ml_input)
            
            # 2. 计算误差
            range_err, node_err, dynamic_err = self.compute_errors(jordan_pred, ml_pred, ground_truth)
            total_err = 0.5 * range_err + 0.3 * node_err + 0.2 * dynamic_err
            iter_count += 1
            
            # 3. 参数更新
            self.optimizer.zero_grad()
            loss = total_err
            loss.backward()
            self.optimizer.step()
            
            # 4. 约旦矩阵参数微调（简化版）
            self._fine_tune_jordan_matrix(range_err)
            
            if iter_count % 10 == 0:
                print(f"Iter {iter_count}, Total Error: {total_err:.4f}")
        
        return total_err, iter_count
    
    def _prepare_ml_input(self, jordan_pred):
        """准备大模型输入数据"""
        # 简化版：将约旦预测结果转换为节点序列
        return torch.tensor(jordan_pred[-1], dtype=torch.long).unsqueeze(0)  # [1, seq_length]
    
    def _fine_tune_jordan_matrix(self, error):
        """微调约旦矩阵参数（简化版）"""
        # 实际应用中通过梯度更新邻接矩阵
        pass

# 消融实验模型（无ERL）
class DNNTSP_NoERL(nn.Module):
    def __init__(self, node_emb_dim=64):
        super(DNNTSP_NoERL, self).__init__()
        self.node_emb = nn.Embedding(1000, node_emb_dim)
        self.lstm = nn.LSTM(node_emb_dim, 32, batch_first=True)
        
    def forward(self, node_seq):
        x = self.node_emb(node_seq)
        _, (h_n, _) = self.lstm(x)
        return h_n.squeeze(0)

# 消融实验模型（无TDL）
class DNNTSP_NoTDL(nn.Module):
    def __init__(self, node_emb_dim=64, graph_conv_dim=32):
        super(DNNTSP_NoTDL, self).__init__()
        self.node_emb = nn.Embedding(1000, node_emb_dim)
        self.graph_conv = GCNConv(node_emb_dim, graph_conv_dim)
        
    def forward(self, node_seq, edge_index, edge_weight):
        x = self.node_emb(node_seq)
        x = self.graph_conv(x, edge_index, edge_weight)
        return x.mean(dim=0)  # 平均池化替代注意力

# 评估指标计算
def calculate_rpa(predicted, ground_truth):
    """计算传播范围预测准确率"""
    tp = len(set(predicted) & set(ground_truth))
    fp = len(set(predicted) - set(ground_truth))
    fn = len(set(ground_truth) - set(predicted))
    return tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

def calculate_f1_cni(predicted_keys, ground_truth_keys):
    """计算关键节点识别F1值"""
    tp = len(set(predicted_keys) & set(ground_truth_keys))
    precision = tp / max(1, len(predicted_keys))
    recall = tp / max(1, len(ground_truth_keys))
    return 2 * precision * recall / max(1, precision + recall)

# 实验主流程
def run_experiment():
    # 1. 生成模拟数据
    adj_matrix, user_data = generate_simulated_data()
    jordan_model = JordanMatrixModel(adj_matrix)
    
    # 2. 初始化模型组件
    ml_model = TransformerFeatureExtractor()
    graph_conv = GraphConvModule()
    attention_module = AttentionEnhancement()
    decision_engine = CollaborativeDecisionEngine(
        jordan_model, ml_model, graph_conv, attention_module
    )
    
    # 3. 运行协同模型
    print("=== 运行协同模型 ===")
    initial_infected = user_data[0][0]
    ground_truth = user_data[0][-1]
    final_error, iter_count = decision_engine.close_loop_optimization(
        initial_infected, ground_truth
    )
    print(f"协同模型最终误差: {final_error:.4f}, 迭代次数: {iter_count}")
    
    # 4. 运行消融实验
    print("\n=== 运行消融实验 ===")
    # 无ERL
    print("--- 无元素关系学习 (DNNTSP w/o ERL) ---")
    no_erl_model = DNNTSP_NoERL()
    erl_input = torch.tensor(initial_infected, dtype=torch.long).unsqueeze(0)
    erl_pred = no_erl_model(erl_input).detach().numpy()
    erl_rpa = calculate_rpa(erl_pred, ground_truth)
    print(f"RPA: {erl_rpa:.4f}")
    
    # 无TDL
    print("--- 无时序依赖学习 (DNNTSP w/o TDL) ---")
    # 准备图数据
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0])
    no_tdl_model = DNNTSP_NoTDL()
    tdl_input = torch.tensor(initial_infected, dtype=torch.long)
    tdl_pred = no_tdl_model(tdl_input, edge_index, edge_weight).detach().numpy()
    tdl_rpa = calculate_rpa(tdl_pred, ground_truth)
    print(f"RPA: {tdl_rpa:.4f}")
    
    # 5. 对比方法
    print("\n=== 对比方法 ===")
    # 传统约旦矩阵模型
    jordan_pred = jordan_model.predict_spread(initial_infected)[-1]
    jm_rpa = calculate_rpa(jordan_pred, ground_truth)
    print(f"传统约旦矩阵模型 RPA: {jm_rpa:.4f}")
    
    # 单一大模型
    ml_pred = ml_model(erl_input).detach().numpy()
    lm_rpa = calculate_rpa(ml_pred, ground_truth)
    print(f"单一大模型 RPA: {lm_rpa:.4f}")
    
    # 6. 可视化实验结果（模拟图表）
    visualize_results(
        [final_error, erl_rpa, tdl_rpa, jm_rpa, lm_rpa],
        ["协同模型", "无ERL", "无TDL", "JM", "LM"]
    )

# 可视化函数
def visualize_results(metrics, labels):
    """可视化实验结果（模拟文章中的图表）"""
    x = np.arange(len(labels))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x, metrics, width)
    plt.xlabel('模型类型')
    plt.ylabel('RPA 指标')
    plt.title('不同模型的传播范围预测准确率对比')
    plt.xticks(x, labels)
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()
    
    # 模拟消融实验图表
    plt.figure(figsize=(10, 6))
    k_values = [10, 20, 30, 40]
    erl_metrics = [0.45, 0.52, 0.57, 0.61]  # 模拟数据
    tdl_metrics = [0.48, 0.55, 0.60, 0.64]
    full_metrics = [0.58, 0.65, 0.70, 0.73]
    
    plt.plot(k_values, erl_metrics, 'o-', label='无ERL')
    plt.plot(k_values, tdl_metrics, 's-', label='无TDL')
    plt.plot(k_values, full_metrics, '^-', label='完整模型')
    plt.xlabel('Top-K 值')
    plt.ylabel('RPA 指标')
    plt.title('消融实验：不同Top-K下的模型性能')
    plt.legend()
    plt.grid(True)
    plt.savefig('ablation_study.png')
    plt.show()

# 主函数
if __name__ == "__main__":
    run_experiment()