import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, lobpcg
import matplotlib.pyplot as plt
from scipy.sparse import csgraph

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
    
    # 处理攻击类别
    attack_cat = le.fit_transform(df['attack_cat'].fillna('Normal'))
    
    return X, y, attack_cat, df

# 构建图数据
def build_graph_data(X, y, attack_cat, df, k_neighbors=3):
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
    attack_cat = torch.tensor(attack_cat, dtype=torch.long)
    
    # 创建图数据对象
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y, attack_cat=attack_cat)
    
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

class JordanMatrixDecomposition:
    def __init__(self, adj_matrix, n_components=50, solver='lobpcg', max_iter=20000, tol=1e-2, 
                 preprocess=True, use_pca_init=False, n_pca_components=100):
        """
        初始化约旦矩阵分解类
        
        参数:
            adj_matrix: 网络邻接矩阵 (scipy稀疏矩阵)
            n_components: 要计算的特征向量和特征值数量
            solver: 特征值求解器 ('arpack' 或 'lobpcg')
            max_iter: 最大迭代次数
            tol: 收敛容差
            preprocess: 是否对矩阵进行预处理
            use_pca_init: 是否使用PCA生成初始向量
            n_pca_components: PCA组件数量
        """
        self.adj_matrix = adj_matrix
        self.n_components = n_components
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.preprocess = preprocess
        self.use_pca_init = use_pca_init
        self.n_pca_components = n_pca_components
        self.jordan_form, self.trans_matrix = self._compute_jordan_form()
        self.eigenvalues = np.diag(self.jordan_form)
    
    def _compute_jordan_form(self):
        """计算约旦标准型的近似"""
        try:
            matrix = self.adj_matrix.asfptype()  # 转换为浮点型稀疏矩阵
            
            # 矩阵预处理
            if self.preprocess:
                # 添加小的对角扰动以改善数值稳定性
                matrix = matrix + sp.eye(matrix.shape[0], format='csr') * 1e-10
                
                # 对称化矩阵
                matrix = 0.5 * (matrix + matrix.T)
                
                # 计算图拉普拉斯矩阵（如果需要）
                # matrix = csgraph.laplacian(matrix, normed=True)
            
            # 生成初始猜测向量
            if self.use_pca_init:
                # 使用PCA生成更有信息的初始向量
                print("使用PCA生成初始向量...")
                from sklearn.decomposition import TruncatedSVD
                svd = TruncatedSVD(n_components=min(self.n_pca_components, matrix.shape[0]-1))
                X = svd.fit_transform(matrix)
            else:
                # 随机初始化
                X = np.random.rand(matrix.shape[0], min(self.n_components, 50))
            
            # 计算特征值和特征向量
            if self.solver == 'arpack':
                # 使用ARPACK求解器
                eigenvalues, eigenvectors = eigsh(
                    matrix,
                    k=min(self.n_components, matrix.shape[0]-1),
                    which='LM',  # 计算最大的特征值
                    maxiter=self.max_iter,
                    tol=self.tol,
                    sigma=1e-10  # 避免零特征值问题
                )
            else:
                # 使用LOBPCG求解器
                # 创建预条件器（如果需要）
                M = None  # 可以在这里添加预条件器
                
                # 计算特征值和特征向量
                eigenvalues, eigenvectors = lobpcg(
                    matrix,
                    X,
                    M=M,
                    largest=True,
                    maxiter=self.max_iter,
                    tol=self.tol,
                    verbosityLevel=0  # 减少输出
                )
            
            # 构建约旦标准型（对角矩阵）
            jordan_form = np.diag(eigenvalues)
            trans_matrix = eigenvectors
            
            return jordan_form, trans_matrix
        except Exception as e:
            print(f"特征值计算失败: {e}")
            
            # 尝试不同的求解器作为回退方案
            if self.solver == 'arpack':
                print("尝试使用LOBPCG求解器...")
                self.solver = 'lobpcg'
                return self._compute_jordan_form()
            else:
                # 如果两种求解器都失败，尝试使用矩阵的一部分
                print("两种求解器都失败，尝试使用矩阵的子集...")
                subset_size = min(3000, matrix.shape[0])  # 进一步增加子集大小
                
                # 选择具有最高度数的节点
                degrees = matrix.sum(axis=1).A1
                subset_indices = np.argsort(-degrees)[:subset_size]
                
                subset_matrix = matrix[subset_indices][:, subset_indices]
                
                # 对矩阵子集计算特征值
                dense_subset = subset_matrix.toarray()
                eigenvalues, eigenvectors = np.linalg.eig(dense_subset)
                
                # 只保留最大的k个特征值
                idx = eigenvalues.argsort()[::-1][:min(self.n_components, subset_size)]
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
                
                jordan_form = np.diag(eigenvalues)
                trans_matrix = eigenvectors
                
                return jordan_form, trans_matrix
    
    def get_propagation_matrix(self, t=1):
        """
        计算t步后的传播矩阵
        
        参数:
            t: 时间步数
        
        返回:
            传播矩阵 (numpy array)
        """
        # 使用约旦分解计算矩阵指数
        jordan_power = np.linalg.matrix_power(self.jordan_form, t)
        propagation_matrix = self.trans_matrix @ jordan_power @ np.linalg.inv(self.trans_matrix)
        
        return propagation_matrix
    
    def predict_infection(self, initial_state, t=1):
        """
        预测t步后的感染状态
        
        参数:
            initial_state: 初始感染状态向量 (numpy array)
            t: 时间步数
        
        返回:
            预测的感染状态向量 (numpy array)
        """
        propagation_matrix = self.get_propagation_matrix(t)
        return propagation_matrix @ initial_state

# 主函数
def main():
    # 加载数据
    X, y, attack_cat, df = load_unsw_nb15(r'D:\code\github\kaggle_data\mrwellsdavid\unsw-nb15\versions\1\UNSW_NB15_training-set.csv')
    
    # 构建图数据
    data = build_graph_data(X, y, attack_cat, df, k_neighbors=3)  # 进一步减少k值
    
    # 构建稀疏邻接矩阵
    num_nodes = data.num_nodes
    edge_index = data.edge_index.cpu().numpy()
    edge_weight = data.edge_attr.cpu().numpy()
    
    # 创建稀疏邻接矩阵 (CSR格式)
    adj_matrix = sp.csr_matrix(
        (edge_weight, (edge_index[0], edge_index[1])),
        shape=(num_nodes, num_nodes)
    )
    
    print(f"邻接矩阵形状: {adj_matrix.shape}")
    print(f"邻接矩阵密度: {adj_matrix.nnz / (num_nodes * num_nodes):.6f}")
    
    # 计算约旦矩阵
    jordan_decomp = JordanMatrixDecomposition(
        adj_matrix, 
        n_components=30,  # 进一步减少特征值数量
        solver='lobpcg',  # 使用LOBPCG求解器
        max_iter=20000,   # 增加最大迭代次数
        tol=1e-2,         # 降低精度要求
        preprocess=True,  # 启用矩阵预处理
        use_pca_init=True,  # 使用PCA生成初始向量
        n_pca_components=100  # PCA组件数量
    )
    
    # 打印特征值分布
    plt.figure(figsize=(10, 6))
    plt.hist(jordan_decomp.eigenvalues, bins=30)
    plt.title('特征值分布')
    plt.xlabel('特征值')
    plt.ylabel('频率')
    
    # 保存图像时忽略字体警告
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.savefig('eigenvalue_distribution.png')
    plt.close()
    
    print(f"计算得到 {len(jordan_decomp.eigenvalues)} 个特征值")
    print(f"特征值范围: [{min(jordan_decomp.eigenvalues):.6f}, {max(jordan_decomp.eigenvalues):.6f}]")
    
    # 后续代码与之前相同...
    # 初始化模型、训练、评估等操作

if __name__ == '__main__':
    main()