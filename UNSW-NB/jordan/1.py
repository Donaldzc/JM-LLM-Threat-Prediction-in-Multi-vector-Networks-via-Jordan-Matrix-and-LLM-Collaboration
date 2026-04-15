import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import time

class JordanMatrixDecomposition:
    """约旦矩阵分解类，用于网络数据分析和异常检测"""
    
    def __init__(self, adj_matrix, n_components=100, solver='lobpcg', max_iter=30000, tol=1e-3, 
                 preprocess=True, use_pca_init=True, n_pca_components=200, postprocess=True,
                 use_pinv=True):
        """
        初始化约旦矩阵分解器
        
        参数:
        adj_matrix: 邻接矩阵，通常是稀疏矩阵
        n_components: 要计算的特征值/特征向量数量
        solver: 特征值求解器 ('lobpcg' 或 'arpack')
        max_iter: 最大迭代次数
        tol: 收敛容差
        preprocess: 是否对矩阵进行预处理
        use_pca_init: 是否使用PCA初始化特征向量
        n_pca_components: PCA组件数量
        postprocess: 是否对结果进行后处理
        use_pinv: 是否使用伪逆计算传播矩阵
        """
        self.adj_matrix = adj_matrix
        self.n_components = n_components
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.preprocess = preprocess
        self.use_pca_init = use_pca_init
        self.n_pca_components = n_pca_components
        self.postprocess = postprocess
        self.use_pinv = use_pinv  # 是否使用伪逆
        self.eigenvalues = None
        self.eigenvectors = None
        self.Q = None  # 特征向量矩阵
        self.J = None  # 约旦标准型
    
    def fit(self):
        """执行约旦矩阵分解"""
        # 矩阵预处理
        A = self.adj_matrix.copy()
        
        if self.preprocess:
            # 添加小的对角扰动以改善数值稳定性
            A = A + sparse.eye(A.shape[0]) * 1e-10
            
            # 对称化矩阵 (A + A^T)/2
            A = (A + A.T) / 2.0
        
        # 计算特征值和特征向量
        if self.solver == 'lobpcg':
            from scipy.sparse.linalg import eigsh
            
            # 使用PCA初始化特征向量
            if self.use_pca_init and A.shape[0] > self.n_pca_components:
                print("使用PCA初始化特征向量...")
                pca = TruncatedSVD(n_components=min(self.n_pca_components, A.shape[0]-1))
                X_pca = pca.fit_transform(A)
                init_vector = X_pca[:, :self.n_components]
            else:
                init_vector = None
            
            # 计算最大的n_components个特征值和特征向量
            try:
                t0 = time.time()
                eigenvalues, eigenvectors = eigsh(A, k=self.n_components, which='LM', 
                                                 maxiter=self.max_iter, tol=self.tol,
                                                 v0=init_vector)
                t1 = time.time()
                print(f"特征值计算完成，耗时: {t1 - t0:.2f}秒")
                
                # 特征值按降序排列
                idx = eigenvalues.argsort()[::-1]
                self.eigenvalues = eigenvalues[idx]
                self.eigenvectors = eigenvectors[:, idx]
                
            except Exception as e:
                print(f"特征值计算失败: {e}")
                print("尝试使用ARPACK求解器...")
                self.solver = 'arpack'
                return self.fit()
        
        elif self.solver == 'arpack':
            from scipy.sparse.linalg import eigs
            
            try:
                t0 = time.time()
                eigenvalues, eigenvectors = eigs(A, k=self.n_components, which='LM', 
                                               maxiter=self.max_iter, tol=self.tol)
                t1 = time.time()
                print(f"特征值计算完成，耗时: {t1 - t0:.2f}秒")
                
                # 保留实部，因为邻接矩阵是对称的，特征值应为实数
                self.eigenvalues = np.real(eigenvalues)
                self.eigenvectors = np.real(eigenvectors)
                
                # 特征值按降序排列
                idx = self.eigenvalues.argsort()[::-1]
                self.eigenvalues = self.eigenvalues[idx]
                self.eigenvectors = self.eigenvectors[:, idx]
                
            except Exception as e:
                print(f"特征值计算失败: {e}")
                raise
        
        else:
            raise ValueError("无效的求解器选择。请使用 'lobpcg' 或 'arpack'。")
        
        # 构建约旦标准型 (简化版，假设矩阵可对角化)
        self.Q = self.eigenvectors
        self.J = np.diag(self.eigenvalues)
        
        if self.postprocess:
            # 归一化特征向量
            for i in range(self.Q.shape[1]):
                self.Q[:, i] = self.Q[:, i] / np.linalg.norm(self.Q[:, i])
        
        return self
    
    def get_propagation_matrix(self, t=1):
        """
        计算传播矩阵 e^(At)，其中A是原始矩阵，t是时间参数
        
        参数:
        t: 时间参数
        
        返回:
        传播矩阵
        """
        if self.J is None or self.Q is None:
            raise ValueError("请先调用fit()方法")
        
        # 计算对角矩阵的指数
        exp_J = np.diag(np.exp(self.eigenvalues * t))
        
        if self.use_pinv:
            # 使用伪逆计算传播矩阵（处理非方阵情况）
            from scipy.linalg import pinv
            Q_inv = pinv(self.Q)
            propagation_matrix = self.Q @ exp_J @ Q_inv
        else:
            # 使用特征向量近似传播矩阵（避免求逆）
            propagation_matrix = self.Q @ exp_J @ self.Q.T
        
        return propagation_matrix
    
    def predict_infection(self, initial_state, t=1):
        """
        预测经过t时间步后的感染状态
        
        参数:
        initial_state: 初始状态向量
        t: 时间参数
        
        返回:
        预测的感染状态向量
        """
        if self.J is None or self.Q is None:
            raise ValueError("请先调用fit()方法")
        
        # 计算传播矩阵
        propagation_matrix = self.get_propagation_matrix(t)
        
        # 预测感染状态
        predicted_state = propagation_matrix @ initial_state
        
        return predicted_state

def load_unsw_nb_data(file_path, sample_size=None):
    """
    加载UNSW-NB15数据集
    
    参数:
    file_path: 数据集文件路径
    sample_size: 采样大小，如果提供则随机采样数据
    
    返回:
    特征数据和标签
    """
    print(f"加载数据从: {file_path}")
    df = pd.read_csv(file_path)
    
    # 数据采样（如果指定）
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
        print(f"数据采样完成，样本大小: {len(df)}")
    
    # 数据预处理
    print("数据预处理开始...")
    
    # 分离特征和标签
    X = df.drop(['label', 'attack_cat'], axis=1, errors='ignore')
    y = df.get('label', pd.Series())
    
    # 处理分类特征
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # 处理缺失值
    X = X.fillna(0)
    
    # 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("数据预处理完成")
    return X_scaled, y

def create_adjacency_matrix(X, threshold=0.5, chunk_size=1000):
    """
    从特征矩阵创建邻接矩阵（内存优化版本）
    
    参数:
    X: 特征矩阵
    threshold: 相似度阈值，用于构建稀疏邻接矩阵
    chunk_size: 分块计算的大小，控制内存使用量
    
    返回:
    邻接矩阵（稀疏矩阵格式）
    """
    print("构建邻接矩阵...")
    n_samples = X.shape[0]
    
    # 使用COO格式构建稀疏矩阵
    rows = []
    cols = []
    data = []
    
    # 分块计算相似度矩阵
    for i in range(0, n_samples, chunk_size):
        end_i = min(i + chunk_size, n_samples)
        print(f"处理样本 {i}-{end_i-1}/{n_samples}")
        
        # 计算当前块与所有样本的相似度
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_chunk = cosine_similarity(X[i:end_i], X)
        
        # 找出大于阈值的相似度值
        for row_idx in range(similarity_chunk.shape[0]):
            for col_idx in range(similarity_chunk.shape[1]):
                if similarity_chunk[row_idx, col_idx] > threshold and row_idx != col_idx:
                    rows.append(i + row_idx)
                    cols.append(col_idx)
                    data.append(similarity_chunk[row_idx, col_idx])
    
    # 创建稀疏矩阵
    adj_matrix = sparse.coo_matrix((data, (rows, cols)), shape=(n_samples, n_samples))
    
    # 转换为CSR格式以提高运算效率
    sparse_adj_matrix = adj_matrix.tocsr()
    
    print(f"邻接矩阵构建完成，稀疏性: {100 * (1 - sparse_adj_matrix.count_nonzero() / (sparse_adj_matrix.shape[0] ** 2)):.2f}%")
    return sparse_adj_matrix

def main():
    """主函数：UNSW-NB数据集攻击预测流程"""
    # 配置参数
    train_file_path = r'D:\code\github\kaggle_data\mrwellsdavid\unsw-nb15\versions\1\UNSW_NB15_training-set.csv'
    test_file_path = r'D:\code\github\kaggle_data\mrwellsdavid\unsw-nb15\versions\1\UNSW_NB15_testing-set.csv'
    sample_size = 10000  # 数据采样大小，减小以加快测试速度
    n_components = 100  # 约旦矩阵分解的组件数
    time_steps = 3  # 传播时间步数
    threshold = 0.7  # 邻接矩阵构建阈值（提高阈值可减少连接数）
    chunk_size = 1000  # 邻接矩阵分块计算大小
    use_pinv = True  # 使用伪逆计算传播矩阵
    
    # 加载训练数据
    X_train, y_train = load_unsw_nb_data(train_file_path, sample_size)
    
    # 加载测试数据
    X_test, y_test = load_unsw_nb_data(test_file_path, sample_size)
    
    # 创建邻接矩阵（内存优化版本）
    adj_matrix = create_adjacency_matrix(X_train, threshold=threshold, chunk_size=chunk_size)
    
    # 约旦矩阵分解
    print("开始约旦矩阵分解...")
    jordan_decomp = JordanMatrixDecomposition(
        adj_matrix, 
        n_components=n_components,
        solver='lobpcg',
        max_iter=30000,
        tol=1e-3,
        preprocess=True,
        use_pca_init=True,
        n_pca_components=min(200, X_train.shape[0]-1),
        use_pinv=use_pinv
    )
    jordan_decomp.fit()
    
    # 可视化特征值分布
    plt.figure(figsize=(10, 6))
    plt.plot(jordan_decomp.eigenvalues, 'o-')
    plt.title('特征值分布')
    plt.xlabel('特征值索引')
    plt.ylabel('特征值大小')
    plt.grid(True)
    plt.savefig('eigenvalues_distribution.png')
    plt.close()
    
    # 构建初始状态向量（基于训练标签）
    initial_state = np.zeros(X_train.shape[0])
    initial_state[y_train == 1] = 1.0  # 标记已知攻击样本
    
    # 预测传播
    print("开始攻击传播预测...")
    predicted_state = jordan_decomp.predict_infection(initial_state, t=time_steps)
    
    # 为测试样本找到最近邻的训练样本
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(X_train)
    distances, indices = nn.kneighbors(X_test)
    
    # 使用最近邻的预测状态作为测试样本的预测
    y_pred_proba = predicted_state[indices].flatten()
    
    # 基于阈值进行分类
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # 评估模型性能
    print("\n模型性能评估:")
    print(classification_report(y_test, y_pred, target_names=['正常', '攻击']))
    print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
    
    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['正常', '攻击'], yticklabels=['正常', '攻击'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    print("分析完成！结果已保存为图表")

if __name__ == "__main__":
    main()    