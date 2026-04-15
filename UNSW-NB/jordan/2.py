import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import time

class JordanMatrixDecomposition:
    """约旦矩阵分解类，用于网络数据分析和异常检测"""
    
    def __init__(self, adj_matrix, n_components=100, solver='lobpcg', max_iter=30000, tol=1e-3, 
                 preprocess=True, use_pca_init=True, n_pca_components=200, postprocess=True,
                 use_pinv=True):
        self.adj_matrix = adj_matrix
        self.n_components = n_components
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.preprocess = preprocess
        self.use_pca_init = use_pca_init
        self.n_pca_components = n_pca_components
        self.postprocess = postprocess
        self.use_pinv = use_pinv  # 控制是否使用伪逆
        self.eigenvalues = None
        self.eigenvectors = None
        self.Q = None  # 特征向量矩阵
        self.J = None  # 约旦标准型
    
    def fit(self):
        """执行约旦矩阵分解"""
        A = self.adj_matrix.copy()
        
        if self.preprocess:
            A = A + sparse.eye(A.shape[0]) * 1e-10  # 添加对角扰动
            A = (A + A.T) / 2.0  # 对称化矩阵
        
        # 计算特征值和特征向量
        if self.solver == 'lobpcg':
            from scipy.sparse.linalg import eigsh
            
            # PCA初始化
            init_vector = None
            if self.use_pca_init and A.shape[0] > self.n_pca_components:
                print("使用PCA初始化特征向量...")
                pca = TruncatedSVD(n_components=min(self.n_pca_components, A.shape[0]-1))
                X_pca = pca.fit_transform(A)
                init_vector = X_pca[:, :self.n_components]
            
            # 计算特征值
            try:
                t0 = time.time()
                eigenvalues, eigenvectors = eigsh(A, k=self.n_components, which='LM', 
                                                 maxiter=self.max_iter, tol=self.tol,
                                                 v0=init_vector)
                t1 = time.time()
                print(f"特征值计算完成，耗时: {t1 - t0:.2f}秒")
                
                # 排序
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
                
                self.eigenvalues = np.real(eigenvalues)
                self.eigenvectors = np.real(eigenvectors)
                
                idx = self.eigenvalues.argsort()[::-1]
                self.eigenvalues = self.eigenvalues[idx]
                self.eigenvectors = self.eigenvectors[:, idx]
                
            except Exception as e:
                print(f"特征值计算失败: {e}")
                raise
        
        else:
            raise ValueError("无效的求解器选择。请使用 'lobpcg' 或 'arpack'。")
        
        # 构建约旦标准型
        self.Q = self.eigenvectors
        self.J = np.diag(self.eigenvalues)
        
        if self.postprocess:
            # 归一化特征向量
            for i in range(self.Q.shape[1]):
                self.Q[:, i] = self.Q[:, i] / np.linalg.norm(self.Q[:, i])
        
        return self
    
    def get_propagation_matrix(self, t=1):
        """计算传播矩阵"""
        if self.J is None or self.Q is None:
            raise ValueError("请先调用fit()方法")
        
        exp_J = np.diag(np.exp(self.eigenvalues * t))
        
        if self.use_pinv:
            from scipy.linalg import pinv
            Q_inv = pinv(self.Q)
            propagation_matrix = self.Q @ exp_J @ Q_inv
        else:
            propagation_matrix = self.Q @ exp_J @ self.Q.T
        
        return propagation_matrix
    
    def predict_infection(self, initial_state, t=1):
        """预测感染状态"""
        if self.J is None or self.Q is None:
            raise ValueError("请先调用fit()方法")
        
        propagation_matrix = self.get_propagation_matrix(t)
        predicted_state = propagation_matrix @ initial_state
        
        return predicted_state


def load_unsw_nb_data(file_path, sample_size=None):
    """加载并预处理数据"""
    print(f"加载数据从: {file_path}")
    df = pd.read_csv(file_path)
    
    # 采样
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
        print(f"数据采样完成，样本大小: {len(df)}")
    
    # 分离特征和标签
    X = df.drop(['label', 'attack_cat'], axis=1, errors='ignore')
    y = df.get('label', pd.Series())
    
    # 处理分类特征
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # 处理缺失值和标准化
    X = X.fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("数据预处理完成")
    return X_scaled, y


def create_adjacency_matrix(X, threshold=0.5, chunk_size=1000):
    """构建邻接矩阵（内存优化）"""
    print("构建邻接矩阵...")
    n_samples = X.shape[0]
    rows, cols, data = [], [], []
    
    # 分块计算
    for i in range(0, n_samples, chunk_size):
        end_i = min(i + chunk_size, n_samples)
        print(f"处理样本 {i}-{end_i-1}/{n_samples}")
        
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_chunk = cosine_similarity(X[i:end_i], X)
        
        # 筛选阈值以上的连接
        for row_idx in range(similarity_chunk.shape[0]):
            for col_idx in range(similarity_chunk.shape[1]):
                if similarity_chunk[row_idx, col_idx] > threshold and row_idx != col_idx:
                    rows.append(i + row_idx)
                    cols.append(col_idx)
                    data.append(similarity_chunk[row_idx, col_idx])
    
    # 构建稀疏矩阵
    adj_matrix = sparse.coo_matrix((data, (rows, cols)), shape=(n_samples, n_samples))
    sparse_adj_matrix = adj_matrix.tocsr()
    
    print(f"邻接矩阵构建完成，稀疏性: {100 * (1 - sparse_adj_matrix.count_nonzero() / (sparse_adj_matrix.shape[0] ** 2)):.2f}%")
    return sparse_adj_matrix


def main():
    """主函数：训练+评估（保留关键指标）"""
    # 配置参数
    train_file_path = r'D:\code\github\kaggle_data\mrwellsdavid\unsw-nb15\versions\1\UNSW_NB15_training-set.csv'
    test_file_path = r'D:\code\github\kaggle_data\mrwellsdavid\unsw-nb15\versions\1\UNSW_NB15_testing-set.csv'
    sample_size = 10000  # 采样大小
    n_components = 100   # 特征值数量
    time_steps = 3       # 传播时间步
    threshold = 0.7      # 邻接矩阵阈值
    chunk_size = 1000    # 分块大小
    use_pinv = True      # 是否使用伪逆
    
    # 加载数据
    X_train, y_train = load_unsw_nb_data(train_file_path, sample_size)
    X_test, y_test = load_unsw_nb_data(test_file_path, sample_size)
    
    # 构建邻接矩阵
    adj_matrix = create_adjacency_matrix(X_train, threshold=threshold, chunk_size=chunk_size)
    
    # 约旦矩阵分解（训练）
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
    
    # 生成训练集预测（计算train acc）
    print("生成训练集预测...")
    initial_state = np.zeros(X_train.shape[0])
    initial_state[y_train == 1] = 1.0  # 标记攻击样本
    train_pred_state = jordan_decomp.predict_infection(initial_state, t=time_steps)
    y_train_pred = (train_pred_state > 0.5).astype(int)  # 二值化
    
    # 生成测试集预测
    print("生成测试集预测...")
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(X_train)
    _, indices = nn.kneighbors(X_test)
    y_test_proba = train_pred_state[indices].flatten()
    y_test_pred = (y_test_proba > 0.5).astype(int)  # 二值化
    
    # 计算关键指标
    ## 训练集指标
    train_acc = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    
    ## 测试集指标
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    # 输出并保留关键指标
    print("\n===== 关键指标 =====")
    print(f"训练集准确率 (Train Acc): {train_acc:.4f}")
    print(f"训练集F1分数 (Train F1): {train_f1:.4f}")
    print(f"测试集准确率 (Test Acc): {test_acc:.4f}")
    print(f"测试集F1分数 (Test F1): {test_f1:.4f}")
    
    # 保存指标到文件（方便后续查看）
    with open('metrics.txt', 'w') as f:
        f.write(f"Train Accuracy: {train_acc:.4f}\n")
        f.write(f"Train F1 Score: {train_f1:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test F1 Score: {test_f1:.4f}\n")
    print("\n指标已保存到 metrics.txt")
    
    # 可视化混淆矩阵
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['正常', '攻击'], yticklabels=['正常', '攻击'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("混淆矩阵已保存为 confusion_matrix.png")


if __name__ == "__main__":
    main()