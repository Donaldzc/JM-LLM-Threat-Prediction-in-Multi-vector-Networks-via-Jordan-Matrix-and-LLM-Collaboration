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


def load_cic_ids_data(file_path, sample_size=None, test_size=0.3, random_state=42):
    """加载并预处理CIC-IDS-2017数据集（增加异常值处理）"""
    print(f"加载数据从: {file_path}")
    
    # 读取数据
    df = pd.read_csv(file_path)
    
    # 数据采样（如果指定）
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=random_state)
        print(f"数据采样完成，样本大小: {len(df)}")
    
    # 处理CIC-IDS-2017数据集的特殊问题
    print("数据预处理开始...")
    
    # 处理列名中的空格
    df.columns = df.columns.str.strip()
    
    # 处理非数值列和缺失值
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    print(f"发现非数值列: {non_numeric_cols}")
    
    # 处理标签列
    if 'Label' in df.columns:
        # 二值化标签: 0=正常，1=攻击
        df['label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
        df = df.drop('Label', axis=1)
    elif 'label' in df.columns:
        print("数据集已包含二值化标签列")
    else:
        raise ValueError("数据集中未找到标签列 (Label 或 label)")
    
    # 处理其他非数值列
    non_numeric_cols = [col for col in non_numeric_cols if col not in ['Label', 'label']]  # 排除已处理的标签列
    for col in non_numeric_cols:
        print(f"处理非数值列 '{col}': 尝试转换为数值或删除")
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # 无法转换的设为NaN
        except:
            print(f"无法转换列 '{col}', 删除该列")
            df = df.drop(col, axis=1)
    
    # 处理缺失值
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        print(f"发现 {missing_values} 个缺失值，用0填充")
        df = df.fillna(0)
    
    # --------------------------
    # 新增：处理无穷大值和异常值
    # --------------------------
    # 检查并替换无穷大值
    inf_count = np.isinf(df).sum().sum()
    if inf_count > 0:
        print(f"发现 {inf_count} 个无穷大值，替换为0")
        df = df.replace([np.inf, -np.inf], 0)
    
    # 检查并处理超出float64范围的极大值（使用99.9%分位数截断）
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'label']  # 排除标签列
    
    for col in numeric_cols:
        # 计算99.9%分位数（避免极端值影响）
        quantile = df[col].quantile(0.999)
        # 截断超过分位数的值
        df[col] = df[col].apply(lambda x: quantile if x > quantile else x)
        # 处理负值（如果是流量特征，负值可能不合理）
        if col not in ['Down/Up Ratio']:  # 特殊列允许负值
            df[col] = df[col].apply(lambda x: 0 if x < 0 else x)
    
    # 分离特征和标签
    X = df.drop('label', axis=1)
    y = df['label']
    
    # 特征标准化
    scaler = StandardScaler()
    try:
        X_scaled = scaler.fit_transform(X)
    except ValueError as e:
        # 再次检查是否还有异常值（用于调试）
        print(f"标准化时出错: {e}")
        print("检查异常值:")
        for col in X.columns:
            if np.isinf(X[col]).any():
                print(f"列 {col} 仍存在无穷大值")
            if (X[col] > np.finfo('float64').max).any():
                print(f"列 {col} 仍存在超出float64范围的值")
        raise  # 抛出异常停止执行
    
    # 划分训练集和测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    
    print("数据预处理完成")
    print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
    print(f"攻击样本比例: {y_train.mean():.4f} (训练集), {y_test.mean():.4f} (测试集)")
    
    return X_train, X_test, y_train, y_test


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
    file_path = r'D:\code\lunwen\data\merged.csv'
    sample_size = 20000  # 采样大小（增加样本量以适应CIC-IDS-2017）
    test_size = 0.3     # 测试集比例
    n_components = 150  # 特征值数量（增加以提高表征能力）
    time_steps = 3      # 传播时间步
    threshold = 0.7     # 邻接矩阵阈值
    chunk_size = 1000   # 分块大小
    use_pinv = True     # 是否使用伪逆
    
    # 加载数据
    X_train, X_test, y_train, y_test = load_cic_ids_data(
        file_path, sample_size=sample_size, test_size=test_size
    )
    
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
    with open('metrics_cic_ids.txt', 'w') as f:
        f.write(f"Train Accuracy: {train_acc:.4f}\n")
        f.write(f"Train F1 Score: {train_f1:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test F1 Score: {test_f1:.4f}\n")
    print("\n指标已保存到 metrics_cic_ids.txt")
    
    # 可视化混淆矩阵
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['正常', '攻击'], yticklabels=['正常', '攻击'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.savefig('confusion_matrix_cic_ids.png')
    plt.close()
    print("混淆矩阵已保存为 confusion_matrix_cic_ids.png")


if __name__ == "__main__":
    main()