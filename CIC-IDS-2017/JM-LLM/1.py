import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

# 设置随机种子确保结果可复现
np.random.seed(42)
torch.manual_seed(42)

# 检查是否有GPU可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义批量大小，用于批量处理数据
BATCH_SIZE = 1024

class JordanModel(nn.Module):
    """约旦递归神经网络模型，用于威胁传播预测"""
    
    def __init__(self, input_dim, output_dim):
        super(JordanModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 定义网络层
        self.hidden = nn.Linear(input_dim + output_dim, 64)
        self.output = nn.Linear(64, output_dim)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        # 优化器
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        
    def forward(self, x):
        batch_size = x.size(0)
        # 初始化上一次的输出（反馈连接）
        prev_output = torch.zeros(batch_size, self.output_dim, device=device)
        
        # 向量化实现，提高效率
        combined = torch.cat([x, prev_output], dim=1)
        hidden = self.tanh(self.hidden(combined))
        output = self.sigmoid(self.output(hidden))
        
        return output
    
    def generate_prediction(self, X):
        """生成预测结果，使用数据加载器处理大数据集"""
        self.eval()
        predictions = []
        
        # 如果是numpy数组，转换为PyTorch张量
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X).to(device)
        
        # 创建数据加载器
        dataset = TensorDataset(X)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        with torch.no_grad():
            for batch in dataloader:
                x_batch = batch[0].to(device)
                pred = self.forward(x_batch)
                predictions.append(pred.cpu().numpy())
        
        return np.vstack(predictions)
    
    def refine_parameters(self, X, y_true, learning_rate=0.001):
        """根据梯度调整模型参数，使用数据加载器处理大数据集"""
        self.train()
        
        # 创建数据加载器
        dataset = TensorDataset(X, y_true)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            self.optimizer.zero_grad()
            
            # 计算损失
            y_pred = self.forward(x_batch)
            loss = self.loss_fn(y_pred, y_batch.view(-1, self.output_dim))
            
            # 反向传播
            loss.backward()
            
            # 手动更新参数
            with torch.no_grad():
                for param in self.parameters():
                    if param.grad is not None:
                        param.data -= learning_rate * param.grad.data

def load_and_preprocess_data(data_path):
    """加载并预处理CIC-IDS-2017数据集"""
    # 检查文件是否存在
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据集文件不存在: {data_path}")
    
    # 加载数据
    try:
        print(f"正在加载数据: {data_path}")
        data = pd.read_csv(data_path)
        print(f"数据加载完成，共有 {data.shape[0]} 行，{data.shape[1]} 列")
    except Exception as e:
        raise Exception(f"读取数据时出错: {e}")
    
    # 检查数据基本情况
    print("数据基本信息:")
    data.info()
    
    # 检查标签列
    label_column = None
    possible_label_columns = ['Label', ' Label', 'label', 'Class', 'class']
    
    for col in possible_label_columns:
        if col in data.columns:
            label_column = col
            print(f"找到标签列: '{label_column}'")
            break
    
    if label_column is None:
        raise ValueError(f"数据集中未找到标签列，请检查数据集格式。可能的标签列名: {possible_label_columns}")
    
    # 检查标签分布
    print("\n标签分布:")
    print(data[label_column].value_counts())
    
    # 提取特征和标签
    X = data.drop([label_column], axis=1)
    y = data[label_column]
    
    # 处理分类特征
    categorical_cols = X.select_dtypes(include=['object']).columns
    if not categorical_cols.empty:
        print(f"发现分类特征: {', '.join(categorical_cols)}")
        print("正在进行独热编码...")
        X = pd.get_dummies(X, columns=categorical_cols)
    else:
        print("未发现分类特征")
    
    # 处理缺失值
    missing_values = X.isnull().sum().sum()
    if missing_values > 0:
        print(f"发现 {missing_values} 个缺失值，正在填充...")
        X = X.fillna(X.mean())
    else:
        print("未发现缺失值")
    
    # 处理无穷大值
    print("检查并处理无穷大值...")
    X = X.replace([np.inf, -np.inf], np.nan)
    nan_count = X.isnull().sum().sum()
    
    if nan_count > 0:
        print(f"发现 {nan_count} 个无穷大值，已替换为NaN，正在填充...")
        X = X.fillna(X.mean())
    
    # 检查特征中的异常值
    print("检查并处理异常值...")
    for col in X.select_dtypes(include=[np.number]).columns:
        # 计算IQR
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # 定义异常值的上下限
        lower_bound = Q1 - 10 * IQR  # 使用10倍IQR，而不是标准的1.5倍，以保留更多数据
        upper_bound = Q3 + 10 * IQR
        
        # 替换异常值
        X.loc[X[col] < lower_bound, col] = lower_bound
        X.loc[X[col] > upper_bound, col] = upper_bound
    
    # 标准化数值特征
    print("正在标准化数值特征...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 将标签转换为二进制（正常/攻击）
    print("正在将标签转换为二进制分类...")
    y_binary = (y != 'BENIGN').astype(int)
    
    # 划分训练集和测试集
    print("正在划分训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.3, random_state=42, stratify=y_binary
    )
    
    print(f"训练集大小: {X_train.shape[0]}，测试集大小: {X_test.shape[0]}")
    
    # 确保标签是一维数组
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
    
    return X_train, X_test, y_train, y_test        

class LMModel(nn.Module):
    """基于Transformer的大语言模型，用于威胁传播分析"""
    
    def __init__(self, input_dim, output_dim):
        super(LMModel, self).__init__()
        
        # 定义网络层
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )
        
        # 优化器
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.encoder(x)
    
    def perform_analysis(self, X):
        """执行威胁传播分析并生成预测，使用数据加载器处理大数据集"""
        self.eval()
        predictions = []
        
        # 如果是numpy数组，转换为PyTorch张量
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X).to(device)
        
        # 创建数据加载器
        dataset = TensorDataset(X)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        with torch.no_grad():
            for batch in dataloader:
                x_batch = batch[0].to(device)
                pred = self.forward(x_batch)
                # 返回softmax概率
                predictions.append(torch.softmax(pred, dim=1).cpu().numpy())
        
        return np.vstack(predictions)
    
    def adjust_attention_weights(self, X, y_true, learning_rate=0.001):
        """根据梯度调整注意力权重，使用数据加载器处理大数据集"""
        self.train()
        
        # 创建数据加载器
        dataset = TensorDataset(X, y_true)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            self.optimizer.zero_grad()
            
            # 计算损失
            y_pred = self.forward(x_batch)
            loss = self.loss_fn(y_pred, y_batch)
            
            # 反向传播
            loss.backward()
            
            # 手动更新参数
            with torch.no_grad():
                for param in self.parameters():
                    if param.grad is not None:
                        param.data -= learning_rate * param.grad.data

def calculate_range_deviation(jordan_pred, real_data):
    """计算预测范围与实际数据的偏差"""
    return mean_squared_error(real_data, jordan_pred)

def measure_node_discrepancy(jordan_pred, real_data):
    """测量预测节点与实际节点的差异"""
    # 将预测值和实际值转换为二进制（0或1）
    jordan_binary = (jordan_pred > 0.5).astype(int)
    real_binary = real_data.astype(int)
    
    # 计算不匹配的节点比例
    return np.mean(jordan_binary != real_binary)

def assess_model_dynamics(jordan_model, lm_model, X):
    """评估两个模型之间的动态差异，使用数据加载器分批计算以避免内存溢出"""
    # 分批次计算预测结果，避免内存溢出
    total_samples = 0
    weighted_mse_sum = 0.0
    
    # 如果是numpy数组，转换为PyTorch张量
    if isinstance(X, np.ndarray):
        X = torch.FloatTensor(X).to(device)
    
    # 创建数据加载器
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    with torch.no_grad():
        for batch in dataloader:
            x_batch = batch[0].to(device)
            
            # 约旦模型预测
            jordan_batch_pred = jordan_model.forward(x_batch).cpu().numpy()
            
            # LM模型预测
            lm_batch_pred = lm_model.forward(x_batch).cpu().numpy()
            
            # 将LM的概率预测转换为二元预测
            lm_batch_binary = np.argmax(lm_batch_pred, axis=1).reshape(-1, 1)
            
            # 计算当前批次的MSE
            batch_size = x_batch.size(0)
            batch_mse = mean_squared_error(jordan_batch_pred, lm_batch_binary)
            
            # 加权累积MSE（按批次大小加权）
            weighted_mse_sum += batch_mse * batch_size
            total_samples += batch_size
    
    # 计算平均MSE
    if total_samples > 0:
        return weighted_mse_sum / total_samples
    else:
        return 0.0    

def combine_errors(range_err, node_err, dynamic_err):
    """组合多种误差指标"""
    # 为不同类型的误差分配权重
    range_weight = 0.4
    node_weight = 0.4
    dynamic_weight = 0.2
    
    # 计算总误差
    total_error = (range_weight * range_err + 
                  node_weight * node_err + 
                  dynamic_weight * dynamic_err)
    
    return total_error

def closed_loop_optimization(jordan_model, lm_model, X_train, y_train, threshold=0.15, max_iterations=100):
    """闭环优化函数，迭代优化约旦模型和大语言模型，使用数据加载器处理大数据集"""
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    
    iteration = 0
    total_error = float('inf')
    
    while total_error > threshold and iteration < max_iterations:
        print(f"\nIteration {iteration+1}:")
        
        # 1. 双模型并行预测
        print("  双模型并行预测中...")
        jordan_pred = jordan_model.generate_prediction(X_train)
        lm_pred = lm_model.perform_analysis(X_train)
        
        # 2. 多维误差计算
        print("  计算多维误差...")
        range_err = calculate_range_deviation(jordan_pred, y_train)
        node_err = measure_node_discrepancy(jordan_pred, y_train)
        dynamic_err = assess_model_dynamics(jordan_model, lm_model, X_train)
        total_error = combine_errors(range_err, node_err, dynamic_err)
        
        print(f"  Total Error = {total_error:.4f}")
        
        # 3. 收敛条件检测
        if total_error <= threshold:
            print(f"收敛成功，总误差: {total_error:.4f}")
            break
        
        # 4. 约旦矩阵参数调整
        print("  调整约旦模型参数...")
        jordan_model.refine_parameters(X_train_tensor, y_train_tensor.view(-1, 1))
        
        # 5. 大模型注意力优化
        print("  优化大语言模型注意力权重...")
        y_train_indices = torch.LongTensor(y_train).to(device)
        lm_model.adjust_attention_weights(X_train_tensor, y_train_indices)
        
        iteration += 1
    
    if iteration == max_iterations:
        print(f"达到最大迭代次数，最终误差: {total_error:.4f}")
    
    return jordan_model, lm_model

def evaluate_model(jordan_model, lm_model, X_test, y_test):
    """评估优化后的模型性能，使用数据加载器处理大数据集"""
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("评估模型性能中...")
    
    # 获取预测结果
    jordan_pred = jordan_model.generate_prediction(X_test)
    lm_pred = lm_model.perform_analysis(X_test)
    
    # 将预测结果转换为二进制分类
    jordan_binary = (jordan_pred > 0.5).astype(int).reshape(-1)
    lm_binary = np.argmax(lm_pred, axis=1)
    
    # 计算模型性能指标
    jordan_mse = mean_squared_error(y_test, jordan_pred)
    jordan_acc = np.mean(jordan_binary == y_test)
    lm_acc = np.mean(lm_binary == y_test)
    
    print("\n约旦模型性能:")
    print(f"MSE: {jordan_mse:.4f}")
    print(f"准确率: {jordan_acc:.4f}")
    print("分类报告:")
    print(classification_report(y_test, jordan_binary, target_names=['正常', '攻击']))
    print("混淆矩阵:")
    print(confusion_matrix(y_test, jordan_binary))
    
    print("\n大语言模型性能:")
    print(f"准确率: {lm_acc:.4f}")
    print("分类报告:")
    print(classification_report(y_test, lm_binary, target_names=['正常', '攻击']))
    print("混淆矩阵:")
    print(confusion_matrix(y_test, lm_binary))
    
    return jordan_mse, jordan_acc, lm_acc

def main():
    """主函数，执行完整的威胁传播预测流程"""
    # 数据路径
    data_path = r"D:\code\lunwen\data\merged.csv"
    
    try:
        # 加载并预处理数据
        print("开始加载并预处理CIC-IDS-2017数据集...")
        X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)
        
        # 创建模型并移至设备
        print("\n初始化模型...")
        input_dim = X_train.shape[1]
        output_dim = 1  # 二分类问题
        
        jordan_model = JordanModel(input_dim, output_dim).to(device)
        lm_model = LMModel(input_dim, 2).to(device)  # LMModel使用2个输出类别
        
        # 闭环优化
        print("\n开始闭环优化...")
        optimized_jordan, optimized_lm = closed_loop_optimization(
            jordan_model, lm_model, X_train, y_train, threshold=0.1, max_iterations=20
        )
        
        # 评估模型
        print("\n评估模型性能...")
        evaluate_model(optimized_jordan, optimized_lm, X_test, y_test)
        
        print("\n威胁传播预测流程完成!")
        
    except Exception as e:
        print(f"程序执行出错: {e}")

if __name__ == "__main__":
    main()    