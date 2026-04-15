import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
import os

# 设置随机种子确保结果可复现
np.random.seed(42)
torch.manual_seed(42)

# 检查是否有GPU可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        # 初始化上一次的输出（反馈连接）
        batch_size = x.size(0)
        prev_output = torch.zeros(batch_size, self.output_dim, device=device)
        
        # 对批次中的每个样本进行处理
        outputs = []
        for i in range(batch_size):
            # 合并输入和上一次的输出
            combined = torch.cat([x[i], prev_output[i]], dim=0)
            
            # 前向传播
            hidden = self.tanh(self.hidden(combined))
            output = self.sigmoid(self.output(hidden))
            
            # 更新上一次的输出
            prev_output[i] = output.detach()
            outputs.append(output)
        
        return torch.stack(outputs)
    
    def generate_prediction(self, X):
        """生成预测结果"""
        self.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.FloatTensor(X).to(device)
            return self.forward(X).cpu().numpy()
    
    def refine_parameters(self, X, y_true, learning_rate=0.001):
        """根据梯度调整模型参数"""
        self.train()
        self.optimizer.zero_grad()
        
        # 计算损失
        y_pred = self.forward(X)
        loss = self.loss_fn(y_pred, y_true.view(-1, self.output_dim))
        
        # 反向传播
        loss.backward()
        
        # 手动更新参数
        with torch.no_grad():
            for param in self.parameters():
                if param.grad is not None:
                    param.data -= learning_rate * param.grad.data

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
        """执行威胁传播分析并生成预测"""
        self.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.FloatTensor(X).to(device)
            # 返回softmax概率
            return torch.softmax(self.forward(X), dim=1).cpu().numpy()
    
    def adjust_attention_weights(self, X, y_true, learning_rate=0.001):
        """根据梯度调整注意力权重"""
        self.train()
        self.optimizer.zero_grad()
        
        # 计算损失
        y_pred = self.forward(X)
        loss = self.loss_fn(y_pred, y_true)
        
        # 反向传播
        loss.backward()
        
        # 手动更新参数
        with torch.no_grad():
            for param in self.parameters():
                if param.grad is not None:
                    param.data -= learning_rate * param.grad.data

def load_and_preprocess_data(train_path, test_path):
    """加载并预处理UNSW-NB15数据集"""
    # 检查文件是否存在
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("训练集或测试集文件不存在，请检查文件路径")
    
    # 加载数据
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
    except Exception as e:
        raise Exception(f"读取数据时出错: {e}")
    
    # 合并训练集和测试集进行预处理
    data = pd.concat([train_data, test_data], ignore_index=True)
    
    # 提取特征和标签
    X = data.drop(['id', 'label', 'attack_cat'], axis=1, errors='ignore')
    y = data['label']
    
    # 处理分类特征
    categorical_cols = X.select_dtypes(include=['object']).columns
    if not categorical_cols.empty:
        X = pd.get_dummies(X, columns=categorical_cols)
    
    # 标准化数值特征
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 确保标签是一维数组
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
    
    return X_train, X_test, y_train, y_test

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
    """评估两个模型之间的动态差异"""
    jordan_pred = jordan_model.generate_prediction(X)
    lm_pred = lm_model.perform_analysis(X)
    
    # 将LM的概率预测转换为二元预测
    lm_binary = np.argmax(lm_pred, axis=1).reshape(-1, 1)
    
    # 计算两个模型预测结果的差异
    return mean_squared_error(jordan_pred, lm_binary)

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
    """闭环优化函数，迭代优化约旦模型和大语言模型"""
    iteration = 0
    total_error = float('inf')
    
    while total_error > threshold and iteration < max_iterations:
        # 1. 双模型并行预测
        jordan_pred = jordan_model.generate_prediction(X_train)
        lm_pred = lm_model.perform_analysis(X_train)
        
        # 2. 多维误差计算
        range_err = calculate_range_deviation(jordan_pred, y_train)
        node_err = measure_node_discrepancy(jordan_pred, y_train)
        dynamic_err = assess_model_dynamics(jordan_model, lm_model, X_train)
        total_error = combine_errors(range_err, node_err, dynamic_err)
        
        print(f"Iteration {iteration+1}: Total Error = {total_error:.4f}")
        
        # 3. 收敛条件检测
        if total_error <= threshold:
            print(f"收敛成功，总误差: {total_error:.4f}")
            break
        
        # 4. 约旦矩阵参数调整
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_tensor = torch.FloatTensor(y_train).to(device).view(-1, 1)
        jordan_model.refine_parameters(X_train_tensor, y_train_tensor)
        
        # 5. 大模型注意力优化
        y_train_indices = torch.LongTensor(y_train).to(device)
        lm_model.adjust_attention_weights(X_train_tensor, y_train_indices)
        
        iteration += 1
    
    if iteration == max_iterations:
        print(f"达到最大迭代次数，最终误差: {total_error:.4f}")
    
    return jordan_model, lm_model

def evaluate_model(jordan_model, lm_model, X_test, y_test):
    """评估优化后的模型性能"""
    # 获取预测结果
    jordan_pred = jordan_model.generate_prediction(X_test)
    lm_pred = lm_model.perform_analysis(X_test)
    
    # 将LM的概率预测转换为二元预测
    lm_binary = np.argmax(lm_pred, axis=1)
    
    # 计算模型性能指标
    jordan_mse = mean_squared_error(y_test, jordan_pred)
    lm_accuracy = np.mean(lm_binary == y_test)
    
    print(f"约旦模型测试MSE: {jordan_mse:.4f}")
    print(f"大语言模型测试准确率: {lm_accuracy:.4f}")
    
    return jordan_mse, lm_accuracy

def main():
    """主函数，执行完整的威胁传播预测流程"""
    # 数据路径
    train_path = r"D:\code\github\kaggle_data\mrwellsdavid\unsw-nb15\versions\1\UNSW_NB15_training-set.csv"
    test_path = r"D:\code\github\kaggle_data\mrwellsdavid\unsw-nb15\versions\1\UNSW_NB15_testing-set.csv"
    
    try:
        # 加载并预处理数据
        print("加载并预处理数据...")
        X_train, X_test, y_train, y_test = load_and_preprocess_data(train_path, test_path)
        
        # 创建模型并移至设备
        print("初始化模型...")
        input_dim = X_train.shape[1]
        output_dim = 1  # 二分类问题
        
        jordan_model = JordanModel(input_dim, output_dim).to(device)
        lm_model = LMModel(input_dim, 2).to(device)  # LMModel使用2个输出类别
        
        # 闭环优化
        print("开始闭环优化...")
        optimized_jordan, optimized_lm = closed_loop_optimization(
            jordan_model, lm_model, X_train, y_train, threshold=0.1
        )
        
        # 评估模型
        print("评估模型性能...")
        evaluate_model(optimized_jordan, optimized_lm, X_test, y_test)
        
        print("威胁传播预测流程完成!")
        
    except Exception as e:
        print(f"程序执行出错: {e}")

if __name__ == "__main__":
    main()    