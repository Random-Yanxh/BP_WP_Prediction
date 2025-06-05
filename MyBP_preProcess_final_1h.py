import numpy as np          # 用于数值计算
import pandas as pd         # 用于数据处理
import matplotlib.pyplot as plt  # 用于绘图
import os                   # 用于文件操作
from datetime import datetime    # 用于时间处理
from scipy.stats import pearsonr  # 用于计算相关性

# ------------------------------
# 宏定义部分
# ------------------------------
# 预测相关参数
PREDICTION_STEPS = 1        # 预测步数（每次预测几个时间点）
TARGET_FORECAST_STEP = 4    # 提前预测步数（1步 = 15分钟）

# 神经网络结构参数
NETWORK_SHAPE = [None, 16, PREDICTION_STEPS]  # 网络层数和每层神经元数量
BATCH_SIZE = 16             # 每批训练的样本数量
LEARNING_RATE = 0.004       # 学习率（控制参数更新的步长）
EPOCHS = 300                # 训练轮数
WINDOW_SIZE = 96            # 滑动窗口大小（用多少个历史数据点来预测）

# 特征选择参数
ENABLE_FEATURE_SELECTION = False    # 是否启用特征选择 True / False
SELECTED_FEATURES_COUNT = 16       # 选择多少个最重要的特征
CORRELATION_METHOD = 'kendall'     # 相关性计算方法，可选 pearson, kendall, spearman 

# 文件路径配置
TRAIN_FILE_PATH = r'./train.xlsx'  # 训练数据文件路径
TEST_FILE_PATH = r'./test.xlsx'   # 测试数据文件路径
TARGET_COLUMN = '实际功率'  # 目标列名称

# ------------------------------
# 特征选择和相关性分析函数
# ------------------------------

def analyze_time_lag_correlation(data, window_size, target_forecast_step, method='pearson'):
    print(f"\n--- 开始时间滞后相关性分析 ---")
    print(f"窗口大小: {window_size}, 目标步数: {target_forecast_step}, 方法: {method}")

    # 确保数据是一维的
    if data.ndim > 1:
        data = data.flatten()

    # 初始化相关性列表
    correlations = []

    # 计算目标步骤的索引（从0开始）
    target_step_index = target_forecast_step - 1

    # 准备窗口数据和目标数据
    windows = []  # 存储所有窗口
    targets = []  # 存储所有目标值

    # 创建滑动窗口和对应的目标值
    for i in range(window_size, len(data) - target_step_index):
        # 提取当前窗口的数据
        current_window = data[i - window_size : i]
        windows.append(current_window)

        # 提取对应的目标值
        target_value = data[i + target_step_index]
        targets.append(target_value)

    # 转换为numpy数组
    windows = np.array(windows)
    targets = np.array(targets)

    print(f"创建了 {len(windows)} 个训练样本")

    # 计算每个时间位置的相关性
    for lag_position in range(window_size):
        # 提取当前时间位置的所有值
        lag_values = windows[:, lag_position]

        # 检查数据的变异性
        unique_lag_values = len(np.unique(lag_values))
        unique_target_values = len(np.unique(targets))

        if unique_lag_values < 2 or unique_target_values < 2:
            # 如果数据没有变异性，相关性设为0
            correlation_coefficient = 0.0
        else:
            # 根据指定方法计算相关性
            if method == 'pearson':
                correlation_coefficient, _ = pearsonr(lag_values, targets)
            elif method == 'spearman':
                from scipy.stats import spearmanr
                correlation_coefficient, _ = spearmanr(lag_values, targets)
            elif method == 'kendall':
                from scipy.stats import kendalltau
                correlation_coefficient, _ = kendalltau(lag_values, targets)
            else:
                raise ValueError(f"不支持的相关性方法: {method}")

        # 处理NaN值并取绝对值
        if np.isnan(correlation_coefficient):
            correlation_coefficient = 0.0

        correlations.append(abs(correlation_coefficient))

    # 转换为numpy数组
    correlations = np.array(correlations)

    # 选择最重要的特征
    total_features = len(correlations)
    features_to_select = SELECTED_FEATURES_COUNT

    if total_features < features_to_select or features_to_select <= 0:
        # 如果要选择的特征数量大于总特征数，就选择所有特征
        if total_features > 0:
            selected_indices = np.argsort(correlations)[-total_features:]
        else:
            selected_indices = np.array([])
    else:
        # 选择相关性最高的特征
        selected_indices = np.argsort(correlations)[-features_to_select:]

    # 保持时间顺序
    selected_indices = np.sort(selected_indices)

    print(f"相关性分析完成，选择了 {len(selected_indices)} 个特征")
    return correlations, selected_indices
    """
    返回：
    - correlations: 每个时间点的相关性值
    - selected_indices: 选择的特征索引
    """

#    根据选择的特征索引对输入数据进行特征选择。
def apply_feature_selection(X, selected_indices):
    if X.ndim != 2: raise ValueError(f"输入数据必须是二维的，当前形状: {X.shape}")
    if selected_indices.size == 0 :
        return X # 如果没有选择任何索引，则保持逻辑以返回原始 X
    X_selected = X[:, selected_indices]
    return X_selected

# ------------------------------
# 数据处理函数 (Data Processing Functions)
# ------------------------------
def create_sliding_window_dataset(data, window_size, target_forecast_step):
    """
    使用滑动窗口方法创建特征(X)和目标(y)数据集。
    """
    X, y = [], []
    X_current_actuals = [] 
    if data.ndim == 1: data = data.reshape(-1, 1)
    
    target_step_index = target_forecast_step - 1

    for i in range(window_size, len(data) - target_step_index):
        window_features = data[i-window_size:i, 0]
        X.append(window_features)
        X_current_actuals.append(window_features[-1])
        target_steps = data[i + target_step_index : i + target_step_index + 1, 0]
        y.append(target_steps)
        
    X, y, X_current_actuals = np.array(X), np.array(y), np.array(X_current_actuals)
    return X, y, X_current_actuals.reshape(-1,1)

def load_and_preprocess_data(file_path, target_column, window_size, target_forecast_step):
    """
    加载、预处理数据，创建滑动窗口数据集。
    """
    print(f"--- 开始加载和预处理数据: {file_path} ---")
    
    df = pd.read_excel(file_path, header=None, usecols=[0])
    df.rename(columns={0: target_column}, inplace=True)
 
    
    data_series = df[target_column].copy()
    if data_series.isnull().sum() > 0:
        data_series.fillna(method='ffill', inplace=True); data_series.fillna(method='bfill', inplace=True)
        if data_series.isnull().sum() > 0: data_series.fillna(0, inplace=True)
    
    data_series = pd.to_numeric(data_series, errors='coerce')
    if data_series.isnull().sum() > 0 : data_series.dropna(inplace=True)
 
    
    scaled_data = data_series.values.reshape(-1, 1)
    X, y, X_current_actuals = create_sliding_window_dataset(scaled_data, window_size, target_forecast_step)

    input_dim, output_dim = X.shape[1], y.shape[1] 
    # 自动修正神经网络的输入层和输出层的神经元个数，使其分别与输入数据的特征维度和输出的目标维度匹配。
    if NETWORK_SHAPE[0] is None or NETWORK_SHAPE[0] != input_dim : NETWORK_SHAPE[0] = input_dim
    if NETWORK_SHAPE[-1] != output_dim : NETWORK_SHAPE[-1] = output_dim
    
    print(f"--- 数据加载和预处理完成 (输入维度: {input_dim}, 输出维度: {output_dim}) ---")
    return X, y, X_current_actuals, input_dim, output_dim

# ------------------------------
# 激活函数和损失函数 (Activation & Loss Functions)，激活函数使用ReLU，损失函数用mse，最后评估使用C_R
# ------------------------------
def activation_relu(inputs): return np.maximum(0, inputs)
def activation_relu_derivative(inputs): return np.where(inputs > 0, 1.0, 0.0)
def activation_linear(inputs): return inputs
def activation_linear_derivative(inputs): return np.ones_like(inputs)

def loss_mse(y_pred, y_true):
    if y_pred.shape != y_true.shape: y_true = y_true.reshape(y_pred.shape)
    if y_true.shape[0] == 0: return 0.0
    return np.mean((y_pred - y_true) ** 2)

def loss_mse_derivative(y_pred, y_true):#导数
    if y_pred.shape != y_true.shape: y_true = y_true.reshape(y_pred.shape)
    if y_true.shape[0] == 0: return np.zeros_like(y_pred)
    return 2.0 * (y_pred - y_true) / y_true.shape[0]
# 给定的准确率C_R计算函数
def calculate_cr_accuracy(P_M, P_P):
    P_M, P_P = np.array(P_M), np.array(P_P)
    if P_M.shape != P_P.shape: raise ValueError(f"实际功率和预测功率的形状必须一致。P_M: {P_M.shape}, P_P: {P_P.shape}")
    if P_M.size == 0: return 0.0
    N = len(P_M); R_values = np.zeros_like(P_M, dtype=float); epsilon = 1e-9
    mask_gt_02 = P_M > 0.2
    R_values[mask_gt_02] = (P_M[mask_gt_02] - P_P[mask_gt_02]) / (P_M[mask_gt_02] + epsilon)
    mask_le_02 = P_M <= 0.2
    R_values[mask_le_02] = (P_M[mask_le_02] - P_P[mask_le_02]) / (0.2 + epsilon)
    term_inside_sqrt = np.sum(R_values**2) / N
    if term_inside_sqrt < 0: term_inside_sqrt = 0
    return (1 - np.sqrt(term_inside_sqrt)) * 100

# ------------------------------
# 网络层定义 (Layer Definition)
# ------------------------------
class Layer:
    def __init__(self, n_inputs, n_outputs):
        limit = np.sqrt(6.0 / (n_inputs + n_outputs))
        self.weights = np.random.uniform(-limit, limit, (n_inputs, n_outputs))
        self.biases = np.zeros((1, n_outputs))
        self.inputs = self.weighted_sum = self.activation_output = self.dweights = self.dbiases = self.dinputs = None
        
    def forward(self, inputs):
        self.inputs = inputs
        self.weighted_sum = np.dot(inputs, self.weights) + self.biases
        return self.weighted_sum
    
    def backward(self, dvalues, activation_derivative_func):
        delta = dvalues * activation_derivative_func(self.weighted_sum)
        self.dweights = np.dot(self.inputs.T, delta)
        self.dbiases = np.sum(delta, axis=0, keepdims=True)
        self.dinputs = np.dot(delta, self.weights.T); return self.dinputs

# ------------------------------
# 神经网络定义 (Network Definition)
# ------------------------------
class Network:
    def __init__(self, network_shape):
        self.layers = []
        self.loss_history = []

        for i in range(len(network_shape) - 1):
            self.layers.append(Layer(network_shape[i], network_shape[i+1]))
    def forward(self, inputs):
        current_output = inputs
        for i, layer in enumerate(self.layers):
            weighted_sum = layer.forward(current_output)
            current_output = activation_relu(weighted_sum) if i < len(self.layers) - 1 else activation_linear(weighted_sum)
            layer.activation_output = current_output
        return current_output
    
    def backward(self, y_pred, y_true):
        dvalues = loss_mse_derivative(y_pred, y_true)
        for i in reversed(range(len(self.layers))):
            activation_deriv_func = activation_linear_derivative if i == len(self.layers) - 1 else activation_relu_derivative
            dvalues = self.layers[i].backward(dvalues, activation_deriv_func)
            
    def _update_parameters(self, learning_rate):#梯度下降法
        for layer in self.layers:
            layer.weights -= learning_rate * layer.dweights
            layer.biases -= learning_rate * layer.dbiases
            
    def train(self, X_train, y_train, epochs, batch_size, learning_rate):
        n_samples = X_train.shape[0] #训练样本数量
        self.loss_history = [] #记录每轮的平均损失
        print(f"\n--- 开始训练 (Epochs: {epochs}, Batch Size: {batch_size}, LR: {learning_rate}) ---")
        
        for epoch in range(epochs):
            epoch_loss = 0
            #打乱训练数据顺序
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_train_shuffled, y_train_shuffled = X_train[indices], y_train[indices]
            #按批次处理
            for i in range(0, n_samples, batch_size):
                X_batch, y_batch = X_train_shuffled[i:i+batch_size], y_train_shuffled[i:i+batch_size]
                y_pred_batch = self.forward(X_batch) #前向传播计算预测值
                epoch_loss += loss_mse(y_pred_batch, y_batch) * X_batch.shape[0] #累加损失
                # 反向传播 计算梯度 更新参数
                self.backward(y_pred_batch, y_batch)
                self._update_parameters(learning_rate)
            # 计算当前轮次平均loss    
            avg_epoch_loss = epoch_loss / n_samples; self.loss_history.append(avg_epoch_loss)
            # 每10轮输出一次Loss
            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1: 
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.6f}")
        print("--- 训练完成 ---")
        return self.loss_history
    
    def predict(self, X): 
        return self.forward(X)

# --- 绘图函数 ---
def plot_loss(history):
    """绘制训练过程中的损失变化。"""
    plt.figure(figsize=(10, 6)); plt.plot(history, label='Training Loss'); plt.title('Model Loss During Training')
    plt.xlabel('Epoch'); plt.ylabel('Mean Squared Error (MSE)'); plt.legend(); plt.grid(True); plt.show()

def plot_predictions(y_true_actual, y_pred_actual, dataset_name="Train", max_points=None, 
                     mse_value=None, cr_value=None, network_shape_str=None, learning_rate_val=None,
                     epochs_val=None, batch_size_val=None, window_size_val=None, forecast_step_val=None,
                     start_point_display_offset=0):
    """
    绘制真实值与预测值的对比图。
    max_points=None 会绘制所有点。
    """
    plt.figure(figsize=(15, 9))
    y_true_actual, y_pred_actual = y_true_actual.flatten(), y_pred_actual.flatten()
    num_points = len(y_true_actual)
    # 是否绘制全部的点。这里全绘制的效果其实一般，不太看得清细节。如果是局部点的绘制看得更清楚
    if max_points is None:
        points_to_plot = num_points
    else:
        if num_points < max_points:
            points_to_plot = num_points
        else:
            points_to_plot = max_points
    x_axis_values = np.arange(points_to_plot) 
    
    plt.plot(x_axis_values, y_true_actual[:points_to_plot], label='Actual Power', color='blue', linewidth=1.5) # 实际值折线图样式
    plt.plot(x_axis_values, y_pred_actual[:points_to_plot], label='Predicted Power', color='red', linestyle='--', linewidth=1.5) # 预测值的折线图样式
    
    title_base = f'Wind Power Prediction - {dataset_name} (Target Step: {forecast_step_val}, Points: {points_to_plot})'
    plt.xlabel('Data Point Index') 
    plt.ylabel('Normalized Power')
    
    num_ticks = min(10, points_to_plot)
    if points_to_plot > 0 :
        tick_indices_plot = np.linspace(0, points_to_plot - 1, num=num_ticks, dtype=int)
        tick_labels_actual = [str(int(idx + start_point_display_offset)) for idx in tick_indices_plot]
        plt.xticks(ticks=tick_indices_plot, labels=tick_labels_actual, rotation=45)
    
    plt.legend(loc='upper right'); plt.grid(True)
    info_lines = []
    if mse_value is not None: info_lines.append(f"MSE: {mse_value:.4f}")
    if cr_value is not None: info_lines.append(f"C_R: {cr_value:.2f}%")
    if window_size_val is not None: info_lines.append(f"Input Dim: {window_size_val}") # Changed to Input Dim
    
    if info_lines: plt.title(title_base + "\n" + " | ".join(info_lines), fontsize=10)
    else: plt.title(title_base)
    plt.subplots_adjust(bottom=0.15, top=0.85)
    
    figures_dir = 'figures'; os.makedirs(figures_dir, exist_ok=True); timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    safe_dataset_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in dataset_name)
    
    # 根据特征选择状态修改文件名
    if ENABLE_FEATURE_SELECTION:
        feature_selection_tag = CORRELATION_METHOD
    else:
        feature_selection_tag = "feature_unselected"
   #命名规则    
    plot_filename = os.path.join(figures_dir, f"{safe_dataset_name.replace(' ', '_').lower()}_{feature_selection_tag}_{timestamp}.png")
    plt.savefig(plot_filename); print(f"预测结果图已保存为: {plot_filename}"); plt.show()

# --- 主执行函数 ---
def main():
    """主函数：加载数据、训练模型、评估和绘图。"""
    # 计算预测时间（小时）
    prediction_hours = TARGET_FORECAST_STEP * 15 / 60
    print(f"===== 开始风电功率预测任务 ({prediction_hours:.2f}小时 / {TARGET_FORECAST_STEP}步 提前预测) =====")

    # 第一步：加载和预处理训练数据
    print("正在加载训练数据...")
    train_data_result = load_and_preprocess_data(TRAIN_FILE_PATH, TARGET_COLUMN, WINDOW_SIZE, TARGET_FORECAST_STEP)
    X_train = train_data_result[0]  # 训练输入数据
    y_train = train_data_result[1]  # 训练目标数据
    input_dim_initial = train_data_result[3]  # 初始输入维度
    output_dim = train_data_result[4]  # 输出维度

    # 初始化网络输入维度和特征选择索引
    input_dim_for_network = input_dim_initial
    selected_indices = None

    # 第二步：特征选择（如果启用的话）这里选取的特征数据点还是重新按照时间排序的
    if ENABLE_FEATURE_SELECTION:
        print("开始进行特征选择...")
        try:
            # 重新加载原始数据用于相关性分析
            print("重新加载原始数据...")
            df_raw = pd.read_excel(TRAIN_FILE_PATH, header=None, usecols=[0])
            # 重命名列
            df_raw.rename(columns={0: TARGET_COLUMN}, inplace=True)

            # 复制数据并处理缺失值
            raw_data = df_raw[TARGET_COLUMN].copy()
            # 向前填充缺失值
            raw_data.fillna(method='ffill', inplace=True)
            # 向后填充缺失值
            raw_data.fillna(method='bfill', inplace=True)
            # 剩余缺失值用0填充
            raw_data.fillna(0, inplace=True)

            # 转换为数值类型并删除无效值
            raw_data = pd.to_numeric(raw_data, errors='coerce').dropna().values
            # 将负数设置为0
            raw_data = np.where(raw_data < 0, 0, raw_data)

            # 分析时间滞后相关性
            print("分析时间滞后相关性...")
            correlation_result = analyze_time_lag_correlation(raw_data, WINDOW_SIZE, TARGET_FORECAST_STEP, CORRELATION_METHOD)
            selected_indices = correlation_result[1]

            # 检查特征选择结果
            if selected_indices is not None:
                if selected_indices.size > 0 and selected_indices.size < WINDOW_SIZE:
                    print(f"特征选择成功，从{WINDOW_SIZE}个特征中选择了{selected_indices.size}个特征")
                    # 应用特征选择
                    X_train = apply_feature_selection(X_train, selected_indices)
                    input_dim_for_network = X_train.shape[1]
                else:
                    print("特征选择结果不符合条件，使用全部特征")
                    selected_indices = None
                    input_dim_for_network = input_dim_initial
            else:
                print("特征选择失败，使用全部特征")
                selected_indices = None
                input_dim_for_network = input_dim_initial

        except Exception as e:
            print("特征选择过程中出现错误，使用全部特征")
            selected_indices = None
            input_dim_for_network = input_dim_initial
    else:
        print("特征选择已禁用。")
        input_dim_for_network = input_dim_initial

    # 第三步：更新神经网络结构
    print("配置神经网络结构...")

    # 检查并更新网络输入维度
    if NETWORK_SHAPE[0] != input_dim_for_network:
        print(f"更新网络输入维度: {NETWORK_SHAPE[0]} -> {input_dim_for_network}")
        NETWORK_SHAPE[0] = input_dim_for_network

    # 检查并更新网络输出维度
    if NETWORK_SHAPE[-1] != PREDICTION_STEPS:
        NETWORK_SHAPE[-1] = PREDICTION_STEPS
        print(f"更新网络输出维度为: {PREDICTION_STEPS}")

    # 第四步：创建和训练神经网络
    print("创建神经网络...")
    network = Network(NETWORK_SHAPE)
    print(f"网络结构: {NETWORK_SHAPE}")

    print("开始训练神经网络...")
    training_history = network.train(X_train, y_train, EPOCHS, BATCH_SIZE, LEARNING_RATE)

    # 绘制训练损失曲线
    print("绘制训练损失曲线...")
    plot_loss(training_history)

    # 第五步：在训练集上评估模型性能
    print("\n--- 训练集评估 ---")
    print("在训练集上进行预测...")
    train_predictions = network.predict(X_train)

    # 计算训练集评估指标
    train_mse = loss_mse(train_predictions, y_train)
    train_cr = calculate_cr_accuracy(y_train, train_predictions)

    print(f"训练集 MSE: {train_mse:.4f}, C_R: {train_cr:.2f}% (预测未来{TARGET_FORECAST_STEP}步 vs 实际未来{TARGET_FORECAST_STEP}步)")

    # 绘制训练集预测结果图
    train_dataset_name = f"Train_Step{TARGET_FORECAST_STEP}"
    plot_predictions(y_train, train_predictions,
                     dataset_name=train_dataset_name,
                     mse_value=train_mse,
                     cr_value=train_cr,
                     window_size_val=input_dim_for_network,
                     forecast_step_val=TARGET_FORECAST_STEP,
                     start_point_display_offset=1,
                     max_points=None)

    # 第六步：加载和预处理测试数据
    print(f"\n--- 加载和预处理测试数据: {TEST_FILE_PATH} ---")

    # 读取测试数据Excel文件
    df_test_raw = pd.read_excel(TEST_FILE_PATH, header=None, usecols=[0])
    # 重命名列
    df_test_raw.rename(columns={0: TARGET_COLUMN}, inplace=True)

    # 复制测试数据并处理缺失值
    test_series_raw = df_test_raw[TARGET_COLUMN].copy()
    # 向前填充缺失值
    test_series_raw.fillna(method='ffill', inplace=True)
    # 向后填充缺失值
    test_series_raw.fillna(method='bfill', inplace=True)
    # 剩余缺失值用0填充
    test_series_raw.fillna(0, inplace=True)

    # 转换为数值类型并删除无效值，然后展平为一维数组
    test_data_scaled = pd.to_numeric(test_series_raw, errors='coerce').dropna().values.flatten()
    print(f"测试数据长度: {len(test_data_scaled)}")

    # 第七步：在测试集上进行滚动预测
    print("\n--- 在测试集上进行滚动预测 ---")

    # 计算目标步骤的索引（从0开始）
    target_step_index = TARGET_FORECAST_STEP - 1
    # 初始化存储预测结果和实际值
    test_predictions_list = []
    test_actual_values_list = []
    first_actual_index = -1

    # 计算可以进行多少次预测
    total_predictions_possible = len(test_data_scaled) - WINDOW_SIZE - target_step_index
    print(f"可以进行 {total_predictions_possible} 次预测")


    # 开始滚动预测循环
    for i in range(total_predictions_possible):
        # 提取当前的输入窗口数据
        current_window_start = i
        current_window_end = i + WINDOW_SIZE
        current_input_window = test_data_scaled[current_window_start : current_window_end]
        # 重塑为网络需要的形状 (1, WINDOW_SIZE)
        current_input_window = current_input_window.reshape(1, WINDOW_SIZE)

        # 准备输入给模型的数据
        model_input = current_input_window

        # 如果启用了特征选择，则应用特征选择
        if ENABLE_FEATURE_SELECTION and selected_indices is not None and selected_indices.size > 0:
            # 检查特征索引是否有效
            max_feature_index = np.max(selected_indices)
            if max_feature_index < current_input_window.shape[1]:
                model_input = apply_feature_selection(current_input_window, selected_indices)

        # 使用神经网络进行预测
        prediction_result = network.predict(model_input)
        predicted_value = prediction_result[0, 0]
        test_predictions_list.append(predicted_value)

        # 计算对应的实际值索引
        actual_value_index = i + WINDOW_SIZE + target_step_index
        actual_value = test_data_scaled[actual_value_index]
        test_actual_values_list.append(actual_value)

        # 记录第一个实际值的索引（用于绘图显示）
        if first_actual_index == -1:
            first_actual_index = actual_value_index + 1  # 转换为1-based索引用于显示

    # 将列表转换为numpy数组
    test_predictions_array = np.array(test_predictions_list)
    test_actual_values_array = np.array(test_actual_values_list)



    # 第八步：评估测试集预测结果
    if len(test_predictions_array) > 0 and len(test_actual_values_array) == len(test_predictions_array):
        print("\n--- 测试集评估 ---")

        # 计算测试集评估指标
        test_mse = loss_mse(test_predictions_array, test_actual_values_array)
        test_cr = calculate_cr_accuracy(test_actual_values_array, test_predictions_array)

        print(f"测试集 MSE: {test_mse:.4f}, C_R: {test_cr:.2f}% (预测未来{TARGET_FORECAST_STEP}步 vs 实际未来{TARGET_FORECAST_STEP}步)")

        # 绘制测试集预测结果图
        test_dataset_name = f"Test_Step{TARGET_FORECAST_STEP}"

        # 确定绘图起始点偏移量
        if first_actual_index != -1:
            plot_start_offset = first_actual_index
        else:
            plot_start_offset = 1

        plot_predictions(test_actual_values_array, test_predictions_array,
                         dataset_name=test_dataset_name,
                         mse_value=test_mse,
                         cr_value=test_cr,
                         window_size_val=input_dim_for_network,
                         forecast_step_val=TARGET_FORECAST_STEP,
                         start_point_display_offset=plot_start_offset,
                         max_points=None)

    # 任务完成提示
    final_prediction_hours = TARGET_FORECAST_STEP * 15 / 60
    print(f"\n===== {final_prediction_hours:.2f}小时 / {TARGET_FORECAST_STEP}步 提前预测任务完成 =====")

# 程序入口点
if __name__ == "__main__":
    # 运行主函数
    main()
