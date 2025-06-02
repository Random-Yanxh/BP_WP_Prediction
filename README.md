# 风电功率预测项目

## 项目简介

本项目旨在通过历史风电数据预测未来的风电功率。项目采用基于神经网络的方法，并包含数据预处理、特征选择、模型训练、评估和结果可视化等模块。

## 功能特性

- **数据加载与预处理**:
    - 支持从 Excel 文件 (`.xlsx`) 加载数据。
    - 对数据进行清洗，包括缺失值填充（向前填充、向后填充、0值填充）和类型转换。
    - 将原始数据转换为适用于时间序列预测的滑动窗口数据集。
- **特征选择**:
    - 可选的时间滞后相关性分析，用于选择与目标预测值最相关的历史数据点作为模型输入特征。
    - 支持多种相关性计算方法：Pearson, Spearman, Kendall。
    - 用户可以配置是否启用特征选择以及选择的特征数量。
- **神经网络模型**:
    - 实现了一个可自定义层数和神经元数量的前馈神经网络。
    - 激活函数：隐藏层使用 ReLU，输出层使用线性激活函数。
    - 损失函数：均方误差 (MSE)。
- **模型训练与预测**:
    - 支持批量训练 (Batch Training)。
    - 可配置学习率、训练轮数 (Epochs) 和批次大小 (Batch Size)。
    - 训练过程中记录并可以绘制损失变化曲线。
    - 支持在训练集和测试集上进行预测。测试集采用滚动预测方式。
- **模型评估**:
    - 使用均方误差 (MSE) 和自定义的准确率指标 C_R (Coefficient of Residuals) 来评估模型性能。
- **结果可视化**:
    - 绘制训练过程中的损失函数变化图。
    - 绘制实际功率与预测功率的对比图，直观展示预测效果。
    - 图像文件名根据数据集类型、预测步长、是否进行特征选择以及所用方法（如果启用）和时间戳自动生成，并保存在 `figures` 目录下。

## 项目依赖

本项目基于 `Python 3.10`构建，主要依赖以下 Python 库：

- **NumPy**: 用于科学计算，特别是多维数组操作。
- **Pandas**: 用于数据处理和分析，特别是表格数据操作。
- **Matplotlib**: 用于数据可视化，绘制图表。
- **SciPy**: 用于科学和技术计算，本项目中主要使用其统计模块 (`scipy.stats`) 进行相关性分析。

您可以使用 pip 来安装这些依赖：
```bash
pip install numpy pandas matplotlib scipy
```
## 技术实现

### 1. 数据处理 (`load_and_preprocess_data`, `create_sliding_window_dataset`)

- **数据加载**: 使用 `pandas` 读取 Excel 文件。
- **缺失值处理**:
    - `ffill()`: 向前填充。
    - `bfill()`: 向后填充。
    - `fillna(0)`: 用0填充剩余缺失。
- **滑动窗口**:
    - `create_sliding_window_dataset` 函数将时间序列数据转换为 (X, y) 对。
    - `X` 是由 `WINDOW_SIZE` 个连续历史数据点组成的窗口。
    - `y` 是在 `TARGET_FORECAST_STEP` 步之后的目标预测值。

### 2. 特征工程 (`analyze_time_lag_correlation`, `apply_feature_selection`)

- **启用/禁用**: 通过 `ENABLE_FEATURE_SELECTION` (布尔值) 控制。
- **相关性分析**:
    - `analyze_time_lag_correlation` 函数计算输入窗口中每个历史数据点与未来目标值之间的相关性。
    - `CORRELATION_METHOD` 参数指定相关性计算方法 ('pearson', 'kendall', 'spearman')。
    - `SELECTED_FEATURES_COUNT` 参数指定选择最相关的特征数量。
- **应用选择**:
    - `apply_feature_selection` 函数根据选择的特征索引从输入数据 `X` 中提取相应的特征。

### 3. 模型结构 (`Layer`, `Network`)

- **Layer 类**: 定义单个神经网络层。
    - 权重初始化：Xavier/Glorot 初始化 (`np.sqrt(6.0 / (n_inputs + n_outputs))`)。
    - 包含前向传播 (`forward`) 和反向传播 (`backward`) 方法。
- **Network 类**: 组织多个 `Layer` 构成神经网络。
    - `network_shape` 参数定义网络结构，例如 `[input_dim, 16, output_dim]` 表示一个输入层、一个包含16个神经元的隐藏层和一个输出层。
    - `forward` 方法执行完整的前向传播。
    - `backward` 方法执行完整的反向传播。
    - `_update_parameters` 方法使用梯度下降更新权重和偏置。
    - `train` 方法执行完整的训练流程。
    - `predict` 方法用于生成预测。

### 4. 激活函数与损失函数

- **激活函数**:
    - `activation_relu`: ReLU 激活函数 (`np.maximum(0, inputs)`)。
    - `activation_linear`: 线性激活函数 (`inputs`)。
    - 对应的导数函数用于反向传播。
- **损失函数**:
    - `loss_mse`: 均方误差 (`np.mean((y_pred - y_true) ** 2)`)。
    - `loss_mse_derivative`: MSE 的导数。

### 5. 评估指标 (`calculate_cr_accuracy`)

- **C_R (Coefficient of Residuals)**: 一个自定义的准确率评估指标，具体计算方式见代码内函数。

### 6. 绘图 (`plot_loss`, `plot_predictions`)

- **`plot_loss`**: 绘制训练期间的 MSE 损失随 Epoch 变化的曲线。
- **`plot_predictions`**:
    - 绘制真实值与预测值的对比图。
    - 图表标题包含数据集名称、目标预测步长、点数、MSE、C_R 和输入维度等信息。
    - 图像自动保存到 `figures` 目录，文件名包含数据集、预测步长、特征选择信息（方法或"unselected"）和时间戳。

## 如何运行

1.  **准备数据**:
    - 将训练数据放入 `train.xlsx` 文件。
    - 将测试数据放入 `test.xlsx` 文件。
    - 数据应为单列，代表实际功率值。
2.  **配置参数**:
    - 打开 `MyBP_preProcess_final.py` 文件。
    - 根据需求修改文件顶部的宏定义部分，例如：
        - `PREDICTION_STEPS`: 每次预测的时间点数量。
        - `TARGET_FORECAST_STEP`: 提前预测的步数（1步 = 15分钟）。
        - `NETWORK_SHAPE`: 神经网络结构。
        - `BATCH_SIZE`, `LEARNING_RATE`, `EPOCHS`: 训练参数。
        - `WINDOW_SIZE`: 滑动窗口大小。
        - `ENABLE_FEATURE_SELECTION`: 是否启用特征选择。
        - `SELECTED_FEATURES_COUNT`: 选择的特征数量。
        - `CORRELATION_METHOD`: 相关性计算方法。
        - `TRAIN_FILE_PATH`, `TEST_FILE_PATH`: 数据文件路径。
        - `TARGET_COLUMN`: 目标列名（通常为 '实际功率'）。
3.  **执行脚本**:
    ```bash
    python MyBP_preProcess_final.py
    ```
4.  **查看结果**:
    - 训练过程中的损失和评估指标会打印到控制台。
    - 生成的图像会保存在项目根目录下的 `figures` 文件夹中。

## 文件说明

- **`MyBP_preProcess_final.py`**: 主脚本文件，包含所有代码逻辑。
- **`train.xlsx`**: 训练数据文件（需要用户提供）。
- **`test.xlsx`**: 测试数据文件（需要用户提供）。
- **`figures/`**: 存放预测结果图和损失曲线图的目录（自动创建）。

## 主要可配置参数 (在 `MyBP_preProcess_final.py` 中)

- `PREDICTION_STEPS`: 预测未来多少个时间点。
- `TARGET_FORECAST_STEP`: 提前多少步进行预测（例如，1 表示提前15分钟）。
- `NETWORK_SHAPE`: 定义神经网络的层数和每层的神经元数量。例如 `[None, 16, PREDICTION_STEPS]`，其中 `None` 会被自动替换为输入特征维度。
- `BATCH_SIZE`: 每次训练迭代中使用的样本数量。
- `LEARNING_RATE`: 学习率，控制模型参数更新的幅度。
- `EPOCHS`: 训练的总轮数。
- `WINDOW_SIZE`: 用于预测的历史数据点数量。
- `ENABLE_FEATURE_SELECTION`: 布尔值，`True` 启用特征选择，`False` 禁用。
- `SELECTED_FEATURES_COUNT`: 如果启用特征选择，选择多少个最相关的特征。
- `CORRELATION_METHOD`: 特征选择时使用的相关性计算方法，可选 `'pearson'`, `'kendall'`, `'spearman'`。