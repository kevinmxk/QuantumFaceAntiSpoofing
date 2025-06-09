# 优化的量子神经网络深度伪造检测模型

import os
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset, random_split, WeightedRandomSampler
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator
import matplotlib.pyplot as plt
from collections import Counter
import joblib
import warnings
import time  # 添加计时模块

warnings.filterwarnings('ignore')


def load_quantum_data(folder_path, prefix, max_samples=200, augment=True):
    """改进的数据加载函数，增加更多样化的数据增强"""
    if not os.path.exists(folder_path):
        print(f"警告: 路径不存在: {folder_path}")
        return np.array([])

    files = sorted([
        f for f in os.listdir(folder_path)
        if f.startswith(prefix) and f.endswith('.npy')
    ])[:max_samples]

    if not files:
        print(f"警告: 在{folder_path}中没有找到以{prefix}开头的.npy文件")
        return np.array([])

    data_list = []
    for file in files:
        file_path = os.path.join(folder_path, file)
        data = np.load(file_path)
        data_list.append(data)

        # 更多样化的数据增强
        if augment:
            # 1. 相位扰动
            perturbed1 = data * np.exp(1j * np.random.normal(0, 0.05, data.shape))
            data_list.append(perturbed1)

            # 2. 幅度缩放
            scale_factor = np.random.uniform(0.9, 1.1, data.shape)
            perturbed2 = data * scale_factor
            data_list.append(perturbed2)

            # 3. 添加微小噪声
            noise = np.random.normal(0, 0.01, data.shape) + 1j * np.random.normal(0, 0.01, data.shape)
            perturbed3 = data + noise
            data_list.append(perturbed3)

    return np.array(data_list)


def advanced_feature_encoder(X, amplification_factor=3):
    """改进的特征编码器，提取更丰富的量子特征"""
    # 分离幅度和相位
    amplitudes = np.abs(X)
    phases = np.angle(X)

    # 1. 增强的相位特征
    phase_median = np.median(phases, axis=1, keepdims=True)
    phase_std = np.std(phases, axis=1, keepdims=True) + 1e-8
    phases_normalized = (phases - phase_median) / phase_std
    phases_enhanced = np.tanh(phases_normalized * amplification_factor)

    # 2. 对数幅度和归一化幅度
    log_amplitudes = np.log1p(amplitudes)
    normalized_amplitudes = amplitudes / (np.max(amplitudes, axis=1, keepdims=True) + 1e-8)

    # 3. 相位梯度特征（相邻元素相位差）
    phase_gradients = np.diff(phases, axis=1, prepend=phases[:, :1])

    # 4. 幅度-相位交叉特征
    amplitude_phase_product = amplitudes * np.cos(phases)

    # 5. 统计特征
    amplitude_mean = np.mean(amplitudes, axis=1, keepdims=True)
    amplitude_var = np.var(amplitudes, axis=1, keepdims=True)
    phase_coherence = np.abs(np.mean(np.exp(1j * phases), axis=1, keepdims=True))

    # 合并所有特征
    features = np.concatenate([
        log_amplitudes,
        normalized_amplitudes,
        phases_enhanced,
        phase_gradients,
        amplitude_phase_product,
        np.tile(amplitude_mean, (1, amplitudes.shape[1])),
        np.tile(amplitude_var, (1, amplitudes.shape[1])),
        np.tile(phase_coherence.real, (1, amplitudes.shape[1]))
    ], axis=1)

    return features


def create_optimized_circuit(num_qubits=8):
    """创建优化的量子电路，增加表达能力"""
    qc = QuantumCircuit(num_qubits)

    # 输入参数
    x_params = [Parameter(f'x{i}') for i in range(num_qubits)]
    # 增加参数数量以提高表达能力
    theta_params = [Parameter(f'theta{i}') for i in range(6 * num_qubits)]

    # 1. 特征映射层（Hadamard + 旋转）
    for i in range(num_qubits):
        qc.h(i)
        qc.ry(x_params[i], i)

    # 2. 多层变分量子电路
    param_idx = 0
    for layer in range(3):  # 3层足够，避免过拟合
        # 环形纠缠
        for i in range(num_qubits):
            qc.cx(i, (i + 1) % num_qubits)

        # Y和Z旋转
        for i in range(num_qubits):
            qc.ry(theta_params[param_idx], i)
            qc.rz(theta_params[param_idx + num_qubits], i)
            param_idx += 1

        param_idx += num_qubits

    return qc


class FocalLoss(nn.Module):
    """Focal Loss用于处理类别不平衡"""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class OptimizedQuantumClassifier(nn.Module):
    def __init__(self, qnn, input_dim, num_qubits):
        super().__init__()

        # 特征预处理网络
        self.feature_processor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 量子特征适配器
        self.quantum_adapter = nn.Sequential(
            nn.Linear(128, num_qubits * 2),
            nn.LayerNorm(num_qubits * 2),
            nn.Tanh(),  # 限制输入范围
            nn.Linear(num_qubits * 2, num_qubits),
            nn.LayerNorm(num_qubits)
        )

        # 量子层
        self.qnn = TorchConnector(qnn)

        # 后处理网络
        self.post_processor = nn.Sequential(
            nn.Linear(1, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

        # 残差连接
        self.residual_processor = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # 特征处理
        processed_features = self.feature_processor(x)

        # 量子计算路径
        quantum_input = self.quantum_adapter(processed_features)
        quantum_output = self.qnn(quantum_input)
        quantum_result = self.post_processor(quantum_output)

        # 经典残差路径
        classical_result = self.residual_processor(processed_features)

        # 融合输出
        combined = quantum_result + 0.3 * classical_result
        return torch.sigmoid(combined)


def train_with_early_stopping(model, train_loader, val_loader, num_epochs=50, patience=10):
    """带早停的训练函数"""
    # 使用Focal Loss处理类别不平衡
    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    # 优化器配置
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )

    # 学习率调度
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    best_val_acc = 0
    patience_counter = 0
    train_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_train_loss += loss.item()

            if batch_idx % 5 == 0:
                print(f'Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.6f}')

        # 验证阶段
        model.eval()
        correct = 0
        total = 0
        val_loss = 0

        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += criterion(output, target).item()
                predicted = (output > 0.5).float()
                total += target.size(0)
                correct += (predicted == target).sum().item()

        val_acc = 100 * correct / total
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_accuracies.append(val_acc)

        print(f'Epoch {epoch + 1}: Train Loss: {avg_train_loss:.6f}, '
              f'Val Loss: {avg_val_loss:.6f}, Val Acc: {val_acc:.2f}%')

        # 学习率调整
        scheduler.step(val_acc)

        # 早停检查
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_optimized_quantum_model_LAB.pth')
            print(f'新的最佳验证准确率: {best_val_acc:.2f}%')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'早停触发，最佳验证准确率: {best_val_acc:.2f}%')
            break

    return train_losses, val_accuracies, best_val_acc


# 主训练流程
def main():
    # 修正文件夹路径
    real_folder = 'quantum_encoded_states/v4/realhsi'
    deepfake_folder = 'quantum_encoded_states/v4/deepfakehsi'

    # 加载数据
    print("加载数据...")
    real = load_quantum_data(real_folder, 'real', max_samples=200)
    fake = load_quantum_data(deepfake_folder, 'deepfake', max_samples=200)

    print(f"Real samples: {len(real)}, Fake samples: {len(fake)}")

    if len(real) == 0 or len(fake) == 0:
        print("错误: 没有加载到数据，请检查数据路径和文件")
        return

    # 数据预处理
    X = np.vstack([real, fake])
    y = np.array([0] * len(real) + [1] * len(fake))

    print("特征编码...")
    X_processed = advanced_feature_encoder(X)

    # 使用RobustScaler，对异常值更鲁棒
    scaler = RobustScaler().fit(X_processed)
    X_scaled = scaler.transform(X_processed)

    # 保存标准化器
    joblib.dump(scaler, 'scaler_LAB.pkl')
    print("数据标准化器已保存为 scaler_LAB.pkl")

    print(f"处理后数据形状: {X_scaled.shape}")

    # 转换为Tensor
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    # 创建平衡的数据集
    dataset = TensorDataset(X_tensor, y_tensor)

    # 分层划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # 计算类别权重用于加权采样
    train_targets = [dataset[i][1].item() for i in train_ds.indices]
    class_counts = Counter(train_targets)
    class_weights = {0: len(train_targets) / class_counts[0], 1: len(train_targets) / class_counts[1]}
    sample_weights = [class_weights[int(target)] for target in train_targets]

    # 加权采样器
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # 数据加载器
    train_loader = DataLoader(train_ds, batch_size=35, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=35, shuffle=False)

    # 创建优化的量子电路
    num_qubits = 8  # 增加量子比特数
    qc = create_optimized_circuit(num_qubits)

    # 创建QNN
    estimator = Estimator()
    qnn = EstimatorQNN(
        circuit=qc,
        input_params=qc.parameters[:num_qubits],
        weight_params=qc.parameters[num_qubits:],
        estimator=estimator
    )

    # 初始化模型
    model = OptimizedQuantumClassifier(qnn, X_scaled.shape[1], num_qubits)

    print("开始训练...")
    start_time = time.time()  # 记录开始时间
    train_losses, val_accuracies, best_acc = train_with_early_stopping(
        model, train_loader, val_loader, num_epochs=25, patience=5
    )

    end_time = time.time()  # 记录结束时间
    print(f'训练耗时: {end_time - start_time:.2f} 秒')  # 输出训练耗时
    print(f'最终最佳验证准确率: {best_acc:.2f}%')

    # 加载最佳模型权重
    model.load_state_dict(torch.load('best_optimized_quantum_model_LAB.pth'))
    # 保存整个模型
    torch.save(model, 'optimized_quantum_model_full_LAB.pth')
    print("完整模型已保存为 optimized_quantum_model_full_LAB.pth")

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')

    plt.tight_layout()
    plt.savefig('training_curves_LAB.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()