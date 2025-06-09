import os
import numpy as np
import torch
from torch import nn
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator
from qiskit.quantum_info import Statevector
from PIL import Image
import joblib
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# 配置参数
IMAGE_SIZE = (8, 8)  # 输入图像尺寸
NUM_QUBITS = 8  # 量子比特数

def load_quantum_data(folder_path, prefix, max_samples=60):
    """加载测试数据"""
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

    return np.array(data_list)

def preprocess_image(img_path):
    """图像预处理函数"""
    img = Image.open(img_path).convert('L')
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img, dtype=np.float32)

    # 基础归一化
    normalized = img_array / 255.0

    # 提取多尺度特征
    flat = normalized.flatten()

    # 添加梯度特征
    grad_x = np.gradient(normalized, axis=1).flatten()
    grad_y = np.gradient(normalized, axis=0).flatten()

    # 添加局部统计特征
    local_mean = np.mean(normalized)
    local_std = np.std(normalized)
    local_features = np.array([local_mean, local_std])

    # 合并特征并选择前256个（2^8）
    combined_features = np.concatenate([
        flat,  # 64维
        grad_x[:32],  # 32维梯度特征
        grad_y[:32],  # 32维梯度特征
        np.tile(local_features, 64)  # 重复统计特征到128维
    ])

    # 确保特征数量为256（2^8）
    if len(combined_features) > 256:
        combined_features = combined_features[:256]
    elif len(combined_features) < 256:
        padding = np.zeros(256 - len(combined_features))
        combined_features = np.concatenate([combined_features, padding])

    # 归一化为单位向量
    return combined_features / (np.linalg.norm(combined_features) + 1e-8)

def quantum_amplitude_encoding(img_vector, target_qubits=8):
    """量子振幅编码"""
    target_length = 2 ** target_qubits

    if len(img_vector) > target_length:
        img_vector = img_vector[:target_length]
    elif len(img_vector) < target_length:
        padding = np.zeros(target_length - len(img_vector))
        img_vector = np.concatenate([img_vector, padding])

    normalized_vector = img_vector / (np.linalg.norm(img_vector) + 1e-8)

    try:
        state = Statevector(normalized_vector)
        complex_data = state.data.astype(np.complex128)

        # 相位规范化
        phase = np.angle(complex_data)
        phase_normalized = (phase + np.pi) % (2 * np.pi) - np.pi

        # 重构复数向量
        encoded = np.abs(complex_data) * np.exp(1j * phase_normalized)
        return encoded

    except Exception as e:
        print(f"量子编码错误: {e}")
        return normalized_vector.astype(np.complex128)

def process_image_to_quantum(img_path, label, output_dir):
    """处理单张图像并保存量子编码结果"""
    try:
        # 预处理图像
        processed_vector = preprocess_image(img_path)
        
        # 量子编码
        quantum_encoded = quantum_amplitude_encoding(processed_vector, NUM_QUBITS)
        
        # 生成输出文件名
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        filename_out = f"{label}_{base_name}_orig.npy"
        filepath_out = os.path.join(output_dir, filename_out)
        
        # 保存量子编码数据
        np.save(filepath_out, quantum_encoded)
        return True
        
    except Exception as e:
        print(f"处理 {img_path} 时出错: {str(e)}")
        return False

def process_directory(image_dir, label, output_dir):
    """处理整个目录的图像"""
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    successful_encodings = 0

    image_files = [f for f in os.listdir(image_dir)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"处理 {len(image_files)} 个 {label} 图像...")

    for filename in tqdm(image_files, desc=f"处理{label}图像"):
        img_path = os.path.join(image_dir, filename)
        if process_image_to_quantum(img_path, label, output_dir):
            successful_encodings += 1
        count += 1

    print(f"完成 {label} 处理: {successful_encodings} 个成功编码的样本")
    return successful_encodings

def advanced_feature_encoder(X, amplification_factor=3):
    """特征编码器"""
    amplitudes = np.abs(X)
    phases = np.angle(X)

    phase_median = np.median(phases, axis=1, keepdims=True)
    phase_std = np.std(phases, axis=1, keepdims=True) + 1e-8
    phases_normalized = (phases - phase_median) / phase_std
    phases_enhanced = np.tanh(phases_normalized * amplification_factor)

    log_amplitudes = np.log1p(amplitudes)
    normalized_amplitudes = amplitudes / (np.max(amplitudes, axis=1, keepdims=True) + 1e-8)
    phase_gradients = np.diff(phases, axis=1, prepend=phases[:, :1])
    amplitude_phase_product = amplitudes * np.cos(phases)

    amplitude_mean = np.mean(amplitudes, axis=1, keepdims=True)
    amplitude_var = np.var(amplitudes, axis=1, keepdims=True)
    phase_coherence = np.abs(np.mean(np.exp(1j * phases), axis=1, keepdims=True))

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
    """创建量子电路"""
    qc = QuantumCircuit(num_qubits)
    x_params = [Parameter(f'x{i}') for i in range(num_qubits)]
    theta_params = [Parameter(f'theta{i}') for i in range(6 * num_qubits)]

    for i in range(num_qubits):
        qc.h(i)
        qc.ry(x_params[i], i)

    param_idx = 0
    for layer in range(3):
        for i in range(num_qubits):
            qc.cx(i, (i + 1) % num_qubits)

        for i in range(num_qubits):
            qc.ry(theta_params[param_idx], i)
            qc.rz(theta_params[param_idx + num_qubits], i)
            param_idx += 1

        param_idx += num_qubits

    return qc

class OptimizedQuantumClassifier(nn.Module):
    def __init__(self, qnn, input_dim, num_qubits):
        super().__init__()
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

        self.quantum_adapter = nn.Sequential(
            nn.Linear(128, num_qubits * 2),
            nn.LayerNorm(num_qubits * 2),
            nn.Tanh(),
            nn.Linear(num_qubits * 2, num_qubits),
            nn.LayerNorm(num_qubits)
        )

        self.qnn = TorchConnector(qnn)

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

        self.residual_processor = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        processed_features = self.feature_processor(x)
        quantum_input = self.quantum_adapter(processed_features)
        quantum_output = self.qnn(quantum_input)
        quantum_result = self.post_processor(quantum_output)
        classical_result = self.residual_processor(processed_features)
        combined = quantum_result + 0.3 * classical_result
        return torch.sigmoid(combined)

def predict_images():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 处理测试图像
    print("处理测试图像...")
    real_folder = 'predict_test_images/real'
    deepfake_folder = 'predict_test_images/deepfake'
    quantum_output_dir = 'predict_test_images/quantum_encoded'

    # 处理真实图像
    real_count = process_directory(real_folder, 'real', os.path.join(quantum_output_dir, 'real'))
    
    # 处理伪造图像
    fake_count = process_directory(deepfake_folder, 'deepfake', os.path.join(quantum_output_dir, 'deepfake'))

    if real_count == 0 or fake_count == 0:
        print("错误: 没有成功处理任何测试图像")
        return

    # 加载量子编码数据
    print("加载量子编码数据...")
    real_data = load_quantum_data(os.path.join(quantum_output_dir, 'real'), 'real', max_samples=60)
    fake_data = load_quantum_data(os.path.join(quantum_output_dir, 'deepfake'), 'deepfake', max_samples=60)

    if len(real_data) == 0 or len(fake_data) == 0:
        print("错误: 没有加载到量子编码数据")
        return

    # 合并数据
    X_test = np.vstack([real_data, fake_data])
    y_test = np.array([0] * len(real_data) + [1] * len(fake_data))

    # 特征处理
    print("处理特征...")
    X_processed = advanced_feature_encoder(X_test)

    # 加载标准化器
    scaler = joblib.load('scaler_RGB.pkl')
    X_scaled = scaler.transform(X_processed)

    # 转换为Tensor
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    # 创建量子电路和QNN
    num_qubits = 8
    qc = create_optimized_circuit(num_qubits)
    estimator = Estimator()
    qnn = EstimatorQNN(
        circuit=qc,
        input_params=qc.parameters[:num_qubits],
        weight_params=qc.parameters[num_qubits:],
        estimator=estimator
    )

    # 加载模型
    print("加载模型...")
    model = OptimizedQuantumClassifier(qnn, X_scaled.shape[1], num_qubits)
    model.load_state_dict(torch.load('best_optimized_quantum_model.pth'))
    model = model.to(device)
    model.eval()

    # 预测
    print("开始预测...")
    predictions = []
    with torch.no_grad():
        for i in tqdm(range(len(X_tensor))):
            output = model(X_tensor[i:i+1])
            pred = (output > 0.5).float().item()
            predictions.append(pred)

    # 计算准确率
    correct = sum(p == t for p, t in zip(predictions, y_test))
    accuracy = correct / len(y_test) * 100

    # 输出结果
    print("\n预测结果:")
    print(f"总样本数: {len(y_test)}")
    print(f"正确预测数: {correct}")
    print(f"准确率: {accuracy:.2f}%")

    # 详细结果
    print("\n详细预测结果:")
    for i, (pred, true) in enumerate(zip(predictions, y_test)):
        status = "正确" if pred == true else "错误"
        label = "真实" if true == 0 else "伪造"
        pred_label = "真实" if pred == 0 else "伪造"
        print(f"样本 {i+1}: 实际={label}, 预测={pred_label}, {status}")

if __name__ == "__main__":
    predict_images() 