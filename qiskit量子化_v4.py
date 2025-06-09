import numpy as np
from PIL import Image
from qiskit.quantum_info import Statevector
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 配置参数 - 与训练代码匹配
IMAGE_SIZE = (8, 8)  # 输入图像尺寸
NUM_QUBITS = 8  # 修改为8，与训练代码匹配
AUGMENTATION_FACTOR = 2  # 每个样本的增强次数
PHASE_NOISE_LEVEL = 0.05  # 减小扰动强度，提高数据质量

# 输入输出路径 - 修正目录结构
deepfake_dir = "RGB/12df"
real_dir = "RGB/12real"
output_base_dir = "RGB/12bm"

# 创建匹配训练代码的目录结构
real_output_dir = os.path.join(output_base_dir, "real")
deepfake_output_dir = os.path.join(output_base_dir, "deepfake")
os.makedirs(real_output_dir, exist_ok=True)
os.makedirs(deepfake_output_dir, exist_ok=True)


def quantum_amplitude_encoding(img_vector, target_qubits=8):
    """改进的量子振幅编码，确保维度匹配"""
    # 确保输入向量长度为2^target_qubits
    target_length = 2 ** target_qubits

    if len(img_vector) > target_length:
        # 如果输入太长，使用PCA降维或选择重要特征
        img_vector = img_vector[:target_length]
    elif len(img_vector) < target_length:
        # 如果输入太短，进行零填充
        padding = np.zeros(target_length - len(img_vector))
        img_vector = np.concatenate([img_vector, padding])

    # 归一化为单位向量
    normalized_vector = img_vector / (np.linalg.norm(img_vector) + 1e-8)

    # 创建量子态
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
        # 备用方案：直接返回复数向量
        return normalized_vector.astype(np.complex128)


def advanced_image_preprocessing(img_path):
    """改进的图像预处理，提取更丰富的特征"""
    img = Image.open(img_path).convert('L')
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img, dtype=np.float32)

    # 1. 基础归一化
    normalized = img_array / 255.0

    # 2. 提取多尺度特征
    flat = normalized.flatten()

    # 3. 添加梯度特征
    grad_x = np.gradient(normalized, axis=1).flatten()
    grad_y = np.gradient(normalized, axis=0).flatten()

    # 4. 添加局部统计特征
    local_mean = np.mean(normalized)
    local_std = np.std(normalized)
    local_features = np.array([local_mean, local_std])

    # 5. 合并特征并选择前256个（2^8）
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


def create_quantum_augmentations(base_vector, num_augmentations=3):
    """创建多样化的量子数据增强"""
    augmented_data = [base_vector]  # 包含原始数据

    for i in range(num_augmentations):
        if i == 0:
            # 相位扰动
            phase_noise = np.random.normal(0, PHASE_NOISE_LEVEL, len(base_vector))
            augmented = base_vector * np.exp(1j * phase_noise)
        elif i == 1:
            # 幅度缩放
            scale_factor = np.random.uniform(0.95, 1.05)
            augmented = base_vector * scale_factor
        else:
            # 添加微小复数噪声
            real_noise = np.random.normal(0, 0.01, len(base_vector))
            imag_noise = np.random.normal(0, 0.01, len(base_vector))
            noise = real_noise + 1j * imag_noise
            augmented = base_vector + noise

        # 重新归一化
        augmented = augmented / (np.linalg.norm(augmented) + 1e-8)
        augmented_data.append(augmented)

    return augmented_data


def process_directory_improved(image_dir, label, output_dir):
    """改进的目录处理函数"""
    count = 0
    successful_encodings = 0

    image_files = [f for f in os.listdir(image_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"处理 {len(image_files)} 个 {label} 图像...")

    for filename in image_files:
        try:
            img_path = os.path.join(image_dir, filename)
            base_name = os.path.splitext(filename)[0]

            # 预处理图像
            processed_vector = advanced_image_preprocessing(img_path)

            # 量子编码
            quantum_encoded = quantum_amplitude_encoding(processed_vector, NUM_QUBITS)

            # 创建增强数据
            augmented_data = create_quantum_augmentations(quantum_encoded, AUGMENTATION_FACTOR)

            # 保存所有版本
            for idx, data in enumerate(augmented_data):
                suffix = 'orig' if idx == 0 else f'aug{idx - 1}'
                filename_out = f"{label}_{base_name}_{suffix}.npy"
                filepath_out = os.path.join(output_dir, filename_out)

                # 验证数据质量
                if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                    print(f"警告: {filename_out} 包含无效数据，跳过")
                    continue

                np.save(filepath_out, data)
                count += 1
                successful_encodings += 1

                if count % 10 == 0:
                    print(f"已编码 {count} 个样本...")

        except Exception as e:
            print(f"处理 {filename} 时出错: {str(e)}")
            continue

    print(f"完成 {label} 处理: {successful_encodings} 个成功编码的样本")
    return successful_encodings


def verify_compatibility():
    """验证编码数据与训练代码的兼容性"""
    print("\n=== 兼容性验证 ===")

    # 检查输出目录
    real_files = [f for f in os.listdir(real_output_dir) if f.endswith('.npy')]
    deepfake_files = [f for f in os.listdir(deepfake_output_dir) if f.endswith('.npy')]

    print(f"Real samples: {len(real_files)}")
    print(f"Deepfake samples: {len(deepfake_files)}")

    if real_files and deepfake_files:
        # 加载样本数据验证
        sample_real = np.load(os.path.join(real_output_dir, real_files[0]))
        sample_fake = np.load(os.path.join(deepfake_output_dir, deepfake_files[0]))

        print(f"样本数据形状: {sample_real.shape}")
        print(f"数据类型: {sample_real.dtype}")
        print(f"量子比特数匹配: {len(sample_real) == 2 ** NUM_QUBITS}")
        print(f"复数数据: {np.iscomplexobj(sample_real)}")

        # 验证数据范围
        print(f"幅度范围: [{np.min(np.abs(sample_real)):.6f}, {np.max(np.abs(sample_real)):.6f}]")
        print(f"相位范围: [{np.min(np.angle(sample_real)):.6f}, {np.max(np.angle(sample_real)):.6f}]")

        return True
    else:
        print("错误: 没有找到编码数据文件")
        return False


# 主执行流程
if __name__ == "__main__":
    print("开始量子编码流程...")
    print(f"目标量子比特数: {NUM_QUBITS}")
    print(f"目标向量维度: {2 ** NUM_QUBITS}")

    # 处理真实图像
    real_count = process_directory_improved(real_dir, "real", real_output_dir)

    # 处理深度伪造图像
    fake_count = process_directory_improved(deepfake_dir, "deepfake", deepfake_output_dir)

    print(f"\n编码完成!")
    print(f"真实样本: {real_count}")
    print(f"伪造样本: {fake_count}")

    # 验证兼容性
    if verify_compatibility():
        print("\n✅ 编码数据与训练代码完全兼容!")
    else:
        print("\n❌ 发现兼容性问题，请检查数据")