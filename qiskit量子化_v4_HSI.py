import numpy as np  # 导入数值计算库
from PIL import Image  # 导入图像处理库
from qiskit.quantum_info import Statevector  # 导入 Qiskit 的量子态向量类
import os  # 导入操作系统交互库
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # 导入数据预处理工具
from skimage import color  # 导入颜色空间转换模块
from concurrent.futures import ProcessPoolExecutor

# 配置参数
IMAGE_SIZE = (8, 8)  # 图像缩放尺寸
NUM_QUBITS = 8  # 使用的量子比特数
AUGMENTATION_FACTOR = 2  # 数据增强因子
PHASE_NOISE_LEVEL = 0.05  # 相位噪声水平

# 输入输出路径
input_dirs = {
    "real": "output_images/real_v2",  # 真实图像目录
    "deepfake": "output_images/deepfake_v2"  # 深度伪造图像目录
}
output_base_dir = "quantum_encoded_states/v4"  # 量子编码输出目录


# 工具函数

def pad_or_truncate(vec, target_length):
    """
    对向量进行填充或截断以达到目标长度
    :param vec: 原始向量
    :param target_length: 目标长度
    :return: 调整后的向量
    """
    if len(vec) > target_length:
        vec = vec[:target_length]  # 截断
    elif len(vec) < target_length:
        vec = np.concatenate([vec, np.zeros(target_length - len(vec))])  # 填充
    return vec / (np.linalg.norm(vec) + 1e-8)  # 归一化


def preprocess_hsi_image(img_path):
    """
    预处理 HSI 格式的图像
    :param img_path: 图像文件路径
    :return: 处理后的向量
    """
    img = Image.open(img_path).convert('RGB')  # 打开并转换为 RGB 格式
    img = img.resize(IMAGE_SIZE)  # 缩放图像
    img_array = np.array(img, dtype=np.float32) / 255.0  # 归一化像素值
    hsi = color.rgb2hsv(img_array)  # 转换为 HSV 颜色空间
    return pad_or_truncate(hsi.transpose(2, 0, 1).reshape(-1), 256)  # 调整维度并填充或截断


def preprocess_lab_image(img_path):
    """
    预处理 LAB 格式的图像
    :param img_path: 图像文件路径
    :return: 处理后的向量
    """
    img = Image.open(img_path).convert('RGB')  # 打开并转换为 RGB 格式
    img = img.resize(IMAGE_SIZE)  # 缩放图像
    img_array = np.array(img, dtype=np.float32) / 255.0  # 归一化像素值
    lab = color.rgb2lab(img_array)  # 转换为 LAB 颜色空间
    return pad_or_truncate(lab.transpose(2, 0, 1).reshape(-1), 256)  # 调整维度并填充或截断


def quantum_amplitude_encoding(img_vector, target_qubits=8):
    """
    量子振幅编码
    :param img_vector: 图像向量
    :param target_qubits: 目标量子比特数
    :return: 编码后的量子态
    """
    target_length = 2 ** target_qubits  # 计算目标长度（2^qubits）
    if len(img_vector) > target_length:
        img_vector = img_vector[:target_length]  # 截断
    elif len(img_vector) < target_length:
        padding = np.zeros(target_length - len(img_vector))  # 填充零
        img_vector = np.concatenate([img_vector, padding])

    normalized_vector = img_vector / (np.linalg.norm(img_vector) + 1e-8)  # 归一化

    try:
        state = Statevector(normalized_vector)  # 创建量子态
        complex_data = state.data.astype(np.complex128)  # 转换为复数类型
        phase = np.angle(complex_data)  # 获取相位
        phase_normalized = (phase + np.pi) % (2 * np.pi) - np.pi  # 规范化相位
        encoded = np.abs(complex_data) * np.exp(1j * phase_normalized)  # 应用相位编码
        return encoded
    except Exception as e:
        print(f"量子编码错误: {e}")
        return normalized_vector.astype(np.complex128)


def create_quantum_augmentations(base_vector, num_augmentations=3):
    """
    创建量子数据增强
    :param base_vector: 基础向量
    :param num_augmentations: 增强数量
    :return: 增强后的数据列表
    """
    augmented_data = [base_vector]  # 初始向量
    for i in range(num_augmentations):
        if i == 0:
            phase_noise = np.random.normal(0, PHASE_NOISE_LEVEL, len(base_vector))  # 相位噪声
            augmented = base_vector * np.exp(1j * phase_noise)
        elif i == 1:
            scale_factor = np.random.uniform(0.95, 1.05)  # 缩放因子
            augmented = base_vector * scale_factor
        else:
            real_noise = np.random.normal(0, 0.01, len(base_vector))  # 实部噪声
            imag_noise = np.random.normal(0, 0.01, len(base_vector))  # 虚部噪声
            noise = real_noise + 1j * imag_noise  # 合成噪声
            augmented = base_vector + noise

        augmented = augmented / (np.linalg.norm(augmented) + 1e-8)  # 归一化
        augmented_data.append(augmented)

    return augmented_data


def process_single_image(img_path, label, output_dir, preprocessing_fn, target_qubits=NUM_QUBITS):
    """
    处理单张图像的封装函数，供多进程调用
    """
    try:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        processed_vector = preprocessing_fn(img_path)
        quantum_encoded = quantum_amplitude_encoding(processed_vector, target_qubits)
        augmented_data = create_quantum_augmentations(quantum_encoded, AUGMENTATION_FACTOR)

        count = 0
        for idx, data in enumerate(augmented_data):
            suffix = 'orig' if idx == 0 else f'aug{idx - 1}'
            filename_out = f"{label}_{base_name}_{suffix}.npy"
            filepath_out = os.path.join(output_dir, filename_out)

            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                print(f"警告: {filename_out} 包含无效数据，跳过")
                continue

            np.save(filepath_out, data)
            count += 1

            if count % 10 == 0:
                print(f"已编码 {count} 个样本...")

        return count
    except Exception as e:
        print(f"处理图像时出错: {str(e)}")
        return 0


def process_directory(image_dir, label, output_dir, preprocessing_fn):
    """
    并行处理指定目录下的图像
    """
    count = 0
    image_files = [f for f in os.listdir(image_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"处理 {len(image_files)} 个 {label} 图像...")

    with ProcessPoolExecutor(max_workers=4) as executor:  # 控制最大并发数为 4
        futures = [
            executor.submit(process_single_image, os.path.join(image_dir, filename),
                            label, output_dir, preprocessing_fn)
            for filename in image_files
        ]
        for future in futures:
            count += future.result()

    print(f"完成 {label} 处理: {count} 个成功编码的样本")
    return count


if __name__ == "__main__":
    for mode, preprocess_fn in {
        "HSI": preprocess_hsi_image,
        "LAB": preprocess_lab_image
    }.items():
        print(f"\n==== 开始 {mode} 模式的图像处理 ====")

        real_out = os.path.join(output_base_dir, f"real_{mode}")  # 真实图像输出目录
        fake_out = os.path.join(output_base_dir, f"deepfake_{mode}")  # 深度伪造图像输出目录
        os.makedirs(real_out, exist_ok=True)  # 创建目录
        os.makedirs(fake_out, exist_ok=True)  # 创建目录

        real_count = process_directory(input_dirs["real"], "real", real_out, preprocess_fn)  # 处理真实图像
        fake_count = process_directory(input_dirs["deepfake"], "deepfake", fake_out, preprocess_fn)  # 处理深度伪造图像

        print(f"[{mode}] 编码完成: real={real_count}, deepfake={fake_count}")  # 输出结果
