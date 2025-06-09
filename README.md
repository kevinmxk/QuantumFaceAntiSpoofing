# QuantumFaceAntiSpoofing

本仓库致力于基于量子特征编码和量子神经网络的活体人脸检测与深度伪造（Deepfake）识别。核心思想是将人脸图像经过多尺度特征提取和量子化编码后，利用量子神经网络模型进行真假分类，大幅提升对抗深度伪造的能力。

## 主要特性

- 支持多种颜色空间（RGB、HSI、LAB）下的人脸图像量子编码
- 多种预处理与增强策略，兼容不同分辨率、灰度/彩色图片
- 量子特征编码与Qiskit集成，支持模拟与真实量子后端
- 深度伪造批量检测与准确率评估
- 支持PyTorch下的量子神经网络训练与推理
- 并行处理、增强、特征工程、数据检查等功能完善

## 文件结构

```
.
├── lowbit_sampling.py                # 图像下采样（如8x8、4x4），为量子编码做准备
├── predict_deepfake.py               # 主推理脚本：图片→量子编码→伪造检测
├── qiskit量子化_v4.py                # 量子编码主流程（特征提取、增强、编码、数据检查）
├── qiskit量子化_v4_HSI.py            # 针对HSI/LAB等颜色空间的量子编码，支持多进程
├── Qiskit 构建量子神经网络_v4.py     # 量子神经网络模型训练、评估、可视化
├── requirements.txt                  # 依赖库列表
└── ...
```

## 快速开始

### 1. 环境准备

建议使用conda或virtualenv，并安装依赖：

```bash
pip install -r requirements.txt
```

### 2. 数据预处理

- 使用 `lowbit_sampling.py` 对原始图像进行下采样（如8x8分辨率），保存至指定文件夹。

### 3. 图像量子编码

- 使用 `qiskit量子化_v4.py` 或 `qiskit量子化_v4_HSI.py` 将图片批量编码为量子态（.npy文件），支持增强与多空间处理。

### 4. 训练量子神经网络

- 运行 `Qiskit 构建量子神经网络_v4.py`，加载量子编码数据，进行特征工程、模型训练与保存。

### 5. 伪造检测与推理

- 运行 `predict_deepfake.py`，将待测图片批量转为量子编码，并调用训练好的模型进行真假预测，输出准确率和详细结果。

## 核心流程示意

1. **数据采样与预处理：**
   - 图像 → 下采样/多通道处理 → 特征提取（像素、梯度、统计等）
2. **量子化编码：**
   - 特征向量 → 归一化 → 振幅编码/相位编码 → `.npy`保存
3. **模型训练：**
   - 量子特征数据 → 特征增强 → Qiskit量子神经网络（PyTorch集成）→ 训练/评估
4. **推理检测：**
   - 新图片 → 量子编码 → 模型推理 → 真假判断与准确率分析

## 主要依赖

- Python 3.8+
- numpy, torch, scikit-learn, joblib
- qiskit, qiskit-machine-learning
- tqdm, matplotlib, Pillow

---

## 数据集声明

本项目使用的人脸数据集为**Human Faces Dataset 人脸数据集**，相关信息如下：

- **数据集名称**: Human Faces Dataset 人脸数据集
- **发布机构**: Kaggle 
- **下载地址**: https://orion.hyper.ai/datasets/33799
- **原始发布地址**: https://www.kaggle.com/datasets/kaustubhdhote/human-faces-dataset
- **数据预估大小**: 119 GB
- **简介**: 该数据集包含约 9.6k 张人脸图像，5k 张真实人脸图像，4.63k 张 AI 生成的人脸图像。

### 特别说明

- 数据集由超神经 Hyper.AI（https://orion.hyper.ai）提供大陆范围内公开下载节点，原始版权归Kaggle及数据集原作者所有。
- 本项目仅用于学术研究用途，严禁用于任何商业和违法用途。
- 若需使用数据集，请务必遵循相关平台和原作者的许可协议与使用规范。

---

## 参考 & 鸣谢

- [Qiskit](https://qiskit.org/)
- 量子机器学习相关文献
- 各类公开deepfake数据集
- 超神经 Hyper.AI 提供的数据集节点及服务

---

如需自定义特征工程、量子编码方式、网络结构等，请查阅各脚本详细注释。欢迎Issue和PR！
