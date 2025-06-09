from PIL import Image
import os

# 定义路径
input_folder = "RGB/11real"  # 原始图片存放的文件夹
output_folder = "RGB/12real"  # 下采样后图片保存的文件夹（自动创建）
target_size = (8, 8)  # 目标分辨率：4x4或8x8

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 遍历输入文件夹中的所有图片
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # 打开原始图片
        input_path = os.path.join(input_folder, filename)
        img = Image.open(input_path)

        # 下采样到目标尺寸
        img_low = img.resize(target_size, 1)

        # 生成新文件名（添加_low后缀）
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_low{ext}"
        output_path = os.path.join(output_folder, output_filename)

        # 保存下采样图片
        img_low.save(output_path)
        print(f"Saved: {output_path}")