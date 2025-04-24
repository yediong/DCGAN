import os
import torch
import pyarrow.parquet as pq
import torchvision.transforms as transforms
from PIL import Image
import io
from tqdm import tqdm

# 数据集存放位置
dataroot = "data"
# 缓存目录
cache_dir = "cached_data"
# 确保缓存目录存在
os.makedirs(cache_dir, exist_ok=True)

# 训练图片的空间尺寸
image_size = 64

# 预处理
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Parquet文件列表
parquet_files = [
    os.path.join(dataroot, "train-00000-of-00003.parquet"),
    os.path.join(dataroot, "train-00001-of-00003.parquet"),
    os.path.join(dataroot, "train-00002-of-00003.parquet")
]

# 创建索引文件以存储所有缓存的文件路径
cache_paths = []

# 处理所有parquet文件
for file_idx, parquet_file in enumerate(parquet_files):
    print(f"处理文件 {file_idx+1}/{len(parquet_files)}: {parquet_file}")
    
    # 读取parquet文件
    table = pq.read_table(parquet_file)
    df = table.to_pandas()
    
    # 处理每一行
    for idx in tqdm(range(len(df))):
        row = df.iloc[idx]
        try:
            # 从字典中提取bytes数据
            image_data = row['image']
            if isinstance(image_data, dict) and 'bytes' in image_data:
                bytes_data = image_data['bytes']
                # 打开图像
                image = Image.open(io.BytesIO(bytes_data))
                # 转换为RGB
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # 应用转换
                tensor = transform(image)
                
                # 保存为.pt文件
                cache_path = os.path.join(cache_dir, f"file_{file_idx}_sample_{idx}.pt")
                torch.save(tensor, cache_path)
                cache_paths.append(cache_path)
        except Exception as e:
            print(f"处理样本 {idx} 时出错: {e}")

# 保存索引文件
torch.save(cache_paths, os.path.join(cache_dir, "cache_paths.pt"))
print(f"预处理完成。缓存了 {len(cache_paths)} 个样本到 {cache_dir}")