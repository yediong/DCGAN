import os
import torch
import torch.nn as nn
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='model/model_DCGAN_final.pth', help='模型路径')
parser.add_argument('--output_dir', type=str, default='generated_images', help='输出图像目录')
parser.add_argument('--num_images', type=int, default=64, help='生成图像数量')
parser.add_argument('--image_size', type=int, default=64, help='图像大小')
parser.add_argument('--batch_size', type=int, default=16, help='批量生成的图像数量')
args = parser.parse_args()

# 创建输出目录
os.makedirs(args.output_dir, exist_ok=True)

# 设置参数
nz = 100  # 生成器输入噪声维度
ngf = 64  # 生成器特征图数量
nc = 3    # 通道数(RGB)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义生成器类 (必须与训练时定义完全一致)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 输入是Z，进入卷积转置
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # 状态尺寸: (ngf*16) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 状态尺寸: (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 状态尺寸: (ngf*4) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 状态尺寸: (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # 最终状态尺寸: (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

# 创建生成器实例
netG = Generator().to(device)

# 加载预训练模型
print(f"加载模型: {args.model_path}")
try:
    netG.load_state_dict(torch.load(args.model_path, map_location=device))
    print("模型加载成功")
except Exception as e:
    print(f"加载模型出错: {e}")
    # 尝试备用方案
    try:
        state_dict = torch.load(args.model_path, map_location=device)
        if 'state_dict' in state_dict:
            netG.load_state_dict(state_dict['state_dict'])
        print("使用备用方案加载成功")
    except Exception as e2:
        print(f"备用加载也失败: {e2}")
        exit(1)

# 设置为评估模式
netG.eval()

# 生成图像函数
def generate_images(num_images, batch_size=16):
    images = []
    num_batches = int(np.ceil(num_images / batch_size))
    
    with torch.no_grad():  # 不需要计算梯度
        for i in range(num_batches):
            # 确定当前批次的大小
            current_batch_size = min(batch_size, num_images - i * batch_size)
            if current_batch_size <= 0:
                break
                
            # 生成随机噪声
            noise = torch.randn(current_batch_size, nz, 1, 1, device=device)
            
            # 通过生成器生成图像
            fake = netG(noise).detach().cpu()
            
            # 收集生成的图像
            images.append(fake)
            
            # 保存每张图像
            for j in range(fake.size(0)):
                img_idx = i * batch_size + j
                if img_idx >= num_images:
                    break
                    
                # 转换为PIL图像并保存
                img = fake[j].cpu().clone()
                img = img * 0.5 + 0.5  # 反归一化
                img = img.permute(1, 2, 0).numpy()  # CHW -> HWC
                img = (img * 255).astype(np.uint8)
                img = Image.fromarray(img)
                img.save(os.path.join(args.output_dir, f'generated_face_{img_idx:04d}.png'))
    
    # 将所有图像合并为一个tensor
    all_images = torch.cat(images, 0)
    return all_images[:num_images]  # 确保只返回指定数量的图像

# 生成并保存图像
print(f"正在生成 {args.num_images} 张图像...")
generated_images = generate_images(args.num_images, args.batch_size)

# 创建网格展示
grid = vutils.make_grid(generated_images, padding=2, normalize=True, nrow=8)
plt.figure(figsize=(10, 10))
plt.axis("off")
plt.title("生成的人脸")
plt.imshow(np.transpose(grid, (1, 2, 0)))
plt.savefig(os.path.join(args.output_dir, 'generated_faces_grid.png'), dpi=300, bbox_inches='tight')

print(f"所有图像已保存到 {args.output_dir} 目录")
print(f"生成完成! 共生成 {args.num_images} 张图像。")

# 可选：显示图像
# plt.show()