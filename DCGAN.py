import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from matplotlib import MatplotlibDeprecationWarning
import warnings
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# 缓存数据目录
cache_dir = "cached_data"

# 训练批量大小 (可以增大)
batch_size = 256

# 训练图片的空间尺寸
image_size = 64

# 训练通道数
nc = 3

# 生成器输入大小
nz = 100

# 生成器特征图大小
ngf = 64

# 辨别器特征图大小
ndf = 64

# 训练周期
num_epochs = 20

# 优化器学习率 (降低判别器学习率)
lr = 0.0002
lr_D = lr / 4  # 判别器学习率减小为生成器的1/4

# Adam优化器中的超参数
beta1 = 0.7  # 调整为0.7

# 可用的GPU数量
ngpu = 1

# 设备配置
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# 自定义缓存数据集
class CachedDataset(Dataset):
    def __init__(self, cache_paths):
        self.cache_paths = cache_paths
        
    def __len__(self):
        return len(self.cache_paths)
    
    def __getitem__(self, idx):
        # 直接加载预处理的张量
        image = torch.load(self.cache_paths[idx])
        return image, 0  # 标签为0，GAN不需要标签

# 加载缓存路径
cache_paths = torch.load(os.path.join(cache_dir, "cache_paths.pt"))
print(f"加载了 {len(cache_paths)} 个缓存样本")

# 创建数据集
dataset = CachedDataset(cache_paths)

# 加载数据 (启用多进程和pin_memory)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=4, 
                                         pin_memory=True)

# 生成器定义
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# 判别器定义
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()#映射到区间为(0, 1)
        )

    def forward(self, input):
        return self.main(input)


# 权重初始化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# 创建生成器
netG = Generator(ngpu).to(device)
netG.apply(weights_init)

# 创建判别器
netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)

# 损失函数
loss_fn = nn.BCELoss()

# 固定噪声（用于可视化）
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# 真假标签（增加标签平滑）
real_label = 0.9  # 标签平滑
fake_label = 0.0

# 优化器（判别器学习率降低）
optimizerD = optim.Adam(netD.parameters(), lr=lr_D, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# 用于可视化的函数
def gen_img_plot(netG, noise):
    with torch.no_grad():
        fake = netG(noise).detach().cpu()
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(np.transpose(vutils.make_grid(fake, padding=2, normalize=True), (1, 2, 0)))
    output_dir = './train_result/'
    os.makedirs(output_dir, exist_ok=True)  # 创建文件夹（如果不存在）
    plt.savefig(os.path.join(output_dir, f'generated_image_iters_{iters}.png'))

# 训练状态记录
img_list = []
G_losses = []
D_losses = []
iters = 0

print("开始训练...")
# 训练循环
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader):
        ############################
        # (1) 更新判别器: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        netD.zero_grad()
        
        # 真实图像训练
        real_cpu = data[0].to(device, non_blocking=True)  # 异步传输
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = netD(real_cpu).view(-1)
        errD_real = loss_fn(output, label)
        errD_real.backward()
        D_x = output.mean().item()
        
        # 生成的假图像训练
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        errD_fake = loss_fn(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        
        # 计算总误差并更新
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) 更新生成器: maximize log(D(G(z)))
        ###########################
        # 每 2 次更新判别器，更新 1 次生成器
        if i % 2 == 0:
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = loss_fn(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
        
        # 输出训练状态
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        
        # 保存损失
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
        # 每500次迭代保存生成的图像
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
                gen_img_plot(netG, fixed_noise)
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            
        iters += 1
        
    # 每个epoch结束保存模型
    output_dir = './model/'
    os.makedirs(output_dir, exist_ok=True)  # 创建文件夹（如果不存在）
    torch.save(netG.state_dict(), f'model/model_DCGAN_epoch_{epoch}.pth')

# 最终模型保存
torch.save(netG.state_dict(), 'model/model_DCGAN_final.pth')

# 绘制损失曲线
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('loss_curve.png')
print("训练完成！")