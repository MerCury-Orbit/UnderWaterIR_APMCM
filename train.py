import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import models_unet
from losses import generator_loss, discriminator_loss, cycle_consistency_loss
from dataset import SpaceTelescopeDataset
import dataset
import csv
from torch.cuda.amp import autocast, GradScaler
import os
from torchvision.utils import save_image
import torch.nn.functional as F
# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数设置
learning_rate = 0.0016
batch_size = 8
num_epochs = 60
checkpoint_epoch = 5  # 每5个epoch保存一次模型
lambda_ssim = 1.0  # SSIM损失的权重
lambda_cycle = 10.0  # 循环一致性损失的权重
log_file = './training_log.csv'  # 保存训练日志的CSV文件路径


# 创建数据集
degraded_img_dir = ''
real_img_dir = ''
psf_dir = ''

dataset = SpaceTelescopeDataset(degraded_img_dir, real_img_dir, psf_dir, transform=dataset.transform)
print(dataset[0])
# 数据加载器
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# for batch in dataloader:
#     print(batch['degraded_img'].shape)  # 打印退化图像的尺寸
#     print(batch['real_img'].shape)  # 打印真实图像的尺寸
#     print(batch['psf_matrix'].shape)  # 打印PSF矩阵的尺寸
#
#     break
# 初始化模型
generator_G = models_unet.GeneratorUNet().to(device)  # 生成器 G：从退化图像到真实图像
generator_F = models_unet.GeneratorUNet().to(device)  # 生成器 F：从真实图像到退化图像
discriminator_X = models_unet.DiscriminatorUNet().to(device)  # 判别器 X：判定退化图像是否真实
discriminator_Y = models_unet.DiscriminatorUNet().to(device)  # 判别器 Y：判定真实图像是否真实

# 初始化优化器
optimizer_G = optim.Adam(list(generator_G.parameters()) + list(generator_F.parameters()), lr=learning_rate, betas=(0.5, 0.99))
optimizer_D_X = optim.Adam(discriminator_X.parameters(), lr=learning_rate, betas=(0.5, 0.9))
optimizer_D_Y = optim.Adam(discriminator_Y.parameters(), lr=learning_rate, betas=(0.5, 0.9))

# 恢复训练时加载模型和优化器的状态
start_epoch = 0
checkpoint_dir = './checkpoints/'

# 如果有已保存的模型和优化器状态，恢复它们
if os.path.exists(checkpoint_dir):
    try:
        # 从检查点恢复模型和优化器的状态
        generator_G.load_state_dict(torch.load(os.path.join(checkpoint_dir, f'generator_G_epoch_{start_epoch + 1}.pth')))
        generator_F.load_state_dict(torch.load(os.path.join(checkpoint_dir, f'generator_F_epoch_{start_epoch + 1}.pth')))
        discriminator_X.load_state_dict(torch.load(os.path.join(checkpoint_dir, f'discriminator_X_epoch_{start_epoch + 1}.pth')))
        discriminator_Y.load_state_dict(torch.load(os.path.join(checkpoint_dir, f'discriminator_Y_epoch_{start_epoch + 1}.pth')))
        optimizer_G.load_state_dict(torch.load(os.path.join(checkpoint_dir, f'optimizer_G_epoch_{start_epoch + 1}.pth')))
        optimizer_D_X.load_state_dict(torch.load(os.path.join(checkpoint_dir, f'optimizer_D_X_epoch_{start_epoch + 1}.pth')))
        optimizer_D_Y.load_state_dict(torch.load(os.path.join(checkpoint_dir, f'optimizer_D_Y_epoch_{start_epoch + 1}.pth')))
        print(f"Resuming from epoch {start_epoch + 1}")
    except FileNotFoundError:
        print("No checkpoint found, starting from scratch.")
else:
    print("Checkpoint directory not found, starting from scratch.")

# 打开训练日志文件并写入表头
with open(log_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'Batch', 'Generator_Loss', 'Discriminator_Loss_X', 'Discriminator_Loss_Y', 'Adversarial_Loss', 'SSIM_Loss', 'Cycle_Loss'])





generator_G.train()
generator_F.train()
discriminator_X.train()
discriminator_Y.train()

# 训练循环
for epoch in range(start_epoch, num_epochs):
    generator_G.train()
    generator_F.train()
    discriminator_X.train()
    discriminator_Y.train()

    for i, data in enumerate(dataloader):
        # 发送到设备
        degraded_image = data['degraded_img']
        real_image = data['real_img']
        psf_matrix = data['psf_matrix']
        degraded_image = degraded_image.to(device)
        real_image = real_image.to(device)
        psf_matrix = psf_matrix.to(device)

        # 生成器梯度清空
        optimizer_G.zero_grad()


        real_image_from_degraded = generator_G(degraded_image,psf_matrix)  # 使用GeneratorG从退化图像生成真实图像
        fake_degraded_from_real = generator_F(real_image,psf_matrix)  # 使用GeneratorF从真实图像生成退化图像

        # 判别器Y的真实和假图像预测
        # 判别器Y
        # 真实图像：来自目标域的真实图像。
        # 假图像：生成器G从退化图像生成的图像（即 real_image_from_degraded）。
        # Y的任务是判断输入的图像是否属于目标域 （真实图像）。
        fake_preds_G = discriminator_Y(real_image_from_degraded)  # 判别器Y预测生成的真实图像
        real_preds_Y = discriminator_Y(real_image)  # 判别器Y预测真实图像

        real_image = F.interpolate(real_image, size=(584, 584), mode='bilinear', align_corners=False)
        # 计算生成器G的损失（包含对抗损失和SSIM损失）
        g_loss, adv_loss, ssim_loss = generator_loss(fake_preds_G, real_image, real_image_from_degraded, lambda_ssim=lambda_ssim)


        recovered_image = generator_F(real_image_from_degraded, psf_matrix)  # 使用GeneratorF从真实图像还原到退化图像


        # 计算循环一致性损失
        # real_image 是原始的真实图像。
        # recovered_image 是通过生成器 G 和 F 转换后，最终恢复的图像。即，首先从退化图像生成一个伪真实图像，然后用 F 将该伪真实图像恢复成一个退化图像。
        # lambda_cycle 是一个超参数，用来控制循环一致性损失在总损失中的权重。较大的 lambda_cycle 会让模型更加注重恢复一致性。

        recovered_image = F.interpolate(recovered_image, size=(584, 584), mode='bilinear', align_corners=False)
        cycle_loss = cycle_consistency_loss(real_image, recovered_image, lambda_cycle=lambda_cycle)

        # 总生成器损失 = 生成器损失 + 循环一致性损失
        total_g_loss = g_loss + cycle_loss

        # 反向传播和更新生成器，optimizer_G.step()这个操作是用来更新生成器（Generator）的参数
        total_g_loss.backward()
        optimizer_G.step()
        # 梯度下降法在深度学习中是通过计算损失函数相对于各层参数的偏导数来更新模型的参数
        # 优化器（Optimizer）是深度学习中用来更新神经网络参数的算法，它的作用是在训练过程中根据梯度信息调整模型的参数，以最小化损失函数。优化器决定了如何根据计算得到的梯度来更新网络中的各个参数
        # 优化器是得出偏导数的值以后，再来计算梯度要如何下降



        # optimizer_D_X.zero_grad()这个操作是用来清空判别器X的梯度缓存。
        # 在 PyTorch 中，梯度是累积的，因此在每次反向传播之前，需要清空之前的梯度，否则梯度会累积导致不正确的更新
        optimizer_D_X.zero_grad()

        # 计算判别器损失
        # 判别器X
        # 真实图像：来自源域的退化图像（即原始退化图像）。
        # 假图像：由生成器 F从真实图像生成的退化图像。
        # X的任务是判断输入的图像是否属于源域（退化图像）
        fake_preds_X = discriminator_X(fake_degraded_from_real.detach())  # 判别器X预测生成的退化图像
        real_preds_X = discriminator_X(real_image)  # 判别器X预测源图像
        d_loss_X = discriminator_loss(real_preds_X, fake_preds_X)

        # 反向传播和更新判别器X
        d_loss_X.backward()
        optimizer_D_X.step()

        optimizer_D_Y.zero_grad()

        # 计算判别器Y损失
        fake_preds_Y = discriminator_Y(real_image_from_degraded.detach())  # 判别器Y预测生成的真实图像
        d_loss_Y = discriminator_loss(real_preds_Y, fake_preds_Y)

        # 反向传播和更新判别器Y
        d_loss_Y.backward()
        optimizer_D_Y.step()

        # 每100个batch输出一次训练信息
        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(dataloader)}], "
                  f"Generator Loss: {total_g_loss.item():.4f}, Discriminator Loss X: {d_loss_X.item():.4f}, "
                  f"Discriminator Loss Y: {d_loss_Y.item():.4f}, "
                  f"Adversarial Loss: {adv_loss.item():.4f}, SSIM Loss: {ssim_loss.item():.4f}, Cycle Loss: {cycle_loss.item():.4f}")
        sample_interval = 100
        if i % sample_interval == 0:  # 每sample_interval步保存一次
            samples_dir = "output_samples"
            os.makedirs(samples_dir, exist_ok=True)

            # Normalize images to [0, 1] range for saving
            real_image_from_degraded_normalized = (real_image_from_degraded + 1) / 2
            fake_degraded_from_real_normalized = (fake_degraded_from_real + 1) / 2

            # Save images
            save_image(real_image_from_degraded_normalized,
                       f"{samples_dir}/epoch_{epoch}_batch_{i}_real_image_from_degraded.png")
            save_image(fake_degraded_from_real_normalized,
                       f"{samples_dir}/epoch_{epoch}_batch_{i}_fake_degraded_from_real.png")
        # 记录每个batch的损失到CSV文件
        with open(log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, i, total_g_loss.item(), d_loss_X.item(), d_loss_Y.item(), adv_loss.item(), ssim_loss.item(), cycle_loss.item()])

    # 保存模型和优化器状态
    if (epoch + 1) % checkpoint_epoch == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(generator_G.state_dict(), os.path.join(checkpoint_dir, f'generator_G_epoch_{epoch + 1}.pth'))
        torch.save(generator_F.state_dict(), os.path.join(checkpoint_dir, f'generator_F_epoch_{epoch + 1}.pth'))
        torch.save(discriminator_X.state_dict(), os.path.join(checkpoint_dir, f'discriminator_X_epoch_{epoch + 1}.pth'))
        torch.save(discriminator_Y.state_dict(), os.path.join(checkpoint_dir, f'discriminator_Y_epoch_{epoch + 1}.pth'))
        torch.save(optimizer_G.state_dict(), os.path.join(checkpoint_dir, f'optimizer_G_epoch_{epoch + 1}.pth'))
        torch.save(optimizer_D_X.state_dict(), os.path.join(checkpoint_dir, f'optimizer_D_X_epoch_{epoch + 1}.pth'))
        torch.save(optimizer_D_Y.state_dict(), os.path.join(checkpoint_dir, f'optimizer_D_Y_epoch_{epoch + 1}.pth'))
        print(f"Model and optimizer states saved for epoch {epoch + 1}.")
