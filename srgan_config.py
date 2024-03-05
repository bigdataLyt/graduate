import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# When evaluating the performance of the SR model, whether to verify only the Y channel image data
only_test_y_channel = True
# Model architecture name
d_arch_name = "discriminator"
g_arch_name = "srresnet_x4"
# Model arch config
in_channels = 3  #输入通道数
out_channels = 3  #输出通道数
channels = 64  #中间层通道数
num_rcb = 16  #残差卷积块数
# Test upscale factor
upscale_factor = 4
# Current configuration parameter method
mode = "test"
# Experiment name, easy to save weights and log files
exp_name = "SRGAN_x4-DIV2K"

if mode == "train":
    # Dataset address
    train_gt_images_dir = f"./data/ImageNet/SRGAN/Set5train"
    test_gt_images_dir = f"./data/Set5/GTmod12"
    test_lr_images_dir = f"./data/Set5/LRbicx{upscale_factor}"

    gt_image_size = 96  #目标图像尺寸
    batch_size = 16  #批次大小，指定模型更新时所使用的样本数量
    num_workers = 4  #数据加载器的工作进程数

    # The address to load the pretrained model
    pretrained_d_model_weights_path = f""  #判别器预训练权重文件地址
    pretrained_g_model_weights_path = f"./results/SRResNet_x4-DIV2K/g_best.pth.tar"  #生成器预训练权重文件地址
    # Incremental training and migration training
    resume_d_model_weights_path = f""  #断点恢复，保存文件中间状态，方便训练和状态评估
    resume_g_model_weights_path = f""  #断点恢复，保存文件中间状态，方便训练和状态评估

    # Total num epochs (200,000 iters)
    epochs = 18      #总训练迭代次数

    # Loss function weight
    pixel_weight = 1.0   #像素损失权重，平衡生成图像与目标图像间像素级差异
    content_weight = 1.0  #内容损失权重，衡量生成与目标间的感知相似性
    adversarial_weight = 0.001  #对抗损失的权重，用于训练--生成器

    # Feature extraction layer parameter configuration
    feature_model_extractor_node = "features.35"
    feature_model_normalize_mean = [0.485, 0.456, 0.406]#红色通道的均值为0.485，绿色通道的均值为0.456，蓝色通道的均值为0.406
    feature_model_normalize_std = [0.229, 0.224, 0.225] #红色通道的标准差为0.229，绿色通道的标准差为0.224，蓝色通道的标准差为0.225

    # Optimizer parameter
    model_lr = 1e-4  #学习率，决定每次参数更新的步长大小
    model_betas = (0.9, 0.999)  #用于优化器，控制梯度和平方梯度的平均衰减率
    model_eps = 1e-8  #优化器中添加的常数，避免计算过程中出现0作除数
    model_weight_decay = 0.01  #权重衰减，有关正则化方面的技术，通过在损失函数中添加权重的平方范数来约束模型的复杂度

    # Dynamically adjust the learning rate policy [100,000 | 200,000]
    lr_scheduler_step_size = epochs // 2  #学习率的调整频率
    lr_scheduler_gamma = 0.1  #调整学习率的比例因子

    # How many iterations to print the training result
    train_print_frequency = 100
    valid_print_frequency = 1

if mode == "test":
    # Test data address
    lr_dir = f"./data/Set14/LRbicx{upscale_factor}"
    sr_dir = f"./results/test/Set14/{exp_name}"
    gt_dir = f"./data/Set14/GTmod12"
    g_model_weights_path = f"./results/pretrained_models/SRGAN_x4-ImageNet-8c4a7569.pth.tar"
