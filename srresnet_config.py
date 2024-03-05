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
cudnn.benchmark = True  #自动寻找最优化处理，加速训练
# When evaluating the performance of the SR model, whether to verify only the Y channel image data
only_test_y_channel = True
# Model architecture name
g_arch_name = "srresnet_x4"
# Model arch config
in_channels = 3  #输入通道数
out_channels = 3  #输出通道数
channels = 64
num_rcb = 16  #通道数信息
# Test upscale factor
upscale_factor = 4  #目标尺寸放大倍数
# Current configuration parameter method
mode = "test"
# Experiment name, easy to save weights and log files
exp_name = "SRResNet_x4-DIV2K"

if mode == "train":
    # Dataset address
    train_gt_images_dir = f"./data/ImageNet/SRGAN/Set5train"

    test_gt_images_dir = f"./data/Set5/GTmod12"
    test_lr_images_dir = f"./data/Set5/LRbicx{upscale_factor}"

    gt_image_size = 96  #目标图像尺寸
    batch_size = 16  #批次大小，指定模型更新时所使用的样本数量
    num_workers = 4  #数据加载器的工作进程数

    # The address to load the pretrained model
    pretrained_model_weights_path = f""

    # Incremental training and migration training
    resume_g_model_weights_path = f""  #增量训练和迁移训练时恢复的模型权重文件路径

    # Total num epochs (1,000,000 iters)
    epochs = 90

    # loss function weights
    loss_weights = 1.0  #损失函数的权重参数

    # Optimizer parameter
    model_lr = 1e-4
    model_betas = (0.9, 0.999)
    model_eps = 1e-8
    model_weight_decay = 0.01

    # How many iterations to print the training result
    train_print_frequency = 100  #训练结果的打印频率
    valid_print_frequency = 1  #验证结果的打印频率，每训练一批次打印一次验证结果

if mode == "test":
    # Test data address
    lr_dir = f"./data/Set14/LRbicx{upscale_factor}"
    sr_dir = f"./results/test/Set14/{exp_name}"
    gt_dir = f"./data/Set14/GTmod12"

    model_weights_path = "./results/pretrained_models/SRResNet_x4-ImageNet-6dd5216c.pth.tar"
