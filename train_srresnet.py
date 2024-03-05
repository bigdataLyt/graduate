import os
import time

import torch
from torch import nn
from torch import optim
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import model
import srresnet_config
from dataset import CUDAPrefetcher, TrainValidImageDataset, TestImageDataset
from image_quality_assessment import PSNR, SSIM
from utils import load_state_dict, make_directory, save_checkpoint, AverageMeter, ProgressMeter


def main():

    start_epoch = 0   #初始轮次为0


    best_psnr = 0.0
    best_ssim = 0.0

    train_prefetcher, test_prefetcher = load_dataset()  #加载数据集
    print("Load all datasets successfully.")

    srresnet_model = build_model()  #构建模型
    print(f"Build `{srresnet_config.g_arch_name}` model successfully.")

    criterion = define_loss()  #定义损失函数
    print("Define all loss functions successfully.")

    optimizer = define_optimizer(srresnet_model)  #定义优化器
    print("Define all optimizer functions successfully.")

    print("Check whether to load pretrained model weights...")
    if srresnet_config.pretrained_model_weights_path:
        srresnet_model = load_state_dict(srresnet_model, srresnet_config.pretrained_model_weights_path)
        print(f"Loaded `{srresnet_config.pretrained_model_weights_path}` pretrained model weights successfully.")
    else:
        print("Pretrained model weights not found.")

    print("Check whether the pretrained model is restored...")
    if srresnet_config.resume_g_model_weights_path:
        srresnet_model, _, start_epoch, best_psnr, best_ssim, optimizer, _ = load_state_dict(
            srresnet_model,
            srresnet_config.resume_g_model_weights_path,
            optimizer=optimizer,
            load_mode="resume")
        print("Loaded pretrained model weights.")
    else:
        print("Resume training model not found. Start training from scratch.")

    # 创建实验结果
    samples_dir = os.path.join("samples", srresnet_config.exp_name)
    results_dir = os.path.join("results", srresnet_config.exp_name)
    make_directory(samples_dir)
    make_directory(results_dir)

    # 创建log文件 以便训练的可视化和追踪
    writer = SummaryWriter(os.path.join("samples", "logs", srresnet_config.exp_name))

    # 初始化梯度标量
    scaler = amp.GradScaler()

    # 相似度评价标准
    psnr_model = PSNR(srresnet_config.upscale_factor, srresnet_config.only_test_y_channel)  #峰值信噪比  均方误差
    ssim_model = SSIM(srresnet_config.upscale_factor, srresnet_config.only_test_y_channel)  #结构相似性指数  compara亮度 对比度和结构

    # 将图像质量评估模型转移到指定设备
    psnr_model = psnr_model.to(device=srresnet_config.device)
    ssim_model = ssim_model.to(device=srresnet_config.device)

    for epoch in range(start_epoch, srresnet_config.epochs):
        train(srresnet_model,
              train_prefetcher,
              criterion,
              optimizer,
              epoch,
              scaler,
              writer)
        psnr, ssim = validate(srresnet_model,
                              test_prefetcher,
                              epoch,
                              writer,
                              psnr_model,
                              ssim_model,
                              "Test")
        print("\n")

        # 训练自动保存最佳权重
        is_best = psnr > best_psnr and ssim > best_ssim
        is_last = (epoch + 1) == srresnet_config.epochs
        best_psnr = max(psnr, best_psnr)
        best_ssim = max(ssim, best_ssim)
        save_checkpoint({"epoch": epoch + 1,
                         "best_psnr": best_psnr,
                         "best_ssim": best_ssim,
                         "state_dict": srresnet_model.state_dict(),  #model初始状态
                         "optimizer": optimizer.state_dict()},  #优化器状态
                        f"g_epoch_{epoch + 1}.pth.tar",
                        samples_dir,  #样本
                        results_dir,  #结果
                        "g_best.pth.tar",
                        "g_last.pth.tar",
                        is_best,
                        is_last)


def load_dataset() -> [CUDAPrefetcher, CUDAPrefetcher]:
    # Load Set5train, test and valid datasets
    train_datasets = TrainValidImageDataset(srresnet_config.train_gt_images_dir,
                                            srresnet_config.gt_image_size,  #图像大小
                                            srresnet_config.upscale_factor,
                                            "Train")
    test_datasets = TestImageDataset(srresnet_config.test_gt_images_dir, srresnet_config.test_lr_images_dir)

    # Generator all dataloader
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=srresnet_config.batch_size,  #批量大小
                                  shuffle=True,  #是否打乱数据
                                  num_workers=srresnet_config.num_workers,  #工作线程数
                                  pin_memory=True,  #是否将数据存储在固定的内存中
                                  drop_last=True,  #是否丢弃最后一个不完整的批次
                                  persistent_workers=True)  #是否启用持久化工作线程
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=1,  #批量大小为1
                                 shuffle=False,  #是否打乱数据
                                 num_workers=1,  #工作线程数为1
                                 pin_memory=True,  #是否将数据存储在固定的内存中
                                 drop_last=False,  #是否丢弃最后一个不完整的批次
                                 persistent_workers=True)  #是否启用持久化工作线程

    # Place all data on the preprocessing data loader
    train_prefetcher = CUDAPrefetcher(train_dataloader, srresnet_config.device)
    test_prefetcher = CUDAPrefetcher(test_dataloader, srresnet_config.device)

    return train_prefetcher, test_prefetcher


def build_model() -> nn.Module:
    srresnet_model = model.__dict__[srresnet_config.g_arch_name](in_channels=srresnet_config.in_channels,
                                                                 out_channels=srresnet_config.out_channels,
                                                                 channels=srresnet_config.channels,
                                                                 num_rcb=srresnet_config.num_rcb)
    srresnet_model = srresnet_model.to(device=srresnet_config.device)

    return srresnet_model


def define_loss() -> nn.MSELoss:
    criterion = nn.MSELoss()  #损失函数使用MSE均方误差
    criterion = criterion.to(device=srresnet_config.device)

    return criterion


def define_optimizer(srresnet_model) -> optim.Adam:    # Adam优化器
    optimizer = optim.Adam(srresnet_model.parameters(),
                           srresnet_config.model_lr,
                           srresnet_config.model_betas,
                           srresnet_config.model_eps,
                           srresnet_config.model_weight_decay)  # 权重衰减（采用L2正则化，模型权重的平方和的一半（乘以一个正则化系数），来惩罚较大的权重值）

    return optimizer


def train(
        srresnet_model: nn.Module,        # 输入：超分辨率网络模型
        train_prefetcher: CUDAPrefetcher, # 输入：数据加载器
        criterion: nn.MSELoss,            # 输入：均方误差损失函数
        optimizer: optim.Adam,            # 输入：优化器
        epoch: int,                       # 输入：当前训练轮数
        scaler: amp.GradScaler,           # 输入：混合精度训练工具
        writer: SummaryWriter            # 输入：用于写入训练日志的对象
) -> None:                             # 输出：无


    batches = len(train_prefetcher)      # 计算当前epoch数据集中包含多少个数据批次

    # 打印过程信息
    batch_time = AverageMeter("Time", ":6.3f")   # 定义计算每个批次数据处理时间的工具类对象
    data_time = AverageMeter("Data", ":6.3f")   # 定义计算每个批次数据加载时间的工具类对象
    losses = AverageMeter("Loss", ":6.6f")      # 定义计算每个批次损失函数值的工具类对象
    progress = ProgressMeter(batches, [batch_time, data_time, losses], prefix=f"Epoch: [{epoch + 1}]") # 定义输出训练进度信息的工具类对象


    srresnet_model.train()             # 设置超分辨率网络模型为训练模式


    batch_index = 0                    # 初始化当前批次序号，用于判断是否需要打印日志


    train_prefetcher.reset()           # 重置数据加载器，以便从头开始获取数据批次
    batch_data = train_prefetcher.next()  # 从数据加载器中获取第一个数据批次


    end = time.time()                  # 获取当前时间，用于计算每个批次处理时间

    while batch_data is not None:      # 当还有数据批次未处理时循环执行以下操作：


        data_time.update(time.time() - end)  # 计算加载数据的时间

        # 选择设备
        gt = batch_data["gt"].to(device=srresnet_config.device, non_blocking=True)  # 将高分辨率图像从内存复制到GPU显存中，并设置为非阻塞模式
        lr = batch_data["lr"].to(device=srresnet_config.device, non_blocking=True)  # 将低分辨率图像从内存复制到GPU显存中，并设置为非阻塞模式


        # 初始化生成器梯度信息
        srresnet_model.zero_grad(set_to_none=True)  # 清空 SRResNet 模型的梯度信息

        # 混合精密训练
        with amp.autocast():  # 开启 Mixed Precision 训练器，自动选择合适精度计算
            sr = srresnet_model(lr)  #使用模型预测得到超分辨后的图像

            #计算损失值loss，其中criterion是损失函数，在计算损失时，使用torch.mul()对损失权重和计算结果进行逐元素相乘
            loss = torch.mul(srresnet_config.loss_weights, criterion(sr, gt))

        # 反向传播（损失函数的梯度）
        scaler.scale(loss).backward()

        scaler.step(optimizer)  # 使用梯度放缩更新优化器，确保在进行梯度更新时梯度的数值范围得到合适的缩放
        scaler.update()  #更新梯度缩放器的状态

        #获取当前批次损失值，获取当前批次的样本数量，更新损失
        losses.update(loss.item(), lr.size(0))

        # 计算set5训练一批数据所需的时间
        batch_time.update(time.time() - end)
        end = time.time()

        # 将训练期间的数据写入训练log日志文件
        if batch_index % srresnet_config.train_print_frequency == 0:

            writer.add_scalar("Train/Loss", loss.item(), batch_index + epoch * batches + 1)  #将训练损失记录到 TensorBoard 日志文件中
            progress.display(batch_index + 1)  #在终端上显示进度条和平均批次时间


        batch_data = train_prefetcher.next()  #加载下一批次的数据


        batch_index += 1  #增加当前批次的索引，以便在终端上正确显示日志信息


def validate(
        srresnet_model: nn.Module,
        data_prefetcher: CUDAPrefetcher,
        epoch: int,
        writer: SummaryWriter,
        psnr_model: nn.Module,
        ssim_model: nn.Module,
        mode: str
) -> [float, float]:
    # 计算每个Epoch中有多少批数据
    batch_time = AverageMeter("Time", ":6.3f")
    psnres = AverageMeter("PSNR", ":4.2f")
    ssimes = AverageMeter("SSIM", ":4.4f")
    progress = ProgressMeter(len(data_prefetcher), [batch_time, psnres, ssimes], prefix=f"{mode}: ")

    #将对抗网络模型置于验证模式
    srresnet_model.eval()

    # 初始化终端打印日志的数据批次数
    batch_index = 0

    # 初始化数据加载器并加载第一批数据
    data_prefetcher.reset()
    batch_data = data_prefetcher.next()

    # 初始化测试时间
    end = time.time()

    with torch.no_grad():
        while batch_data is not None:
            # 将内存中的数据传输到CUDA设备以加快测试速度
            gt = batch_data["gt"].to(device=srresnet_config.device, non_blocking=True)
            lr = batch_data["lr"].to(device=srresnet_config.device, non_blocking=True)

            # 使用生成器模型生成一个假样本
            with amp.autocast():
                sr = srresnet_model(lr)

            # 统计损失值
            psnr = psnr_model(sr, gt)
            ssim = ssim_model(sr, gt)
            psnres.update(psnr.item(), lr.size(0))
            ssimes.update(ssim.item(), lr.size(0))

            # 计算测试一批数据所需的时间
            batch_time.update(time.time() - end)
            end = time.time()

            # 记录日志信息
            if batch_index % srresnet_config.valid_print_frequency == 0:
                progress.display(batch_index + 1)

            # 预加载下一批数据
            batch_data = data_prefetcher.next()

            # 训练完一批数据后，在数据批数上加1，保证terminal正常打印数据
            batch_index += 1

    # 打印
    progress.display_summary()

    if mode == "Valid" or mode == "Test":
        writer.add_scalar(f"{mode}/PSNR", psnres.avg, epoch + 1)
        writer.add_scalar(f"{mode}/SSIM", ssimes.avg, epoch + 1)
    else:
        raise ValueError("Unsupported mode, please use `Valid` or `Test`.")

    return psnres.avg, ssimes.avg


if __name__ == "__main__":
    main()
