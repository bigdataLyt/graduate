import os
import time

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import model
import srgan_config
from dataset import CUDAPrefetcher, TrainValidImageDataset, TestImageDataset
from image_quality_assessment import PSNR, SSIM
from utils import load_state_dict, make_directory, save_checkpoint, AverageMeter, ProgressMeter

model_names = sorted(
    name for name in model.__dict__ if
    name.islower() and not name.startswith("__") and callable(model.__dict__[name]))


def main():
    # 初始化训练的轮数
    start_epoch = 0

    # 初始化用于生成网络评估指标的训练
    best_psnr = 0.0
    best_ssim = 0.0

    train_prefetcher, test_prefetcher = load_dataset()  #加载数据集
    print("Load all datasets successfully.")

    d_model, g_model = build_model()  #构建模型
    print(f"Build `{srgan_config.g_arch_name}` model successfully.")

    pixel_criterion, content_criterion, adversarial_criterion = define_loss()  #定义损失函数
    print("Define all loss functions successfully.")

    d_optimizer, g_optimizer = define_optimizer(d_model, g_model)  #定义优化器
    print("Define all optimizer functions successfully.")

    d_scheduler, g_scheduler = define_scheduler(d_optimizer, g_optimizer)  #定义优化器的学习率调度器
    print("Define all optimizer scheduler functions successfully.")

    print("Check whether to load pretrained d model weights...")
    if srgan_config.pretrained_d_model_weights_path:  #加载与训练的D模型权重
        d_model = load_state_dict(d_model, srgan_config.pretrained_d_model_weights_path)
        print(f"Loaded `{srgan_config.pretrained_d_model_weights_path}` pretrained model weights successfully.")
    else:
        print("Pretrained d model weights not found.")

    print("Check whether to load pretrained g model weights...")
    if srgan_config.pretrained_g_model_weights_path:  #加载预训练的G模型权重
        g_model = load_state_dict(g_model, srgan_config.pretrained_g_model_weights_path)
        print(f"Loaded `{srgan_config.pretrained_g_model_weights_path}` pretrained model weights successfully.")
    else:
        print("Pretrained g model weights not found.")

    print("Check whether the pretrained d model is restored...")
    if srgan_config.resume_d_model_weights_path:  #恢复预训练的D模型
        d_model, _, start_epoch, best_psnr, best_ssim, optimizer, scheduler = load_state_dict(
            d_model,
            srgan_config.resume_d_model_weights_path,
            optimizer=d_optimizer,
            scheduler=d_scheduler,
            load_mode="resume")
        print("Loaded pretrained model weights.")
    else:
        print("Resume training d model not found. Start training from scratch.")

    print("Check whether the pretrained g model is restored...")
    if srgan_config.resume_g_model_weights_path:  #恢复预训练的G模型
        g_model, _, start_epoch, best_psnr, best_ssim, optimizer, scheduler = load_state_dict(
            g_model,
            srgan_config.resume_g_model_weights_path,
            optimizer=g_optimizer,
            scheduler=g_scheduler,
            load_mode="resume")
        print("Loaded pretrained model weights.")
    else:
        print("Resume training g model not found. Start training from scratch.")

    # 创建实验结果目录
    samples_dir = os.path.join("samples", srgan_config.exp_name)
    results_dir = os.path.join("results", srgan_config.exp_name)
    make_directory(samples_dir)
    make_directory(results_dir)

    # 创建训练过程日志文件
    writer = SummaryWriter(os.path.join("samples", "logs", srgan_config.exp_name))

    # 创建图像质量评估模型
    psnr_model = PSNR(srgan_config.upscale_factor, srgan_config.only_test_y_channel)
    ssim_model = SSIM(srgan_config.upscale_factor, srgan_config.only_test_y_channel)

    # 将图像质量评估模型转移到指定设备
    psnr_model = psnr_model.to(device=srgan_config.device)
    ssim_model = ssim_model.to(device=srgan_config.device)

    for epoch in range(start_epoch, srgan_config.epochs):  #训练模型
        train(d_model,
              g_model,
              train_prefetcher,  #数据预处理
              pixel_criterion,  #像素损失函数
              content_criterion,  #内容损失函数
              adversarial_criterion,  #对抗损失函数
              d_optimizer,  #判别器的优化器
              g_optimizer,  #生成器的优化器
              epoch,  #训练伦次
              writer)  #写入训练过程中的指标和日志
          #验证模型并计算图像质量指标
        psnr, ssim = validate(g_model,
                              test_prefetcher,  #测试集欲处理对象
                              epoch,
                              writer,
                              psnr_model,
                              ssim_model,
                              "Test")
        print("\n")

        # 更新学习率
        d_scheduler.step()  #调用学习率调度器step（）
        g_scheduler.step()  #调用学习率调度器step（）

        # 自动保存具有最佳指标的模型
        is_best = psnr > best_psnr and ssim > best_ssim
        is_last = (epoch + 1) == srgan_config.epochs
        best_psnr = max(psnr, best_psnr)
        best_ssim = max(ssim, best_ssim)
        save_checkpoint({"epoch": epoch + 1,
                         "best_psnr": best_psnr,
                         "best_ssim": best_ssim,
                         "state_dict": d_model.state_dict(),  #判别器状态
                         "optimizer": d_optimizer.state_dict(),  #判别器优化器状态
                         "scheduler": d_scheduler.state_dict()},  #判别器调度器状态
                        f"d_epoch_{epoch + 1}.pth.tar",
                        samples_dir,  #用于保存样本输出的目录路径。
                        results_dir,  #用于保存结果输出的目录路径。
                        "d_best.pth.tar",
                        "d_last.pth.tar",
                        is_best,
                        is_last)
        save_checkpoint({"epoch": epoch + 1,
                         "best_psnr": best_psnr,
                         "best_ssim": best_ssim,
                         "state_dict": g_model.state_dict(),  #生成器状态
                         "optimizer": g_optimizer.state_dict(),  #生成器优化器状态
                         "scheduler": g_scheduler.state_dict()},  #生成器调度器状态
                        f"g_epoch_{epoch + 1}.pth.tar",
                        samples_dir,  #样本
                        results_dir,  #结果
                        "g_best.pth.tar",
                        "g_last.pth.tar",
                        is_best,
                        is_last)


def load_dataset() -> [CUDAPrefetcher, CUDAPrefetcher]:
    # 加载 Set5 训练、测试和验证数据集
    train_datasets = TrainValidImageDataset(srgan_config.train_gt_images_dir,  #训练数据集
                                            srgan_config.gt_image_size,  #图像大小
                                            srgan_config.upscale_factor,  #放大因子
                                            "Train")
    test_datasets = TestImageDataset(srgan_config.test_gt_images_dir, srgan_config.test_lr_images_dir)

    # 创建训练和测试数据集的数据加载器
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=srgan_config.batch_size,  #批量大小
                                  shuffle=True,  #是否打乱数据
                                  num_workers=srgan_config.num_workers,  #工作线程数
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

    # 将所有数据放入预处理数据加载器中
    train_prefetcher = CUDAPrefetcher(train_dataloader, srgan_config.device)  #创建预处理加载器，传入训练数据加载器和设备类型参数
    test_prefetcher = CUDAPrefetcher(test_dataloader, srgan_config.device)  #创建测试数据加载器，传入测试数据加载器和设备类型参数

    return train_prefetcher, test_prefetcher


def build_model() -> [nn.Module, nn.Module, nn.Module]:
    d_model = model.__dict__[srgan_config.d_arch_name]()  #创建判别器模型，通过字典访问model中的判别器类，并实例化一个判别器对象
    g_model = model.__dict__[srgan_config.g_arch_name](in_channels=srgan_config.in_channels,  #输入通道数
                                                       out_channels=srgan_config.out_channels,  #输出通道数
                                                       channels=srgan_config.channels,  #生成器模型中的通道数
                                                       num_rcb=srgan_config.num_rcb)  #生成器模型中的残差块
    d_model = d_model.to(device=srgan_config.device)  #创建判别器实例并赋值给d_model
    g_model = g_model.to(device=srgan_config.device)  #创建生成器模型的实例并赋值给g_model

    return d_model, g_model


def define_loss() -> [nn.MSELoss, model.content_loss, nn.BCEWithLogitsLoss]:
    pixel_criterion = nn.MSELoss()  #创建像素损失函数（使用了MSE均方误差函数）
    # 使用模块 model中的内容损失类 model.content_loss。该类需要特征提取器中的参数
    content_criterion = model.content_loss(feature_model_extractor_node=srgan_config.feature_model_extractor_node,  #特征模型提取节点
                                           feature_model_normalize_mean=srgan_config.feature_model_normalize_mean,  #特征模型的归一化均值
                                           feature_model_normalize_std=srgan_config.feature_model_normalize_std)  #和特征模型的归一化标准差
    adversarial_criterion = nn.BCEWithLogitsLoss()  #创建对抗性损失函数（使用二进制交叉熵损失函数）

    # 将损失函数转移到指定设备CUDA
    pixel_criterion = pixel_criterion.to(device=srgan_config.device)  #像素损失函数部署
    content_criterion = content_criterion.to(device=srgan_config.device)  #内容损失函数部署
    adversarial_criterion = adversarial_criterion.to(device=srgan_config.device)  #对抗损失函数部署

    return pixel_criterion, content_criterion, adversarial_criterion


def define_optimizer(d_model, g_model) -> [optim.Adam, optim.Adam]:
    d_optimizer = optim.Adam(d_model.parameters(),  #优化器
                             srgan_config.model_lr,  #学习率
                             srgan_config.model_betas,  #Adam优化算法中的beta参数
                             srgan_config.model_eps,  #Adam优化算法中的epsilon参数
                             srgan_config.model_weight_decay)  #权重衰减
    g_optimizer = optim.Adam(g_model.parameters(),  #优化器
                             srgan_config.model_lr,  #学习率
                             srgan_config.model_betas,  #Adam优化算法中的beta参数
                             srgan_config.model_eps,  #Adam优化算法中的epsilon参数
                             srgan_config.model_weight_decay)  #权重衰减（减少过拟合）

    return d_optimizer, g_optimizer


def define_scheduler(  #定义学习率调整期
        d_optimizer: optim.Adam,  #创建判别优化器d_optimizer利用Adam优化算法
        g_optimizer: optim.Adam  #创建生成优化器g_optimizer利用Adam优化算法
) -> [lr_scheduler.StepLR, lr_scheduler.StepLR]:  #学习率调整器为StepLR调度器
    d_scheduler = lr_scheduler.StepLR(d_optimizer,  #判别优化器对象
                                      srgan_config.lr_scheduler_step_size,
                                      srgan_config.lr_scheduler_gamma)  #学习率调整的乘法因子
    g_scheduler = lr_scheduler.StepLR(g_optimizer,
                                      srgan_config.lr_scheduler_step_size,  #每经过多少轮epochs进行一次学习率的调整
                                      srgan_config.lr_scheduler_gamma)
    return d_scheduler, g_scheduler


def train(
        d_model: nn.Module,
        g_model: nn.Module,
        train_prefetcher: CUDAPrefetcher,  #训练数据加载器
        pixel_criterion: nn.MSELoss,  #像素损失函数（均方误差损失）
        content_criterion: model.content_loss,  #内容损失函数
        adversarial_criterion: nn.BCEWithLogitsLoss,  #对抗损失函数
        d_optimizer: optim.Adam,  #判别优化器
        g_optimizer: optim.Adam,  #生成优化器
        epoch: int,
        writer: SummaryWriter  #用于记录训练日志的SummaryWriter对象
) -> None:
    # 计算每轮次训练中数据批次的数量
    batches = len(train_prefetcher)
    # 打印进度条信息
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    pixel_losses = AverageMeter("Pixel loss", ":6.6f")
    content_losses = AverageMeter("Content loss", ":6.6f")
    adversarial_losses = AverageMeter("Adversarial loss", ":6.6f")
    d_gt_probabilities = AverageMeter("D(GT)", ":6.3f")
    d_sr_probabilities = AverageMeter("D(SR)", ":6.3f")
    progress = ProgressMeter(batches,
                             [batch_time, data_time,
                              pixel_losses, content_losses, adversarial_losses,
                              d_gt_probabilities, d_sr_probabilities],
                             prefix=f"Epoch: [{epoch + 1}]")

    # 将对抗网络设置为训练模式
    d_model.train()
    g_model.train()

    # 初始化终端打印日志的数据批次数
    batch_index = 0

    #初始化数据加载器并加载第一个批次的数据
    train_prefetcher.reset()  #重置数据预处理加载器
    batch_data = train_prefetcher.next()  #获取数据加载器中的下一个批次的数据并赋值给batch_data

    #获取初始化训练时间
    end = time.time()

    while batch_data is not None:
        #计算加载一批数据所需的时间
        data_time.update(time.time() - end)

        #将内存中的数据转移到CUDA设备上加速训练
        gt = batch_data["gt"].to(device=srgan_config.device, non_blocking=True)  #转移至GPU设备设置非阻塞模式
        lr = batch_data["lr"].to(device=srgan_config.device, non_blocking=True)  #转移至GPU设备设置非阻塞模式

        #将真实样本的标签设置为1，将虚假样本的标签设置为0
        batch_size, _, height, width = gt.shape
        real_label = torch.full([batch_size, 1], 1.0, dtype=gt.dtype, device=srgan_config.device)
        fake_label = torch.full([batch_size, 1], 0.0, dtype=gt.dtype, device=srgan_config.device)

        # 开始训练判别器模型
        # 在判别器模型训练期间，启用判别器模型的反向传播
        for d_parameters in d_model.parameters():  #在训练模式下的判别优化器模型进行遍历

            d_parameters.requires_grad = True  #启用对鉴别器模型的梯度计算和参数更新

        # 鉴别器梯度初始化
        d_model.zero_grad(set_to_none=True)  #梯度设置为None

        #计算鉴别器对于真实图像的得分
        gt_output = d_model(gt)  #将gt作为输入，通过鉴别器模型 d_model 进行前向传播（得到输出），计算真实样本被判别为真实的分数或概率（gt_output）
        d_loss_gt = adversarial_criterion(gt_output, real_label)  #得到真实样本与输出结果之间的损失值
        # 调用混合精度API中的梯度缩放函数
        # 反向传播生成样本的梯度信息
        d_loss_gt.backward(retain_graph=True)  #计算损失函数对判别器模型参数的梯度，进行反向传播（计算损失函数对模型的梯度）

        # 计算鉴别器模型对假样本的分类分数
        # 使用生成器模型生成生成样本
        sr = g_model(lr)  #使用生成器模型将低分辨率图像生成高分辨率图像
        sr_output = d_model(sr.detach().clone())  #将生成的超分辨率图像通过判别器模型进行前向传播，得到判别器的输出。detach().clone()的作用是创建sr的副本并断开与生成器的梯度关联，以便只更新判别器的参数
        d_loss_sr = adversarial_criterion(sr_output, fake_label)  #得到生成样本与假样本标签的损失值
        # 调用混合精度API中的梯度缩放函数
        # 反向传播假样本的梯度信息
        d_loss_sr.backward()  #将生成损失的梯度信息传播回判别器

        # 计算总判别器损失值
        d_loss = d_loss_gt + d_loss_sr

        #提高判别器甄别真假图像的能力
        d_optimizer.step()  #优化判别器
        #判别器的训练完成

        # 开始训练生成器
        # 在生成器训练期间，关闭鉴别器反向传播
        for d_parameters in d_model.parameters():
            d_parameters.requires_grad = False

        # 初始化生成器梯度
        g_model.zero_grad(set_to_none=True)  #梯度为None

        # 计算损失函数
        pixel_loss = srgan_config.pixel_weight * pixel_criterion(sr, gt)
        content_loss = srgan_config.content_weight * content_criterion(sr, gt)
        adversarial_loss = srgan_config.adversarial_weight * adversarial_criterion(d_model(sr), real_label)
        # 计算总损失
        g_loss = pixel_loss + content_loss + adversarial_loss
        # 调用混合精度API中的梯度缩放函数
        # 反向传播假样本的梯度信息
        g_loss.backward()

        # 优化生成器
        g_optimizer.step()
        #生成训练完成

        # 计算鉴别器对真实样本和假样本的得分
        # 真实样本得分为1，假样本得分为0
        d_gt_probability = torch.sigmoid_(torch.mean(gt_output.detach()))  #使用detach方法将输出从计算图中分离，确保计算不会影响后续梯度计算和参数更新
        d_sr_probability = torch.sigmoid_(torch.mean(sr_output.detach()))  #使用detach方法将输出从计算图中分离，确保计算不会影响后续梯度计算和参数更新

        #统计精度和损失值
        pixel_losses.update(pixel_loss.item(), lr.size(0))  #更新像素损失的统计信息，(pixel_loss.item)获取像素损失的数值，lr.size(0)获取当前批次的样本数量，通过调用update方法将损失值和样本数量传递给AverageMeter对象，更新统计信息
        content_losses.update(content_loss.item(), lr.size(0))  #更新内容损失的统计信息
        adversarial_losses.update(adversarial_loss.item(), lr.size(0))  #更新对抗损失的统计信息
        d_gt_probabilities.update(d_gt_probability.item(), lr.size(0))  # 更新判别器对真实样本的概率值的统计信息
        d_sr_probabilities.update(d_sr_probability.item(), lr.size(0))  #更新判别器对生成样本的概率值的统计信息

        # 计算set5训练一批数据所需的时间
        batch_time.update(time.time() - end)
        end = time.time()

        # 将训练期间的数据写入训练日志文件
        if batch_index % srgan_config.train_print_frequency == 0:  #控制打印和记录统计信息的频率
            iters = batch_index + epoch * batches + 1  #计算当前的训练迭代次数，它是当前批次的索引加上之前训练过的批次数量
            writer.add_scalar("Train/D_Loss", d_loss.item(), iters)  #将判别器的损失值记录到训练日志
            writer.add_scalar("Train/G_Loss", g_loss.item(), iters)  #将生成器的损失值、像素损失、内容损失、对抗损失以及判别器对真实样本和生成样本的概率值记录到训练日志中
            writer.add_scalar("Train/Pixel_Loss", pixel_loss.item(), iters)
            writer.add_scalar("Train/Content_Loss", content_loss.item(), iters)
            writer.add_scalar("Train/Adversarial_Loss", adversarial_loss.item(), iters)
            writer.add_scalar("Train/D(GT)_Probability", d_gt_probability.item(), iters)
            writer.add_scalar("Train/D(SR)_Probability", d_sr_probability.item(), iters)
            progress.display(batch_index + 1)  #使用ProgressMeter的display方法打印当前训练进度信息，包括当前批次的索引和各项统计信息

        # 预加载下一批数据
        batch_data = train_prefetcher.next()

        # 训练完一批数据后，在数据批数上加1，以保证terminal正常打印数据
        batch_index += 1


def validate(
        g_model: nn.Module,
        data_prefetcher: CUDAPrefetcher,  #数据加载器
        epoch: int,
        writer: SummaryWriter,
        psnr_model: nn.Module,
        ssim_model: nn.Module,
        mode: str
) -> [float, float]:
    # 计算每个Epoch中的批数据
    batch_time = AverageMeter("Time", ":6.3f")  #每个批次的处理时间计量器
    psnres = AverageMeter("PSNR", ":4.2f")
    ssimes = AverageMeter("SSIM", ":4.4f")
    progress = ProgressMeter(len(data_prefetcher), [batch_time, psnres, ssimes], prefix=f"{mode}: ")

    # 将对抗网络模型置于验证模式
    g_model.eval()

    # 初始化终端打印日志的数据批次数
    batch_index = 0

    # 初始化数据加载器并加载第一批数据
    data_prefetcher.reset()  #将数据加载器状态初始化
    batch_data = data_prefetcher.next()  #获取下一个批次的数据，返回一个字典，包含当前批次的输入数据（lr）和目标数据（gt）

    # 获取初始化测试时间
    end = time.time()

    with torch.no_grad():
        while batch_data is not None:
            # 将内存中的数据传输到CUDA设备以加快测试速度
            gt = batch_data["gt"].to(device=srgan_config.device, non_blocking=True)
            lr = batch_data["lr"].to(device=srgan_config.device, non_blocking=True)

            # 使用生成器模型生成一个生成样本
            sr = g_model(lr)

            # 统计损失值
            psnr = psnr_model(sr, gt)
            ssim = ssim_model(sr, gt)
            psnres.update(psnr.item(), lr.size(0))  #用于更新 PSNR的平均值
            ssimes.update(ssim.item(), lr.size(0))  #用于更新 SSIM的平均值

            # 计算一个数据批次所用时间
            batch_time.update(time.time() - end)
            end = time.time()

            # 记录日志信息
            if batch_index % srgan_config.valid_print_frequency == 0:
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
