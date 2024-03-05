import math
from typing import Any
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F_torch
from torchvision import models
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor

__all__ = [
    "SRResNet", "Discriminator",
    "srresnet_x4", "discriminator", "content_loss",
]
class SRResNet(nn.Module):  #定义参差网络模型
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            channels: int,
            num_rcb: int,  #高频信息提取块
            upscale_factor: int
    ) -> None:
        super(SRResNet, self).__init__()
        # 低频信息提取层
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, (9, 9), (1, 1), (4, 4)),  #输入通道数，输出通道数，9*9卷积核，步长，填充
            nn.PReLU(),  #激活函数
        )
        # 高频信息提取层
        trunk = []  #用于保存残差卷积块
        for _ in range(num_rcb):  #迭代4次，创建残差卷积块：避免梯度消失或梯度爆炸
            trunk.append(_ResidualConvBlock(channels))
        self.trunk = nn.Sequential(*trunk)  #将残差卷积块组合成一个顺序的网络模块并赋值给trunk

        # 高频信息融合层
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),  #卷积层
            nn.BatchNorm2d(channels),  #归一化层
        )
        # 上采样层
        upsampling = []
        if upscale_factor == 2 or upscale_factor == 4 or upscale_factor == 8:
            for _ in range(int(math.log(upscale_factor, 2))):
                upsampling.append(_UpsampleBlock(channels, 2))
        elif upscale_factor == 3:
            upsampling.append(_UpsampleBlock(channels, 3))
        self.upsampling = nn.Sequential(*upsampling)
        # 重建块
        self.conv3 = nn.Conv2d(channels, out_channels, (9, 9), (1, 1), (4, 4))
        # 初始化神经网络权重
        self._initialize_weights()
    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    # 超分辨率重建
    def _forward_impl(self, x: Tensor) -> Tensor:
        out1 = self.conv1(x)  #一层提取
        out = self.trunk(out1)  #一层残差（4次）
        out2 = self.conv2(out)  #一层提取
        out = torch.add(out1, out2)  #采用了残差原理保留前三层的特征
        out = self.upsampling(out)  #拿去上采样
        out = self.conv3(out)  #重建
        out = torch.clamp_(out, 0.0, 1.0)  #截断操作，将像素值限制在【0~1】，避免数值溢出导致梯度爆炸
        return out
    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)  #初始化模型权重
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)  #激活值为0
            elif isinstance(module, nn.BatchNorm2d):  #初始归一化层
                nn.init.constant_(module.weight, 1)  #激活值为1
  #鉴别器网络模型特征提取层和分类器层
class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            # 接受大小为3（RGB通道）的96x96的图像。
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),  #泄露线性整流激活函数
            #  (64) x 48 x 48
            nn.Conv2d(64, 64, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),  #归一化处理卷积层
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # (128) x 24 x 24
            nn.Conv2d(128, 128, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # (256) x 12 x 12
            nn.Conv2d(256, 256, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            #  (512) x 6 x 6
            nn.Conv2d(512, 512, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 6 * 6, 1024),  #线性层将输入的通道数与1024个神经元连接
            nn.LeakyReLU(0.2, True),  #负斜率为0.2的激活函数，输出值是输入值的0.2倍。第二个参数True表示将输入值小于0的部分替换为负斜率的乘积
            nn.Linear(1024, 1),  #最终得到一个标量值
        )
    def forward(self, x: Tensor) -> Tensor:
        # 输入图像标准化
        assert x.shape[2] == 96 and x.shape[3] == 96, "Image shape must equal 96x96"
        out = self.features(x)  #提取特征
        out = torch.flatten(out, 1)  #扁平化，使输出结果为一维张量
        out = self.classifier(out)  #输出判别结果
        return out
  #定义残差卷积块
class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(_ResidualConvBlock, self).__init__()
        self.rcb = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),  #卷积层
            nn.BatchNorm2d(channels),  #归一化处理
            nn.PReLU(),  #激活函数
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),  #归一化处理
        )
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.rcb(x)
        out = torch.add(out, identity)
        return out
class _UpsampleBlock(nn.Module):  #上采样
    def __init__(self, channels: int, upscale_factor: int) -> None:
        super(_UpsampleBlock, self).__init__()
        self.upsample_block = nn.Sequential(
            nn.Conv2d(channels, channels * upscale_factor * upscale_factor, (3, 3), (1, 1), (1, 1)),  #卷积层
            nn.PixelShuffle(2),  #像素混洗，对张量的通道进行混洗，让不同通道的数据建立关系。上采样因子为2，将空间维度放大2倍。通道数为channels
            nn.PReLU(),  #激活函数
        )
    def forward(self, x: Tensor) -> Tensor:  # 接收张量
        out = self.upsample_block(x)
        return out
class _ContentLoss(nn.Module):  #定义计算内容损失模块
    def __init__(
            self,
            feature_model_extractor_node: str,  #指定的特征提取节点的名称
            feature_model_normalize_mean: list,  #输入数据的预处理均值列表
            feature_model_normalize_std: list  #输入数据的预处理标准差列表
    ) -> None:
        super(_ContentLoss, self).__init__()
        # 创建了一个特征提取器feature_extractor并指定输出内容损失的节点
        self.feature_model_extractor_node = feature_model_extractor_node
        # 使用的模型为VGG19模型，该模型是在ImageNet数据集上做的训练，并保留了训练过程中的函数权重，激活值，损失函数等，用于本模型的初次训练
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        #创建特征提取器（feature_extractor），该提取器使用的模型为VGG19模型，并指定了要提取的特征节点
        self.feature_extractor = create_feature_extractor(model, [feature_model_extractor_node])
        #将特征提取器设置为评估模式
        self.feature_extractor.eval()
        # 输入数据的预处理方法。采用VGG模型做预处理方法。
        self.normalize = transforms.Normalize(feature_model_normalize_mean, feature_model_normalize_std)
        # 冻结模型参数，防止在后续训练的反向传递过程中改变VGG模型的参数和权重
        for model_parameters in self.feature_extractor.parameters():
            model_parameters.requires_grad = False
    def forward(self, sr_tensor: Tensor, gt_tensor: Tensor) -> Tensor:
        sr_tensor = self.normalize(sr_tensor)  # 对生成的张量进行标准化操作
        gt_tensor = self.normalize(gt_tensor)  # 对目标张量进行标准化操作
        sr_feature = self.feature_extractor(sr_tensor)[self.feature_model_extractor_node]  # 提取生成张量的特征
        gt_feature = self.feature_extractor(gt_tensor)[self.feature_model_extractor_node]  # 提取目标张量的特征
        loss = F_torch.mse_loss(sr_feature, gt_feature)  # 计算特征之间的均方误差损失
        return loss  # 返回损失值
def srresnet_x4(**kwargs: Any) -> SRResNet:
    model = SRResNet(upscale_factor=4, **kwargs)
    return model
def discriminator() -> Discriminator:
    model = Discriminator()
    return model
def content_loss(**kwargs: Any) -> _ContentLoss:
    content_loss = _ContentLoss(**kwargs)
    return content_loss
