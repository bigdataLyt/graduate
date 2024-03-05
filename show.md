## 结果

原图链接: [https://arxiv.org/pdf/1609.04802v5.pdf](https://arxiv.org/pdf/1609.04802v5.pdf)

以下是各个数据集在本次超分重建算法中得到的PSNR以及SSIM平均结果的可视化展示

```bash
本次所用神经网络整体结构图如下：
```
<span align="center"><img width="950" height="640" src="data/img.png"/></span>

| Set14 | Scale |   SRResNet   |    SRGAN     |
|:----:|:-----:|:------------:|:------------:|
| PSNR |   4   | (**28.05**)  | (**27.43**)  |
| SSIM |   4   | (**0.8212**) | (**0.8060**) |

| Set14 | Scale |   SRResNet   |    SRGAN     |
|:-----:|:-----:|:------------:|:------------:|
| PSNR  |   4   | (**28.57**)  | (**27.12**)  |
| SSIM  |   4   | (**0.7815**) | (**0.7321**) |

```text
Set14数据集上的效果展示：
```
Input: 

<span align="center"><img width="240" height="240" src="data/Set14/LRbicx4/baboon.png"/></span>
<span align="center"><img width="240" height="240" src="data/Set14/LRbicx4/barbara.png"/></span>
<span align="center"><img width="240" height="240" src="data/Set14/LRbicx4/face.png"/></span>
<span align="center"><img width="240" height="240" src="data/Set14/LRbicx4/man.png"/></span>
<span align="center"><img width="160" height="240" src="data/Set14/LRbicx4/zebra.png"/></span>

Output: 

<span align="center"><img width="240" height="240" src="results/test/Set14/SRGAN_x4-DIV2K/baboon.png"/></span>
<span align="center"><img width="240" height="240" src="results/test/Set14/SRGAN_x4-DIV2K/barbara.png"/></span>
<span align="center"><img width="240" height="240" src="results/test/Set14/SRGAN_x4-DIV2K/face.png"/></span>
<span align="center"><img width="240" height="240" src="results/test/Set14/SRGAN_x4-DIV2K/man.png"/></span>
<span align="center"><img width="160" height="240" src="results/test/Set14/SRGAN_x4-DIV2K/zebra.png"/></span>

`此处为与SRResNet的生成图像做对比：`

<span align="center"><img width="240" height="240" src="results/test/Set14/SRResNet_x4-DIV2K/baby.png"/></span>
<span align="center"><img width="240" height="240" src="results/test/Set14/SRResNet_x4-DIV2K/bird.png"/></span>
<span align="center"><img width="240" height="240" src="results/test/Set14/SRResNet_x4-DIV2K/butterfly.png"/></span>
<span align="center"><img width="240" height="240" src="results/test/Set14/SRResNet_x4-DIV2K/head.png"/></span>
<span align="center"><img width="160" height="240" src="results/test/Set14/SRResNet_x4-DIV2K/woman.png"/></span>
```text
此处可以与SRCNN的生成图像再做个对比：
```
<span align="center"><img width="240" height="240" src="data\baby_srcnn_x3.png"/></span>
<span align="center"><img width="240" height="240" src="data\bird_srcnn_x3.png"/></span>
<span align="center"><img width="240" height="240" src="data\butterfly_srcnn_x3.png"/></span>
<span align="center"><img width="240" height="240" src="data\head_srcnn_x3.png"/></span>
<span align="center"><img width="160" height="240" src="data\woman_srcnn_x3.png"/></span>
```text
可以看出，作为经典的超分重建算法的SRCNN，在原比例超分重建下得到的效果还是很令人满意的。不过这不是本次实验探讨重点。
本次实验研究的是在对图像大小进行压缩处理后的超分重建。
```
```text
因此，我们现在来对比在局部放大的图像上，SRCNN与SRGAN算法的优劣性：(左侧是本次实验原图，右侧为重建后图像。)
```
<span align="center"><img width="425" height="320" src="data/original.png"/></span>
`SRCNN效果图：`
<span align="center"><img width="240" height="240" src="data/2_srcnn_x3.png"/></span>
`SRGAN效果图：`
<span align="center"><img width="240" height="240" src="figure/2.sr.png"/></span>
```text
由以上几张图片可以明显看出，SRGAN算法在高倍放大图像的超分重建方面表现更为出色！
```
```bash
Set14数据集部分效果展示：
```
Input:

<span align="center"><img width="240" height="240" src="data/Set14/LRbicx4/baboon.png"/></span>
<span align="center"><img width="300" height="240" src="data/Set14/LRbicx4/barbara.png"/></span>
<span align="center"><img width="240" height="240" src="data/Set14/LRbicx4/bridge.png"/></span>
<span align="center"><img width="240" height="240" src="data/Set14/LRbicx4/coastguard.png"/></span>
<span align="center"><img width="160" height="240" src="data/Set14/LRbicx4/comic.png"/></span>
```text
以下为SRResNet的重建效果：
```
<span align="center"><img width="240" height="240" src="results/test/Set14/SRResNet_x4-DIV2K/baboon.png"/></span>
<span align="center"><img width="300" height="240" src="results/test/Set14/SRResNet_x4-DIV2K/barbara.png"/></span>
<span align="center"><img width="240" height="240" src="results/test/Set14/SRResNet_x4-DIV2K/bridge.png"/></span>
<span align="center"><img width="240" height="240" src="results/test/Set14/SRResNet_x4-DIV2K/coastguard.png"/></span>
<span align="center"><img width="160" height="240" src="results/test/Set14/SRResNet_x4-DIV2K/comic.png"/></span>
```text
再来看下SRGAN的重建效果：
```
<span align="center"><img width="240" height="240" src="results/test/Set14/SRGAN_x4-DIV2K/baboon.png"/></span>
<span align="center"><img width="300" height="240" src="results/test/Set14/SRGAN_x4-DIV2K/barbara.png"/></span>
<span align="center"><img width="240" height="240" src="results/test/Set14/SRGAN_x4-DIV2K/bridge.png"/></span>
<span align="center"><img width="240" height="240" src="results/test/Set14/SRGAN_x4-DIV2K/coastguard.png"/></span>
<span align="center"><img width="160" height="240" src="results/test/Set14/SRGAN_x4-DIV2K/comic.png"/></span>
```text
效果很棒！！🌼🌼
```