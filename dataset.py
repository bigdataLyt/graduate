import os
import queue
import threading

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import imgproc

__all__ = [
    "TrainValidImageDataset", "TestImageDataset",
    "PrefetchGenerator", "PrefetchDataLoader", "CPUPrefetcher", "CUDAPrefetcher",
]


class TrainValidImageDataset(Dataset):
    """Define training/valid dataset loading methods.

    Args:
        gt_image_dir (str): Train/Valid dataset address.
        gt_image_size (int): Ground-truth resolution image size.
        upscale_factor (int): Image up scale factor.
        mode (str): Data set loading method, the training data set is for data enhancement, and the
            verification dataset is not for data enhancement.
    """

    def __init__(
            self,
            gt_image_dir: str,
            gt_image_size: int,
            upscale_factor: int,
            mode: str,  #数据集加载模式，用于训练数据集时数据增强
    ) -> None:
        super(TrainValidImageDataset, self).__init__()
        self.image_file_names = [os.path.join(gt_image_dir, image_file_name) for image_file_name in
                                 os.listdir(gt_image_dir)]
        self.gt_image_size = gt_image_size
        self.upscale_factor = upscale_factor
        self.mode = mode

    def __getitem__(self, batch_index: int) -> [Tensor, Tensor]:
        # 读取图像文件，做预处理
        gt_image = cv2.imread(self.image_file_names[batch_index]).astype(np.float32) / 255.  #读取目标图像，将图像数据类型转换为float32，再通过/255将像素范围归一化到0~1，方便深度学习模型的处理和训练数据

        # Image processing operations
        if self.mode == "Train":
            gt_crop_image = imgproc.random_crop(gt_image, self.gt_image_size)  #对图像做随机剪裁 更鲁棒
        elif self.mode == "Valid":
            gt_crop_image = imgproc.center_crop(gt_image, self.gt_image_size)  #对图像做中心剪裁，可能会拟合
        else:
            raise ValueError("Unsupported data processing model, please use `Train` or `Valid`.")

        lr_crop_image = imgproc.image_resize(gt_crop_image, 1 / self.upscale_factor)

        # 图像颜色空间转换
        gt_crop_image = cv2.cvtColor(gt_crop_image, cv2.COLOR_BGR2RGB)
        lr_crop_image = cv2.cvtColor(lr_crop_image, cv2.COLOR_BGR2RGB)

        # 将图片数据返回为tensor格式
        # 将数据在0~1范围内进行归一化
        gt_crop_tensor = imgproc.image_to_tensor(gt_crop_image, False, False)
        lr_crop_tensor = imgproc.image_to_tensor(lr_crop_image, False, False)

        return {"gt": gt_crop_tensor, "lr": lr_crop_tensor}

    def __len__(self) -> int:
        return len(self.image_file_names)


class TestImageDataset(Dataset):


    def __init__(self, test_gt_images_dir: str, test_lr_images_dir: str) -> None:
        super(TestImageDataset, self).__init__()
        # Get all image file names in folder
        self.gt_image_file_names = [os.path.join(test_gt_images_dir, x) for x in os.listdir(test_gt_images_dir)]
        self.lr_image_file_names = [os.path.join(test_lr_images_dir, x) for x in os.listdir(test_lr_images_dir)]

    def __getitem__(self, batch_index: int) -> [torch.Tensor, torch.Tensor]:  #获取测试数据集中的样本
        # Read a batch of image data
        gt_image = cv2.imread(self.gt_image_file_names[batch_index]).astype(np.float32) / 255.  #读取目标图像，将图像数据类型转换为float32，再通过/255将像素范围归一化到0~1，方便深度学习模型的处理和训练数据
        lr_image = cv2.imread(self.lr_image_file_names[batch_index]).astype(np.float32) / 255.

        # BGR convert RGB
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        gt_tensor = imgproc.image_to_tensor(gt_image, False, False)
        lr_tensor = imgproc.image_to_tensor(lr_image, False, False)

        return {"gt": gt_tensor, "lr": lr_tensor}

    def __len__(self) -> int:  #确定数据集长度
        return len(self.gt_image_file_names)


class PrefetchGenerator(threading.Thread):   #数据预处理加速生成器，在后台线程中提前加载数据，减少数据加载和训练间的等待时间。


    def __init__(self, generator, num_data_prefetch_queue: int) -> None:
        threading.Thread.__init__(self)
        self.queue = queue.Queue(num_data_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self) -> None:
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader):  #利用PrefetchGenerator作为迭代器，实现数据加载过程中的数据预处理


    def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None:
        self.num_data_prefetch_queue = num_data_prefetch_queue
        super(PrefetchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)


class CPUPrefetcher:  #加速器，cpu端


    def __init__(self, dataloader: DataLoader) -> None:
        self.original_dataloader = dataloader
        self.data = iter(dataloader)

    def next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)


class CUDAPrefetcher:  #加速器，gpu端
    def __init__(self, dataloader: DataLoader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()
    def preload(self):  #数据预取与重置
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None
        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data
    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()
    def __len__(self) -> int:
        return len(self.original_dataloader)
