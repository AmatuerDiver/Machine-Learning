# Import需要的套件
import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

"""#Read image
利用 OpenCV (cv2) 讀入照片並存放在 numpy array 中
"""

# 当label等于false时即不需要返回label，用来处理test集，方便统一管理
def readfile(path, label):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img, (128, 128))
        if label:
            y[i] = int(file.split("_")[0])
    if label:
        return x, y
    else:
        return x


# 集成了一个 Dataset类之后，我们需要重写 len 方法，该方法提供了dataset的大小；
# getitem 方法， 该方法支持从 0 到 len(self)的索引
class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:  # 如果没有标签那么只返回X
            return X


if __name__ == '__main__':
    print("Reading data")
    train_x, train_y = readfile("./dataset/training", True)
    val_x, val_y = readfile("./dataset/validation", True)
    test_x = readfile("./dataset/testing", False)

    train_transform = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.RandomHorizontalFlip(),  # 随即将图片水平翻转
        # transforms.RandomRotation(15),  # 随即旋转图片15度
        transforms.ToTensor(),  # 将图片转成 Tensor
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
    ])
    # testing 時不需做 data augmentation
    test_transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.ToTensor(),  # 将图片转成 Tensor
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
    ])

    batch_size = 50
    train_set = ImgDataset(train_x, train_y, train_transform)
    test_set = ImgDataset(test_x, transform=test_transform)
    """
    Normlize
    """
    dataloader1 = DataLoader(train_set, batch_size=50, shuffle=False, num_workers=4)
    dataloader2 = DataLoader(test_set, batch_size=20, shuffle=False, num_workers=4)

    pop_mean0 = []
    pop_std0 = []

    pop_mean1 = []
    pop_std1 = []

    for i, data in enumerate(dataloader1):
        # shape (batch_size, 3, height, width)
        numpy_image = data[0].numpy()  # 将数据转换成矩阵

        # shape (3,)
        batch_mean = np.mean(numpy_image, axis=(0, 2, 3))  # 计算mean
        batch_std0 = np.std(numpy_image, axis=(0, 2, 3))  # 计算std

        pop_mean0.append(batch_mean)
        pop_std0.append(batch_std0)

    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    pop_mean0 = np.array(pop_mean0).mean(axis=0)
    pop_std0 = np.array(pop_std0).mean(axis=0)

    for i, data in enumerate(dataloader2):
        # shape (batch_size, 3, height, width)
        numpy_image = data.numpy()

        # shape (3,)
        batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
        batch_std0 = np.std(numpy_image, axis=(0, 2, 3))

        pop_mean1.append(batch_mean)
        pop_std1.append(batch_std0)

    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    pop_mean1 = np.array(pop_mean1).mean(axis=0)
    pop_std1 = np.array(pop_std1).mean(axis=0)

    print(pop_mean0, pop_std0)
    print(pop_mean1, pop_std1)
