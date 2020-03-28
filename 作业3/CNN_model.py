import numpy as np
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            # 第一层卷积层和池化层的配置
            # 第一个参数代表输入数据的通道数，例RGB图片通道数为3；
            # 第二个参数代表输出数据的通道数，这个根据模型调整；
            # 第三个参数是卷积核大小
            # 第四个参数是stride，步长
            # 第五个参数是padding，补1
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]，即输出是64*128*128维的图片

            nn.BatchNorm2d(64),  # 防止梯度消失或爆炸，设置的参数就是卷积的输出通道数
            nn.ReLU(),  # activate函数是relu函数

            # 第一个参数是kernel_size，max pooling的窗口大小，
            # 第二个参数是stride，max pooling的窗口移动的步长。默认值是kernel_size
            # 第三个参数输入的每一条边补充0的层数，默认是0
            nn.MaxPool2d(2, 2, 0),  # [64, 64, 64]

            # 第二层卷积层和池化层的配置
            nn.Conv2d(64, 128, 3, 1, 1),  # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [128, 32, 32]

            # 第三层卷积层和池化层的配置
            nn.Conv2d(128, 256, 3, 1, 1),  # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [256, 16, 16]

            # 第四层卷积层和池化层的配置
            nn.Conv2d(256, 512, 3, 1, 1),  # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 8, 8]

            # 第五层卷积层和池化层的配置
            nn.Conv2d(512, 512, 3, 1, 1),  # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()


if __name__ == '__main__':
    model = Classifier()
    print(params_count(model))
