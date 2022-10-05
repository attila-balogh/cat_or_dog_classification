"""
MODEL 3
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ResidualBlock import ResidualBlock


class Model3(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm_conv_1 = nn.BatchNorm2d(32)
        self.maxpool2d_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.10)

        self.resblock1 = ResidualBlock(channels=32)
        self.resblock2 = ResidualBlock(channels=32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm_conv_2 = nn.BatchNorm2d(64)
        self.maxpool2d_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.15)

        self.resblock3 = ResidualBlock(channels=64)
        self.resblock4 = ResidualBlock(channels=64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm_conv_3 = nn.BatchNorm2d(128)
        self.maxpool2d_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.20)

        self.resblock5 = ResidualBlock(channels=128)
        self.resblock6 = ResidualBlock(channels=128)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm_conv_4 = nn.BatchNorm2d(256)
        self.maxpool2d_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout(0.20)

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm_conv_5 = nn.BatchNorm2d(256)
        self.maxpool2d_5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout5 = nn.Dropout(0.10)

        self.fc1 = nn.Linear(in_features=256 * int(image_size[0] / 32) * int(image_size[1] / 32), out_features=1024)
        self.dropout6 = nn.Dropout(0.30)

        self.out = nn.Linear(in_features=1024, out_features=2)

    def forward(self, t):
        t = self.dropout1(self.maxpool2d_1(self.batchnorm_conv_1(F.relu(self.conv1(t)))))
        t = self.resblock1(t)
        t = self.resblock2(t)
        t = self.dropout2(self.maxpool2d_2(self.batchnorm_conv_2(F.relu(self.conv2(t)))))
        t = self.resblock3(t)
        t = self.resblock4(t)
        t = self.dropout3(self.maxpool2d_3(self.batchnorm_conv_3(F.relu(self.conv3(t)))))
        t = self.resblock5(t)
        t = self.resblock6(t)
        t = self.dropout4(self.maxpool2d_4(self.batchnorm_conv_4(F.relu(self.conv4(t)))))
        t = self.dropout5(self.maxpool2d_5(self.batchnorm_conv_5(F.relu(self.conv5(t)))))

        t = torch.flatten(t, start_dim=1)
        t = self.dropout6(F.relu(self.fc1(t)))
        t = F.relu(self.out(t))

        return t
