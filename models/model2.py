"""
MODEL 2
"""
import torch.nn as nn

from ResidualBlock import ResidualBlock

image_size = (256, 256)

model2 = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
    nn.ReLU(),
    nn.BatchNorm2d(16),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
    nn.ReLU(),
    nn.BatchNorm2d(64),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
    nn.ReLU(),
    nn.BatchNorm2d(64),
    nn.MaxPool2d(kernel_size=2, stride=2),
    ResidualBlock(channels=64),
    ResidualBlock(channels=64),
    ResidualBlock(channels=64),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
    nn.ReLU(),
    nn.BatchNorm2d(128),
    nn.MaxPool2d(kernel_size=2, stride=2),
    ResidualBlock(channels=128),
    ResidualBlock(channels=128),
    ResidualBlock(channels=128),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
    nn.ReLU(),
    nn.BatchNorm2d(256),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(start_dim=1),
    nn.Linear(in_features=256 * int(image_size[0] / 128) * int(image_size[1] / 128), out_features=2048),
    nn.ReLU(),
    nn.BatchNorm1d(2048),
    nn.Linear(in_features=2048, out_features=1024),
    nn.ReLU(),
    nn.BatchNorm1d(1024),
    nn.Linear(in_features=1024, out_features=64),
    nn.ReLU(),
    nn.BatchNorm1d(64),
    nn.Linear(in_features=64, out_features=2)
)
