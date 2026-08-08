import torch
import torch.nn as nn

class SceneVisionCNN(nn.Module):

    def __init__(self,NUM_CLASSES):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)

        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels= 32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(20736,512),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(512,NUM_CLASSES)
        )

    def forward(self,x):

        x = self.conv_block1(x)
        feature_map1 = x

        x = self.conv_block2(x)
        feature_map2 = x

        x = self.conv_block3(x)
        feature_map3 = x

        x = self.conv_block4(x)
        feature_map4 = x

        x = self.classifier(x)

        return x