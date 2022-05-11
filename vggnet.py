# https://arxiv.org/pdf/1409.1556.pdf
# VggNet에서는 5x5 또는 7x7 conv를 사용하는 대신 3x3 conv를 여러겹 사용해서 동일한 효과를 냈다.
# 동일한 receptive field를 가지면서 model의 network parameter를 획기적으로 줄일 수 있었다.

# 모델 파라미터 수 확인: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7

# TODO 단순 sequential말고 함수형태로 모델사이즈를 결정할 수 있도록 해보자.
# 참고: https://github.com/pytorch/vision/blob/e6edcef423e14adb226923889d77bf751956a9cd/torchvision/models/alexnet.py

import torch
import torch.nn as nn
import numpy as np

class VggNet19(nn.Module): 
    def __init__(self):
        super(VggNet19, self).__init__() # super(하위 클래스 이름, ,self) 함수는 부모 클래스를 호출하는 함수.

        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64, kernel_size=3, stride=1, padding=1), # image 224 -> 224
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=64, kernel_size=3, stride=1, padding=1), # image 224 -> 224
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2), # image 224 -> 112
            
            nn.Conv2d(in_channels=64,out_channels=128, kernel_size=3, stride=1, padding=1), # image 112 -> 112
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=128, kernel_size=3, stride=1, padding=1), # image 112 -> 112
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), #image 112 -> 56

            nn.Conv2d(in_channels=128,out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), #image 56 -> 28

            nn.Conv2d(in_channels=256,out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), #image 28 -> 14

            nn.Conv2d(in_channels=512,out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), #image 14 -> 7

        )
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1000)
        )

    def forward(self, x):
        x = self.convnet(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__=="__main__":
    model = VggNet19()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    
    print(model.parameters)
    print("Number of parameters: ", params)