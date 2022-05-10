# https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
# https://seongkyun.github.io/study/2019/01/25/num_of_parameters/

import torch
import torch.nn as nn

class AlexNet(nn.Module): # 클래스 AlexNet은 nn.Module 클래스를 상속받는다.
    def __init__(self):
        super(AlexNet, self).__init__() # super(하위 클래스 이름, ,self) 함수는 부모 클래스를 호출하는 함수.

        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=96, kernel_size=11, stride=4, padding=0), # image 227 -> 55
            nn.ReLU(inplace=True), # inplace 하면 input으로 들어온 것 자체를 수정하겠다는 뜻. 메모리 usage가 좀 좋아짐. 하지만 input을 없앰.
            nn.LocalResponseNorm(size=5, k=2),
            nn.MaxPool2d(kernel_size=3,stride=2), # image 55 -> 27
            
            nn.Conv2d(in_channels=96,out_channels=256, kernel_size=5, stride=1, padding=2), # image 27 -> 27
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2), #image 27 -> 13

            nn.Conv2d(in_channels=256,out_channels=384, kernel_size=3, stride=1, padding=1), # image 13 -> 13
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2),

            nn.Conv2d(in_channels=384,out_channels=384, kernel_size=3, stride=1, padding=1), # image 13 -> 13
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2),

            nn.Conv2d(in_channels=384,out_channels=256, kernel_size=3, stride=1, padding=1), # image 13 -> 13
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2), #image 13 -> 6
        )
        self.densenet = nn.Sequential(
            nn.Linear(256*6*6, 4096),
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
        x = self.densenet(x)
        return x

# model = AlexNet()
# print(model)




