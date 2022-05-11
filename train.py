# TODO Parameter 초기화부분 확인

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import time
from alexnet import AlexNet
device = 'cpu'

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(227),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testset = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

def init_weights(m):
    if type(m) not in [nn.ReLU, nn.LocalResponseNorm, nn.MaxPool2d, nn.Sequential, nn.Dropout, AlexNet]:
        nn.init.normal_(m.weight, 0, 0.01)
        m.bias.data.fill_(1)

model = AlexNet()
model.apply(init_weights)
model = model.to(device)

# 파라미터 초기화 확인
# for p in model.parameters():
#     print(p)
#     break

criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01, momentum=0.9, weight_decay=0.0005)

start_time = time.time()
min_loss = int(1e9)
history = []
for epoch in range(100):  # loop over the dataset multiple times
    epoch_loss = 0.0
    tk0 = tqdm(trainloader, total=len(trainloader),leave=False)
    for step, (inputs, labels) in enumerate(tk0, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        outputs= model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        history.append(loss.item())
    
    # validation
    class_correct = list(0. for i in range(1000))
    class_total = list(0. for i in range(1000))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(labels.size()[0]):
                label = labels[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1

    # print statistics
    tqdm.write('[Epoch : %d] train_loss: %.5f val_acc: %.2f' %
        (epoch + 1, epoch_loss / 157, sum(class_correct) / sum(class_total) * 100))
    if min_loss < epoch_loss:
        count+=1
        if count > 10 :
            for g in optimizer.param_groups:
                g['lr']/=10
    else:
        min_loss = epoch_loss
        count = 0

print(time.time()-start_time)
print('Finished Training')