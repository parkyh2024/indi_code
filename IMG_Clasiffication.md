온전한 코드를 보려면 edit으로 들어가서 봐야 함


# 모두의 딥러닝 10-4-1
import torchvision
from torchvision import transforms

from torch.utils.data import DataLoader

#학습시킬 이미지의 경로를 입력. 이 경로에는 0과 1 이름의 폴더가 두 개 있다.
train_data = torchvision.datasets.ImageFolder(root='C:/Users/Park Yu Hyun/Desktop/dataset/Origin_data', transform=None)

for num, value in enumerate(train_data):
    data, label = value
    print(num, data, label)
print('학습완료!')

# 모두의 딥러닝 10-4-2
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

trans = transforms.Compose([
    transforms.Resize((64, 128)),  # Add this line
    transforms.ToTensor()
])

train_data = torchvision.datasets.ImageFolder(root='C:/Users/Park Yu Hyun/Desktop/dataset/Train_data', transform=trans)

data_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True, num_workers=0)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(16 * 13 * 29, 3),
            nn.ReLU(),
            nn.Linear(3, 2)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.shape[0], -1)
        out = self.layer3(out)
        return out

net = CNN().to(device)
test_input = (torch.Tensor(3, 3, 64, 128)).to(device)
test_out = net(test_input)

optimizer = optim.Adam(net.parameters(), lr=0.0007)
loss_func = nn.CrossEntropyLoss().to(device)

total_batch = len(data_loader)

epochs = 30  # 20
for epoch in range(epochs):
    avg_cost = 0.0
    for num, data in enumerate(data_loader):
        imgs, labels = data
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        out = net(imgs)
        loss = loss_func(out, labels)
        loss.backward()
        optimizer.step()

        avg_cost += loss / total_batch

    print('[Epoch:{}] cost = {}'.format(epoch+1, avg_cost))
print('Learning Finished!')
