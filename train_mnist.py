import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm


data_path = '/Users/cesar.salcedo/Documents/datasets/mnist'
epochs = 1

ts = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(0., 1.),
    transforms.Resize(16),
])

data = torchvision.datasets.MNIST(data_path, transform=ts)
data_loader = torch.utils.data.DataLoader(
    data,
    batch_size=128,
    shuffle=True,
    num_workers=0)


class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 10, 3, padding=1),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        y_hat = self.model(x).reshape(-1, 10)
        return y_hat

model = SimpleConvNet()

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

losses = []
for epoch in range(epochs):
    for x, y in tqdm(data_loader):
        y_hat = model(x)
        loss = criterion(y_hat, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

fig = plt.figure()
plt.plot(losses)
plt.title('Training loss')
plt.ylabel('Loss')
plt.xlabel('Timestep')
plt.savefig('training_loss.png')
plt.close(fig)