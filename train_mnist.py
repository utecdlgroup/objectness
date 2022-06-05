import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.dataset import load_dataset
from src.model import SimpleConvNet


data_path = '/Users/cesar.salcedo/Documents/datasets'
epochs = 1
image_size = 16

data = load_dataset(data_path, 'mnist', 16, 1)
data_loader = torch.utils.data.DataLoader(
    data,
    batch_size=128,
    shuffle=True,
    num_workers=0)


model = SimpleConvNet(16, 1, 32, 10)

print(model)

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

torch.save(model.state_dict(), 'model.pt')
