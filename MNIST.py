import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('files/', train=True, download=True,
                            transform=torchvision.transforms.Compose(
                                [
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.1307,), (0.3081,))
                            ])),
batch_size=24, 
shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('files/', train=False, download=True,
                            transform=torchvision.transforms.Compose(
                            [
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.1307,), (0.3081,))
                            ])),
batch_size=24, 
shuffle=True
)

class Net(nn.Module):

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.layers = nn.Sequential(
           nn.Flatten(),
           nn.Linear(28*28, 80),
           nn.Flatten(),
           nn.Linear(80, 16),
           nn.Flatten(),
           nn.Linear(16, 10)
        )
    
    def forward(self, x):
      return self.layers(x)        

# Hiper-Parâmetros
n_epochs = 5
network = Net()
optimizer = optim.SGD(network.parameters(), lr=0.005, momentum=0.5)
criterion = nn.CrossEntropyLoss()
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = network(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % 10 == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(network.state_dict(), 'results/model_mnist.pth')

def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      test_loss += criterion(output, target)
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
  
if __name__ == '__main__':

    test()
    for epoch in range(1, n_epochs):
       train(epoch)
    test()

    # plt.plot(train_losses, [x for x in range(1, n_epochs + 1)])
    # plt.plot(train_losses, [x for x in range(1, n_epochs + 1)])
    
