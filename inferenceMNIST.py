import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision
import matplotlib.pyplot as plt

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

def predict(model, x):
    with torch.no_grad():
        out = model(x)
        pred = out.data.max(1, keepdim=True)[1]
        return pred

if __name__ == '__main__':
    model = Net()
    model.load_state_dict(torch.load('results/model_mnist.pth'))
    model.eval()
    test_dataset = torchvision.datasets.MNIST('files/', train=False, download=False,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
                ]))
    
    corrects = 0
    for i in range(1, 20):
        plt.subplot(4,5,i)
        plt.tight_layout()
        data, target = test_dataset[i]
        pred = predict(model, data)
        data = data.squeeze()
        plt.imshow(data, cmap='gray')
        plt.title(f'{pred.item(), target}')
    plt.show()
    