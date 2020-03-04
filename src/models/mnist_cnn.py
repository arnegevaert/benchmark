import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from os import path, getenv
from torch.optim.lr_scheduler import StepLR
from models.model import Model
from datasets.mnist import MNIST

DEVICE = torch.device(getenv("TORCH_DEVICE", "cpu"))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def get_logits(self, x):
        relu = nn.ReLU()
        x = self.conv1(x)
        x = relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = relu(x)
        x = self.dropout2(x)
        return self.fc2(x)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        logits = self.get_logits(x)
        return F.softmax(logits, dim=1)


# TODO dataset shouldn't be required on construction
class MNISTCNN(Model):
    def __init__(self, load=True, location=None, dataset=None):
        super().__init__()
        self.net = Net().to(DEVICE)
        if load:
            params_loc = location if location else path.join(path.dirname(__file__), "saved_models", "mnist_cnn.pt")
            # map_location allows taking a model trained on GPU and loading it on CPU
            # without it, a model trained on GPU will be loaded in GPU even if DEVICE is CPU
            self.net.load_state_dict(torch.load(params_loc, map_location=lambda storage, loc: storage))
        self.dataset = dataset if dataset else MNIST(batch_size=64, download=False)
        self.optimizer = optim.Adadelta(self.net.parameters(), lr=1.0)

    def train(self, epochs, save=True):
        scheduler = StepLR(self.optimizer, step_size=1, gamma=0.7)
        train_loader = self.dataset.get_train_data()
        test_loader = self.dataset.get_test_data()
        for epoch in range(epochs):
            # Train for 1 epoch
            self.net.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(DEVICE), target.to(DEVICE)
                self.optimizer.zero_grad()
                output = self.net(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                self.optimizer.step()
                if batch_idx % 100 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))

            # Test
            self.net.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    output = self.net(data)
                    test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
            scheduler.step()

        if save:
            torch.save(self.net.state_dict(), path.join(path.dirname(__file__), "saved_models", "mnist_cnn.pt"))

    def predict(self, x):
        self.net.eval()
        return self.net(x)
