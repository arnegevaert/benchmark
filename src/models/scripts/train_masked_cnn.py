from vars import DATASET_MODELS
import torch
import torch.nn.functional as F
import torch.optim as optim
from models import MaskedCNN

def get_medians(data):
    maxs = []
    mins = []
    for samples, labels in iter(data):
        samples = model(samples)
        samples = samples.reshape(BATCH_SIZE, -1)
        maxs.append(torch.max(samples, dim=1)[0])
        mins.append(torch.min(samples, dim=1)[0])
    maxs = torch.cat(maxs)
    mins = torch.cat(mins)
    print(f"Median of maxima: {torch.median(maxs)}")
    print(f"Median of minima: {torch.median(mins)}")


def train(model, train_loader, optimizer, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        target = torch.all(model.mask(data).reshape(BATCH_SIZE, -1) > MEDIAN_VALUE, dim=1).long()
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            target = torch.all(model.mask(data).reshape(BATCH_SIZE, -1) > MEDIAN_VALUE, dim=1).long()
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    DATASET = "CIFAR10"
    BATCH_SIZE = 64
    EPOCHS = 10
    LR = 1.0
    GAMMA = 0.7
    MEDIAN_VALUE = -.788235

    dataset_constructor = DATASET_MODELS[DATASET]["constructor"]
    dataset = dataset_constructor(batch_size=BATCH_SIZE, shuffle=True, download=False)

    model = MaskedCNN(dataset.sample_shape, mask_radius=5, mask_value=dataset.mask_value)
    optimizer = optim.Adadelta(model.net.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=GAMMA)
    for epoch in range(1, EPOCHS+1):
        train(model.net, train_loader=dataset.get_train_data(), optimizer=optimizer, epoch=epoch, log_interval=100)
        test(model.net, test_loader=dataset.get_test_data())
        scheduler.step()
