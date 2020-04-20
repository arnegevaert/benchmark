import torch
import torch.nn.functional as F
from benchmark.masking_accuracy import MaskedDataset
from benchmark.masking_accuracy import MaskedNeuralNetwork


def train_masked_network(model: MaskedNeuralNetwork, data: MaskedDataset, lr, gamma, epochs):
    optimizer = torch.optim.Adadelta(model.net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(1, epochs+1):
        _train_epoch(model, masked_dataset=data, optimizer=optimizer, epoch=epoch, log_interval=100)
        _test_epoch(model, masked_dataset=data)
        scheduler.step()


def _train_epoch(model: MaskedNeuralNetwork, masked_dataset: MaskedDataset, optimizer, log_interval, epoch):
    model.train()
    for batch_idx, (samples, target) in enumerate(masked_dataset.get_train_data()):
        optimizer.zero_grad()
        output = model(samples)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} ({batch_idx * len(samples)})\tLoss: {loss.item():.6f}")


def _test_epoch(model: MaskedNeuralNetwork, masked_dataset: MaskedDataset):
    model.eval()
    test_loss = 0
    correct = 0
    total_samples = 0
    pos_samples = 0
    with torch.no_grad():
        for samples, target in masked_dataset.get_test_data():
            # target = torch.all(model.mask(data).reshape(BATCH_SIZE, -1) > MEDIAN_VALUE, dim=1).long()
            output = model(samples)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += len(samples)
            pos_samples += target.sum()
    test_loss /= total_samples
    print(f"\nTest set: Average loss: {test_loss:.4f},"
          f" Accuracy: {correct}/{total_samples} ({correct/total_samples:.0f}%)")
    print(f"Positive samples: {pos_samples}/{total_samples} ({pos_samples/total_samples:.0f}%)\n")
