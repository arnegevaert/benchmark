import torch
import torch.nn.functional as F
from .masked_dataset import MaskedDataset
from .masked_neural_network import MaskedNeuralNetwork


def train_masked_network(model: MaskedNeuralNetwork, data: MaskedDataset, lr, gamma, epochs):
    optimizer = torch.optim.Adadelta(model.net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(1, epochs+1):
        model.train()
        for batch_idx, (samples, target) in enumerate(data.get_train_data()):
            optimizer.zero_grad()
            output = model(samples)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f"Train Epoch: {epoch} ({batch_idx * len(samples)})\tLoss: {loss.item():.6f}")
        test_masked_network(model, masked_dataset=data)
        scheduler.step()


def test_masked_network(model: MaskedNeuralNetwork, masked_dataset: MaskedDataset):
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
          f" Accuracy: {correct}/{total_samples} ({100*correct/total_samples:.0f}%)")
    print(f"Positive samples: {pos_samples}/{total_samples} ({100*pos_samples/total_samples:.0f}%)\n")
