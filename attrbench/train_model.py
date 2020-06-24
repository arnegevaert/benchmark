import torch
import torch.nn.functional as F


def train_model(model, dataset, lr, gamma, epochs):
    optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
    loader = dataset.get_dataloader(train=True)
    for epoch in range(1, epochs+1):
        model.train()
        for batch_idx, (samples, target) in enumerate(loader):
            optimizer.zero_grad()
            output = model(samples)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f"Train Epoch: {epoch} ({batch_idx * len(samples)}/{len(loader) * len(samples)})"
                      f"\tLoss: {loss.item():.6f}")
        test_model(model, dataset)
        scheduler.step()


def test_model(model, dataset):
    model.eval()
    test_loss = 0
    correct = 0
    total_samples = 0
    pos_samples = 0
    with torch.no_grad():
        for samples, target in dataset.get_dataloader(train=False):
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
