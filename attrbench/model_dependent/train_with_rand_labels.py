from typing import Iterable, Callable
import numpy
import torch
import torch.nn as nn
import torch.optim as optim


def _train_step(criterium, device, model, optimizer, rng_label_data,scheduler):
    train_loss = 0.
    model.train()
    for batch in rng_label_data:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterium(out, y)
        loss.backward()
        optimizer.step()
        if scheduler is not None: scheduler.step()
        train_loss += loss.item()
    train_loss = train_loss / len(rng_label_data)
    return train_loss


def _test_step(device, model, rng_label_data):
    model.eval()
    correct = 0.
    total = 0.
    with torch.no_grad():
        for batch in rng_label_data:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            _, predicted = torch.max(out.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total


def train_with_random_labels(model: Callable, rng_label_data: Iterable, device='cuda',
                             epochs = 20, lr=1e-3, use_schedule = True):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterium = nn.CrossEntropyLoss()
    if use_schedule:
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs = epochs, steps_per_epoch = len(rng_label_data))
    else:
        scheduler = None
    for e in range(epochs):
        loss = _train_step(criterium, device, model, optimizer, rng_label_data, scheduler)
        acc = _test_step(device, model, rng_label_data)
        print('epoch {}: loss: {} accuracy: {}'.format(e, loss, acc))


    return model
