import torch
import torchvision
from torch import nn
from torch import optim

from datasets import Aptos
import numpy as np
from models.aptos_densenet import Net
from barbar import Bar
import time


def train_model(model, train_dl, val_dl, epochs, loss_function, optimizer, device=None, schedule=None):
    train_loss = []
    val_loss = []

    best_loss = float('Inf')

    for epoch in range(epochs):
        start_time = time.time()
        tl = []
        model.train()
        for idx, (data, targets) in enumerate(Bar(train_dl)):
            optimizer.zero_grad()
            data, targets = data.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            out = model(data)
            loss = loss_function(out, targets)
            loss.backward()
            optimizer.step()
            if schedule is not None:
                schedule.step()
            tl.append(loss.item())
        train_loss.append(np.mean(tl))
        model.eval()
        with torch.no_grad():
            vl = []
            for data, targets in val_dl:
                data, targets = data.cuda(non_blocking=True), targets.cuda(non_blocking=True)
                out = model(data)
                loss = loss_function(out, targets)
                vl.append(loss.item())
        val_loss.append(np.mean(vl))
        print(str(epoch) + " train loss: " + str(train_loss[epoch]) + " val loss: " + str(val_loss[epoch]))
        stop_time = time.time()
        print(f"Elapsed time: {stop_time - start_time:0.4f}")

        if val_loss[epoch] < best_loss:
            print("saving best model")
            best_loss = val_loss[epoch]
            torch.save(model, 'best_model2.pt')

    return train_loss, val_loss


if __name__ == '__main__':

    Aptos_data = Aptos(batch_size=8, img_size=320)
    model = Net("densenet121")

    # model = torchvision.models.densenet121(pretrained=True, progress=True)
    # model.classifier = nn.Sequential(
    #     nn.Linear(1024, 5),
    #     nn.Softmax(dim=1))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    loss = nn.CrossEntropyLoss(weight=Aptos_data.class_weights.to(device))
    optimizer = optim.SGD(model.parameters(), lr=3e-4, momentum=0.9, nesterov=True)
    dl_train = Aptos_data.get_train_data()
    dl_val = Aptos_data.get_test_data()
    epochs = 15
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-3,
                                                   steps_per_epoch=len(dl_train), epochs=epochs, div_factor=25,
                                                   final_div_factor=30)
    train_loss, val_loss = train_model(model, dl_train, dl_val, epochs, loss, optimizer, schedule=schedule)
