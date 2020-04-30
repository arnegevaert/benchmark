import torch
from torch import nn
from torch import optim

from util.datasets import Aptos
import numpy as np
from util.models import AptosDensenet
from tqdm import tqdm
import time


class ModelCheckpointCb:
    def __init__(self) -> None:
        super().__init__()
        self.best_loss = float('Inf')

    def step(self, loss, model):
        if loss < self.best_loss:
            print("saving best model")
            self.best_loss = loss
            torch.save(model.state_dict(), 'best_model_checkpoint.pt')


def train_model(model, dataset, epochs, loss_function, optimizer, schedule=None, cb=None):
    train_loss = []
    val_loss = []

    for epoch in range(epochs):
        start_time = time.time()
        tl = []
        model.train()
        for idx, (data, targets) in enumerate(tqdm(dataset.get_train_data())):
            optimizer.zero_grad()
            data, targets = torch.tensor(data, dtype=torch.float).cuda(non_blocking=True), \
                            torch.tensor(targets, dtype=torch.long).cuda(non_blocking=True)
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
            for data, targets in dataset.get_test_data():
                data, targets = torch.tensor(data, dtype=torch.float).cuda(non_blocking=True), \
                                torch.tensor(targets, dtype=torch.long).cuda(non_blocking=True)
                out = model(data)
                loss = loss_function(out, targets)
                vl.append(loss.item())
        val_loss.append(np.mean(vl))
        print(str(epoch) + " train loss: " + str(train_loss[epoch]) + " val loss: " + str(val_loss[epoch]))
        stop_time = time.time()
        print(f"Elapsed time: {stop_time - start_time:0.4f}")
        if cb is not None: cb.step(val_loss[epoch],model)
    return train_loss, val_loss


if __name__ == '__main__':
    aptos_data = Aptos(batch_size=8, img_size=320, data_location="../../../data/Aptos")

    model = AptosDensenet(output_logits=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    loss = nn.CrossEntropyLoss(weight=torch.tensor(aptos_data.class_weights, dtype=torch.float).to(device))
    optimizer = optim.SGD(model.parameters(), lr=3e-4, momentum=0.9, nesterov=True, weight_decay=1e-4)
    cb = ModelCheckpointCb()
    epochs = 15
    for cycle in range(2):
        schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-3, epochs=epochs, div_factor=25,
                                                       steps_per_epoch=aptos_data.x_train.shape[0], final_div_factor=30)
        train_loss, val_loss = train_model(model, aptos_data, epochs, loss, optimizer, schedule=schedule, cb=cb)
