import numpy as np
import torch
from tqdm import trange
from random import randint


class PatchCheckpointCb:
    def __init__(self, path) -> None:
        super().__init__()
        self.save_path = path
        self.best_loss = float('Inf')

    def step(self, loss, patch):
        if loss < self.best_loss:
            self.best_loss = loss
            torch.save(patch, self.save_path)


def norm(x, x_min, x_max):
    return x * (x_max - x_min) + x_min


def init_patch_square(image_size, image_channels, patch_size_percent, data_min, data_max):
    # get mask
    image_size = image_size ** 2
    noise_size = image_size * patch_size_percent
    noise_dim = int(noise_size ** 0.5)
    patch = np.random.rand(1, image_channels, noise_dim, noise_dim)
    patch = norm(patch, data_min, data_max)
    return patch, patch.shape


def train_patch(model, patch, train_dl, loss_function, optimizer, target_label, data_min, data_max, device,
                schedule=None, cb=None):
    patch_size = patch.shape[-1]
    train_loss = []
    nr_not_successfull = []
    for x, y in train_dl:
        # x, y = torch.tensor(x), torch.tensor(y)
        optimizer.zero_grad()
        y = torch.tensor(np.full(y.shape[0], target_label), dtype=torch.long).to(device)
        image_size = x.shape[-1]

        indx = randint(0, image_size - patch_size)
        indy = randint(0, image_size - patch_size)
        # indx = image_size // 2 - patch_size // 2
        # indy = indx

        images = x.to(device)
        images[:, :, indx:indx + patch_size, indy:indy + patch_size] = patch
        adv_out = model(images)

        loss = loss_function(adv_out, y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            patch.data = torch.clamp(patch.data, min=data_min, max=data_max)
        train_loss.append(loss.item())
    epoch_loss = np.array(train_loss).mean()

    return epoch_loss


def validate(model, patch, data_loader, loss_function, target_label, device):
    patch_size = patch.shape[-1]
    val_loss = []
    preds = []
    with torch.no_grad():
        for x, y in data_loader:
            y = torch.tensor(np.full(y.shape[0], target_label), dtype=torch.long).to(device)
            image_size = x.shape[-1]

            indx = randint(0, image_size - patch_size)
            indy = randint(0, image_size - patch_size)

            images = x.to(device)
            images[:, :, indx:indx + patch_size, indy:indy + patch_size] = patch
            adv_out = model(images)
            loss = loss_function(adv_out, y)

            val_loss.append(loss.item())
            preds.append(adv_out.argmax(axis=1).detach().cpu().numpy())
        val_loss = np.array(val_loss).mean()
        preds = np.concatenate(preds)
        percent_successfull = np.count_nonzero(preds == target_label) / preds.shape[0]
        return val_loss, percent_successfull


def make_patch(dataloader, model, target_label, patch_path, device, patch_percent=0.1, epochs=20,
               data_min=None, data_max=None, lr=0.005):
    # patch values will be clipped between data_min and data_max so that patch will be valid image data.
    if data_max is None or data_min is None:
        for x, _ in dataloader:
            if data_max is None:
                data_max = x.max().item()
            if data_min is None:
                data_min = x.min().item()
            if x.min() < data_min:
                data_min = x.min().item()
            if x.max() > data_max:
                data_max = x.max().item()

    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    x, _ = next(iter(dataloader))
    sample_shape = x.shape

    patch, shape = init_patch_square(sample_shape[-1], sample_shape[1], patch_percent, data_min, data_max)
    patch = torch.tensor(patch, requires_grad=True, device=device)
    optim = torch.optim.Adam([patch], lr=lr, weight_decay=0.)

    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=2, verbose=True)
    cb = PatchCheckpointCb(patch_path)
    loss = torch.nn.CrossEntropyLoss()

    prog = trange(epochs)
    for e in prog:
        epoch_loss = train_patch(model, patch, dataloader, loss, optim, target_label=target_label, data_min=data_min,
                                 data_max=data_max, device=device,
                                 schedule=schedule, cb=cb)
        val_loss, percent_successfull = validate(model, patch, dataloader, loss, target_label, device)
        prog.set_postfix({"train_loss": epoch_loss, "val_loss": val_loss, "success_rate": percent_successfull * 100})

        if schedule is not None:
            schedule.step(val_loss)
        if cb is not None: cb.step(val_loss, patch)
