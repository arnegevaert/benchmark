from vars import DATASET_MODELS
import numpy as np
import torch
import torch.nn.functional as F
from random import randint
DEVICE = "cuda"
DATASET = "CIFAR10"
DOWNLOAD_DATASET = False
# patch values will be clipped between DATA_MIN and DATA_MAX so that patch will be valid image data.
DATA_MIN=-1
DATA_MAX =1
MODEL = "resnet20"
#targeted class
TARGET_LABEL = 0

BATCH_SIZE = 64
EPOCHS = 20
# size of patch wrt original images.
PATCH_SIZE_PERCENT = 0.1
# does it matter if we train the patch on the test set if we are only interested in attacking the test set?
TRAIN_ON_TEST = False

class PatchCheckpointCb():

    def __init__(self) -> None:
        super().__init__()
        self.best_loss = float('Inf')

    def step(self, loss, patch):
        if loss < self.best_loss:
            print("saving best patch")
            self.best_loss = loss
            torch.save(patch, 'patch_checkpoint.pt')


def norm(x, x_min, x_max):
    return x * (x_max - x_min) + x_min


def init_patch_square(image_size, patch_size_precent):
    # get mask
    image_size = image_size ** 2
    noise_size = image_size * patch_size_precent
    noise_dim = int(noise_size ** (0.5))
    patch = np.random.rand(1, 3, noise_dim, noise_dim)
    patch = norm(patch, DATA_MIN, DATA_MAX)
    return patch, patch.shape


def train_patch(model, patch, train_dl, loss_function, optimizer, device=None, schedule=None, cb=None):
    patch_size = patch.shape[-1]
    train_loss = []
    nr_not_successfull = []
    for x, y in train_dl:
        optimizer.zero_grad()
        y = torch.tensor(np.full(y.shape[0], TARGET_LABEL), dtype=torch.long).to(DEVICE)
        image_size = x.shape[-1]
        mask = np.zeros((image_size, image_size))
        ind = randint(0, image_size - patch_size)
        # ind=0
        mask[ind:ind + patch_size, ind:ind + patch_size] = 1.
        mask = torch.tensor(mask).to(DEVICE)

        adv_image = torch.zeros((3, image_size, image_size)).to(DEVICE)
        adv_image[:, ind:ind + patch_size, ind:ind + patch_size] = patch
        images = x.to(DEVICE)

        input = torch.mul((1 - mask), images) + adv_image
        adv_out = model.predict(input)

        loss = loss_function(adv_out, y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            patch.data = torch.clamp(patch.data, min=DATA_MIN, max=DATA_MAX)
        train_loss.append(loss.item())
        nr_not_successfull.append(adv_out.argmax(axis=1).detach().cpu().numpy())
    epoch_loss = np.array(train_loss).mean()
    nr_not_successfull = np.count_nonzero(np.array(nr_not_successfull) != TARGET_LABEL)

    return epoch_loss, nr_not_successfull
def validate(model, patch, val_data, loss_function):
    patch_size = patch.shape[-1]
    val_loss = []
    nr_not_successfull = []
    with torch.no_grad():
        for x, y in val_data:
            y = torch.tensor(np.full(y.shape[0], TARGET_LABEL), dtype=torch.long).to(DEVICE)
            image_size = x.shape[-1]
            mask = np.zeros((image_size, image_size))
            ind = randint(0, image_size - patch_size)
            mask[ind:ind + patch_size, ind:ind + patch_size] = 1.
            mask = torch.tensor(mask).to(DEVICE)
            adv_image = torch.zeros((3, image_size, image_size)).to(DEVICE)
            adv_image[:, ind:ind + patch_size, ind:ind + patch_size] = patch
            images = x.to(DEVICE)
            input = torch.mul((1 - mask), images) + adv_image
            adv_out = model.predict(input)
            loss = loss_function(adv_out, y)

            val_loss.append(loss.item())
            nr_not_successfull.append(adv_out.argmax(axis=1).detach().cpu().numpy())
        val_loss = np.array(val_loss).mean()
        nr_not_successfull = np.count_nonzero(np.array(nr_not_successfull) != TARGET_LABEL)
        return val_loss, nr_not_successfull


if __name__ == '__main__':
    dataset_constructor = DATASET_MODELS[DATASET]["constructor"]
    model_constructor = DATASET_MODELS[DATASET]["models"][MODEL]

    model = model_constructor(output_logits=True)
    model.to(DEVICE)
    for param in model.net.parameters():
        param.requires_grad = False
    model.net.eval()

    dataset = dataset_constructor(batch_size=BATCH_SIZE, shuffle=True, download=DOWNLOAD_DATASET)
    dl = iter(dataset.get_test_data())
    x, _ = dl.next()
    image_size =x.shape[-1]

    patch, shape = init_patch_square(image_size, PATCH_SIZE_PERCENT)
    patch = torch.tensor(patch, requires_grad=True, device=DEVICE)
    # optim = torch.optim.Adam([patch], lr=0.05, weight_decay=0.)
    optim = torch.optim.SGD([patch], lr =0.05, momentum=0.9, weight_decay=0., nesterov=True)

    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=2, verbose=True)
    cb = PatchCheckpointCb()
    loss = torch.nn.CrossEntropyLoss()

    if TRAIN_ON_TEST:
        train_data = dataset.get_test_data()
    else:
        train_data = dataset.get_train_data()

    for e in range(EPOCHS):

        epoch_loss, nr_not_successfull = train_patch(model, patch, train_data, loss, optim, DEVICE, schedule, cb)
        # print(" epoch {} done. train_loss: {}".format(e, epoch_loss))
        # print("{} images not successfully attacked ".format(nr_not_successfull))
        val_loss, nr_not_successfull = validate(model, patch, dataset.get_test_data(),loss)
        print(" epoch {} done. val_loss: {}".format(e, val_loss))
        print("{} images not successfully attacked ".format(nr_not_successfull))

        if schedule is not None:
            schedule.step(val_loss)
        if cb is not None: cb.step(val_loss, patch)
        