import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from models.scripts.train_mnist_cnn import Net
import numpy as np
from lib.sufficient_input_subsets import sis_collection, produce_masked_inputs

# https://github.com/google-research/google-research/tree/master/sufficient_input_subsets


def get_conf_for_digit(digit, imgs):
    res = net(imgs)
    return res[:, digit].detach()


# Helpers for plotting an MNIST digit and its corresponding SIS-collection.
def plot_mnist_digit(ax, image):
    ax.imshow(image.reshape((28, 28)), cmap=plt.get_cmap('gray'))
    ax.axis('off')


def plot_sis_collection(initial_image, collection, fully_masked_image):
    # Grid contains initial image, an empty cell (for spacing), and collection.
    width = len(collection) + 2
    plt.figure(figsize=(width, 1))
    gs = plt.GridSpec(1, width, wspace=0.1)

    # Plot initial image.
    ax = plt.subplot(gs[0])
    plot_mnist_digit(ax, initial_image)

    # Plot each SIS.
    for i, sis_result in enumerate(collection):
        ax = plt.subplot(gs[i+2])
        masked_image = produce_masked_inputs(
            initial_image, fully_masked_image, [sis_result.mask])[0]
        plot_mnist_digit(ax, masked_image)

    plt.show()


if __name__ == '__main__':
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True)

    """
    X_train = []
    y_train = []
    for (images, labels) in iter(train_loader):
        X_train += images
        y_train += labels
    X_train = np.vstack(X_train)
    y_train = np.array(y_train)
    """

    model_path = "../../models/saved_models/mnist_cnn.pth"
    net = Net()
    net.load_state_dict(torch.load(model_path))
    net.eval()

    images, labels = next(iter(train_loader))
    image = images[0]
    label = labels[0]
    threshold = 0.7

    #fully_masked_image = np.full((1, 28, 28), np.mean(X_train).item())
    fully_masked_image = np.zeros((1, 28, 28))
    fig, ax = plt.subplots()
    #plot_mnist_digit(ax, fully_masked_image[0])
    print("Getting collection...")
    collection = sis_collection(lambda img: get_conf_for_digit(label, torch.tensor(img)), threshold, image, fully_masked_image)
    print("Got collection")
    plot_sis_collection(image, collection, fully_masked_image)
