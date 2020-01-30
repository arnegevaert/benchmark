import torch
from torchvision import datasets, transforms
from captum.attr import Saliency, InputXGradient, IntegratedGradients
from os import path
from models.scripts.mnist_cnn import Net
import numpy as np
import matplotlib.pyplot as plt


def generate_perturbations(images, noise_variance, amount):
    tiled_imgs = np.tile(images, (amount, 1, 1))
    noise = (np.random.rand(*tiled_imgs.shape) - 0.5) * noise_variance
    print(np.mean(noise))
    return tiled_imgs + noise


def plot_mnist_digit(image):
    fig, ax = plt.subplots()
    ax.imshow(image.reshape((28, 28)), cmap=plt.get_cmap('gray'))
    ax.axis('off')


if __name__ == '__main__':
    model_path = path.join(path.dirname(__file__), "../models/saved_models/mnist_cnn.pt")
    net = Net()
    net.load_state_dict(torch.load(model_path))
    net.eval()

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=16, shuffle=True)

    X_batch, y_batch = next(iter(train_loader))
    perturbations = generate_perturbations(X_batch, 1, 8)


    saliency = Saliency(net)
    inputxgradient = InputXGradient(net)
    integrated_gradients = IntegratedGradients(net)

    methods = {
        "Saliency": lambda img, lbl: torch.pow(saliency.attribute(img, target=lbl), 2),
        "InputXGradient": lambda img, lbl: inputxgradient.attribute(img, target=lbl),
        "IntegratedGradients": lambda img, lbl: integrated_gradients.attribute(img, target=lbl)
    }

    for name, method in methods.items():
        pass

    """
    attrs = [
        ("Saliency", lambda img: torch.pow(saliency.attribute(img, target=label), 2)),
        ("InputXGradient", lambda img: input_x_gradient.attribute(img, target=label)),
        ("IntegratedGradients", lambda img: integrated_gradients.attribute(img, target=label))
    ]
    """


