from models.scripts.mnist_cnn import Net
import torch
import os
from torchvision import datasets, transforms
from captum.attr import Saliency, InputXGradient, IntegratedGradients
import matplotlib.pyplot as plt
import numpy as np


def plot_attributions(image, attrs):
    for i, attr in enumerate(attrs):
        plt.figure(i)
        plt.imshow(np.reshape(image.detach().numpy(), [28, 28]), cmap='gray')
        plt.imshow(np.reshape(attr[1].detach().numpy(), [28, 28]), cmap='hot', alpha=0.5)
        plt.colorbar()
        plt.title(attr[0])


def get_decay_curve(image, attr, model, target, n_points=10):
    mask_value = -0.4242
    attr = attr.flatten()
    result = []
    for i in range(n_points):
        max_index = torch.argmax(attr).item()
        image = image.flatten()
        image[max_index] = mask_value
        image = image.reshape(1, 1, 28, 28)
        result.append(model(image)[0, target])
        attr[max_index] = 0
    return result


if __name__ == '__main__':
    model_path = os.path.join(os.path.dirname(__file__), "../models/saved_models/mnist_cnn.pt")

    net = Net()
    net.load_state_dict(torch.load(model_path))
    net.eval()

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=1, shuffle=True)

    image, label = next(iter(train_loader))

    saliency = Saliency(net)
    input_x_gradient = InputXGradient(net)
    integrated_gradients = IntegratedGradients(net)

    attrs = [
        ("Saliency", saliency.attribute(image, target=label)),
        ("InputXGradient", input_x_gradient.attribute(image, target=label)),
        ("IntegratedGradients", integrated_gradients.attribute(image, target=label))
    ]

    plot_attributions(image, attrs)

    n_points = 100
    fig = plt.figure()
    ax = plt.axes()
    x = np.linspace(0, n_points, n_points)
    curves = [(attr[0], get_decay_curve(image, attr[1], net, label, n_points)) for attr in attrs]
    for c in curves:
        ax.plot(x, c[1], label=c[0])
    ax.legend()
