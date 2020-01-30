import torch
from torchvision import datasets, transforms
from captum.attr import Saliency, InputXGradient, IntegratedGradients
from models.scripts.mnist_cnn import Net
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == '__main__':
    model_path = os.path.join(os.path.dirname(__file__), "../models/saved_models/mnist_cnn.pt")

    net = Net()
    net.load_state_dict(torch.load(model_path))
    net.eval()

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=1, shuffle=True)

    image, label = next(iter(train_loader))
    image.requires_grad = True

    saliency = Saliency(net)
    input_x_gradient = InputXGradient(net)
    integrated_gradients = IntegratedGradients(net)

    attrs = [
        ("Saliency", torch.pow(saliency.attribute(image, target=label), 2)),
        ("InputXGradient", input_x_gradient.attribute(image, target=label)),
        ("IntegratedGradients", integrated_gradients.attribute(image, target=label))
    ]

    for i, attr in enumerate(attrs):
        plt.figure(i)
        print(i)
        plt.imshow(np.reshape(image.detach().numpy(), [28, 28]), cmap='gray')
        plt.imshow(np.reshape(attr[1].detach().numpy(), [28, 28]), cmap='hot', alpha=0.5)
        plt.colorbar()
        plt.title(attr[0])
