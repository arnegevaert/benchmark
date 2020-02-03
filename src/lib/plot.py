import matplotlib.pyplot as plt
import numpy as np


def plot_attributions(image, attrs):
    for i, attr in enumerate(attrs):
        plt.figure(i)
        plt.imshow(np.reshape(image.detach().numpy(), [28, 28]), cmap='gray')
        plt.imshow(np.reshape(attr[1].detach().numpy(), [28, 28]), cmap='hot', alpha=0.5)
        plt.colorbar()
        plt.title(attr[0])


def plot_mnist_digit(image):
    fig, ax = plt.subplots()
    ax.imshow(image.reshape((28, 28)), cmap=plt.get_cmap('gray'))
    ax.axis('off')
