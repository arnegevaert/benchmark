import matplotlib.pyplot as plt
import numpy as np


def plot_attributions(image, attr):
    plt.figure()
    plt.imshow(np.reshape(image.detach().numpy(), [28, 28]), cmap='gray')
    plt.imshow(np.reshape(attr.detach().numpy(), [28, 28]), cmap='hot', alpha=0.5)
    plt.colorbar()


def plot_mnist_digit(image):
    fig, ax = plt.subplots()
    ax.imshow(image.reshape((28, 28)), cmap=plt.get_cmap('gray'))
    ax.axis('off')
