import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_images(images, nrows, ncols):
    # images is a batch of images:
    # [batch_size, 3, rows, cols] (if color image)
    # [batch_size, 1, rows, cols] (if grayscale image)
    fig = plt.figure()
    if type(images) == torch.Tensor:
        images = images.detach().numpy()
    is_grayscale = images.shape[1] == 1
    for i in range(images.shape[0]):
        fig.add_subplot(nrows, ncols, i+1)
        image = np.transpose(images[i], (1, 2, 0))  # imshow expects [rows, cols, 3]
        if is_grayscale:
            # Image is grayscale, imshow expects dimensions to be squeezed
            image = np.squeeze(image)
        image = (image - np.min(image))/(np.max(image) - np.min(image))
        plt.imshow(image, cmap="gray")  # cmap is ignored for color images
