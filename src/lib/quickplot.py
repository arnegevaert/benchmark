import matplotlib.pyplot as plt
import numpy as np
import torch


def _plot_image(fig, image, nrows, ncols, index):
    fig.add_subplot(nrows, ncols, index)
    image = np.transpose(image, (1, 2, 0))  # imshow expects [rows, cols, 3]
    # If image is grayscale, imshow expects dimensions to be squeezed (if not, squeeze has no effect)
    image = np.squeeze(image)
    image = (image - np.min(image))/(np.max(image) - np.min(image))
    plt.imshow(image, cmap="gray")  # cmap is ignored for color images


def _overlay_attributions(fig, attr, nrows, ncols, index, absolute):
    fig.add_subplot(nrows, ncols, index)
    attr = np.transpose(attr, (1, 2, 0))  # imshow expects [rows, cols, 3]
    # If image is grayscale, imshow expects dimensions to be squeezed (if not, squeeze has no effect)
    attr = np.squeeze(attr)
    # if absolute is True, heatmap should we white-red. Otherwise, blue-white-red (negative attributions are blue)
    cmap = "Reds" if absolute else "bwr"
    # Use max absolute value of vmin/vmax s.t. 0 is always centered
    max_abs_val = np.max(np.abs(attr))
    vmin = 0 if absolute else -max_abs_val
    plt.imshow(attr, cmap=cmap, vmin=vmin, vmax=max_abs_val, alpha=0.5)  # cmap is ignored for color images
    plt.colorbar()


def plot_images(images, nrows, ncols):
    # images is a batch of images:
    # [batch_size, 3, rows, cols] (if color image)
    # [batch_size, 1, rows, cols] (if grayscale image)
    fig = plt.figure()
    if type(images) == torch.Tensor:
        images = images.detach().numpy()
    for i in range(images.shape[0]):
        _plot_image(fig, images[i], nrows, ncols, i+1)


def plot_attributions(images, attributions, nrows, ncols, absolute=True):
    # images and attributions have identical shapes
    # images is a batch of images:
    # [batch_size, 3, rows, cols] (if color image)
    # [batch_size, 1, rows, cols] (if grayscale image)
    if type(images) == torch.Tensor:
        images = images.detach().numpy()
    if type(attributions) == torch.Tensor:
        attributions = attributions.detach().numpy()
    if images.shape[1] == 1:
        # Overlay attributions on top of images
        fig = plt.figure()
        for i in range(images.shape[0]):
            _plot_image(fig, images[i], nrows, ncols, i+1)
            _overlay_attributions(fig, attributions[i], nrows, ncols, i+1, absolute)
    else:
        # Separate plot for images and attributions
        fig_imgs = plt.figure()
        for i in range(images.shape[0]):
            _plot_image(fig_imgs, images[i], nrows, ncols, i+1)
        fig_attrs = plt.figure()
        for i in range(images.shape[0]):
            _plot_image(fig_attrs, attributions[i], nrows, ncols, i+1)
