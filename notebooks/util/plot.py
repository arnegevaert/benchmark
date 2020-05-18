import matplotlib.pyplot as plt
import numpy as np


# Expects attributions shape [rows, columns]
def plot_image_with_attributions(image, attributions):
    image = np.squeeze(image)
    attributions = np.squeeze(attributions)
    is_color = len(image.shape) > 2
    # if color image, shape must be (rows, columns, channels)
    if is_color and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    # Normalize image and attribution values to [0,1]
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    attributions = (attributions - np.min(attributions)) / (np.max(attributions) - np.min(attributions))

    fig = plt.figure(figsize=(8, 8))
    fig.add_subplot(1, 2, 1)
    plt.imshow(image, cmap="gray" if not is_color else None)
    fig.add_subplot(1, 2, 2)
    plt.imshow(attributions, cmap="gray")
