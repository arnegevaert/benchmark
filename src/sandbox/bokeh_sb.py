from bokeh.plotting import figure, show
from lib.plot import plot_images
from vars import DATASET_MODELS
from methods import get_method
import numpy as np
import torch

dataset = DATASET_MODELS["CIFAR10"]["constructor"](batch_size=8)
model = DATASET_MODELS["CIFAR10"]["models"]["resnet20"]()
method = get_method("InputXGradient", model)

iterator = iter(dataset.get_test_data())

images, labels = next(iterator)
attrs = method.attribute(images, target=labels)


def plot_image(img):
    if type(img) == torch.Tensor:
        img = img.detach().numpy()

    plot_images(np.expand_dims(img, 0), 1, 1)

    is_grayscale = img.shape[0] == 1
    # Convert range to 0..255 uint32
    img = (img - np.min(img))/(np.max(img) - np.min(img))
    img *= 255
    img = np.array(img, dtype=np.uint32)

    # If grayscale, just provide rows/cols. If RGBA, convert to RGBA
    if is_grayscale:
        img = np.squeeze(img)
    else:
        img = img.transpose((1, 2, 0))
        rgba_img = np.empty(shape=(img.shape[0], img.shape[1], 4), dtype=np.uint8)
        rgba_img[:, :, :3] = img
        rgba_img[:, :, 3] = 255
        img = rgba_img
    img = np.flip(img, axis=0)
    p = figure(width=200, height=200)
    p.toolbar_location = None
    p.axis.visible = False
    p.grid.visible = False
    plot_fn = p.image if is_grayscale else p.image_rgba
    plot_fn([img], x=0, y=0, dw=1, dh=1)
    show(p)


plot_image(images[0])


"""
img = images[0].detach().numpy()
plot_images(np.expand_dims(img, 0), 1, 1)
img = (img - np.min(img))/(np.max(img) - np.min(img))
img *= 255
img = np.array(img, dtype=np.uint32)
img = img.transpose((1, 2, 0))

rgba_img = np.empty(shape=(img.shape[0], img.shape[1], 4), dtype=np.uint8)
rgba_img[:, :, :3] = img
rgba_img[:, :, 3] = 255
rgba_img = np.flip(rgba_img, axis=0)

p = figure()
p.image_rgba(image=[rgba_img], x=0, y=0, dw=5, dh=5)
show(p)
"""
"""
N = 500
x = np.linspace(0, 10, N)
y = np.linspace(0, 10, N)
xx, yy = np.meshgrid(x, y)
d = np.sin(xx)*np.cos(yy)

p = figure(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])
p.x_range.range_padding = p.y_range.range_padding = 0

# must give a vector of image data for image parameter
p.image(image=[d], x=0, y=0, dw=10, dh=10, palette="Spectral11", level="image")
p.grid.grid_line_width = 0.5

show(p)
"""
