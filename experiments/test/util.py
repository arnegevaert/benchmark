import numpy as np
import torchvision
from matplotlib import colors
import matplotlib.pyplot as plt


def imshow(img, title):
    npimg = torchvision.utils.make_grid(img).cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()

def attrshow(attrs, title):
    npattrs = attrs.squeeze().detach().cpu().numpy()  # [batch_size, rows, cols]
    npattrs = np.concatenate(npattrs, axis=-1)
    min_value = min(np.min(npattrs), -.01)
    max_value = max(np.max(npattrs), .01)
    divnorm = colors.TwoSlopeNorm(vmin=min_value, vcenter=0., vmax=max_value)
    plt.imshow(npattrs, cmap="bwr", norm=divnorm)
    plt.colorbar(orientation="horizontal")
    plt.title(title)
    plt.show()
    
def show_img_attrs(imgs, attrs):
    imshow(imgs, "Images")
    attrshow(attrs, "Attributions")