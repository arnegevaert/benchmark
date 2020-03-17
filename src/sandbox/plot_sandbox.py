from vars import DATASET_MODELS
from lib.plot import *

dataset = DATASET_MODELS["CIFAR10"]["constructor"](batch_size=8)
iterator = iter(dataset.get_test_data())

images, labels = next(iterator)
plot_images(images, 4, 2)
