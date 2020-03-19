from vars import DATASET_MODELS
from methods import get_method
from lib.quickplot import *

dataset = DATASET_MODELS["CIFAR10"]["constructor"](batch_size=8)
model = DATASET_MODELS["CIFAR10"]["models"]["resnet20"]()
method = get_method("InputXGradient", model)

iterator = iter(dataset.get_test_data())

images, labels = next(iterator)
attrs = method.attribute(images, target=labels)
# plot_images(images, 4, 2)
plot_attributions(images, attrs, 4, 2, absolute=False)
