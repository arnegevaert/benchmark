from models import MaskedInputLayer
from vars import DATASET_MODELS
from lib import plot_images


DATASET = "CIFAR10"
BATCH_SIZE = 4

dataset_constructor = DATASET_MODELS[DATASET]["constructor"]
dataset = dataset_constructor(batch_size=BATCH_SIZE, shuffle=False, download=False)

model = MaskedInputLayer(dataset.sample_shape, radius=5, mask_value=dataset.mask_value)
samples, labels = next(iter(dataset.get_test_data()))
out = model(samples)
plot_images(out, 2, 2)
