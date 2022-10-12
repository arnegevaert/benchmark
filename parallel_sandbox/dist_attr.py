from captum.attr import Saliency
from util.get_dataset_model import get_model
from attrbench.distributed import AttributionsComputation
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from util.imagenet_subset import ImagenetSubset
from os import path
import h5py
import numpy as np


class LimitedImagenetSubset(Dataset):
    def __init__(self, num_samples=128):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.dataset = ImagenetSubset(path.join("data", "imagenette2", "val"), transform=transform)
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return self.dataset[item]


def method_factory(model):
    method = Saliency(model)
    return method.attribute


if __name__ == "__main__":
    pam = AttributionsComputation(get_model, method_factory, LimitedImagenetSubset(), batch_size=32,
                                  sample_shape=(3, 224, 224), filename="test.h5", method_name="Saliency")
    pam.start()

    with h5py.File("test.h5", "r") as fp:
        ds = fp["Saliency"]
        print(ds.shape)
        print(np.sum(ds))