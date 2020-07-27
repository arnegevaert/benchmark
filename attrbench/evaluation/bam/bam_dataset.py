from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from os import path


class BAMDataset:
    def __init__(self, batch_size, data_location, shuffle=True):
        self.batch_size = batch_size
        self.data_location = data_location
        self.shuffle = shuffle
        self.datasets = {
            ds_name: {
                version: ImageFolder(path.join(self.data_location, ds_name, version))
                for version in ["train", "val"]
            } for ds_name in ["obj", "scene", "scene_only"]}

    """
    ds_name: - "obj": object labels
             - "scene": scene labels
             - "scene_only": scene labels without object overlays
    """
    def get_dataloader(self, ds_name, train=True):
        return DataLoader(
            self.datasets[ds_name]["train" if train else "val"],
            batch_size=self.batch_size, shuffle=self.shuffle, drop_last=True
        )
