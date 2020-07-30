from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from os import path


class BAMDataset(Dataset):
    def __init__(self, batch_size, data_location, train, shuffle=True):
        self.batch_size = batch_size
        self.data_location = data_location
        self.shuffle = shuffle
        self.transforms = transforms.Compose([transforms.ToTensor()])  # TODO normalize
        version = "train" if train else "val"
        self.datasets = {
            ds_name: ImageFolder(path.join(self.data_location, ds_name, version), transform=self.transforms)
            for ds_name in ["obj", "scene", "scene_only"]
        }

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

    """
    ds_name: - "obj": object labels
             - "scene": scene labels
             - "scene_only": scene labels without object overlays
    """
    def get_dataloader(self, ds_name, train=True):
        return DataLoader(
            self.datasets[ds_name]["train" if train else "val"],
            batch_size=self.batch_size, shuffle=self.shuffle, drop_last=True, num_workers=4)
