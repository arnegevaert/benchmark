from torchvision import transforms
from torch.utils.data import Dataset
from os import path
import os
from PIL import Image


class BAMDataset(Dataset):
    def __init__(self, data_location, train, include_orig_scene=False, include_mask=False):
        self.data_location = data_location
        self.include_orig_scene = include_orig_scene
        self.include_mask = include_mask
        # TODO Overlay/Scene: normalize to distribution around 0
        # Mask: Divide by 255 to get binary mask
        self.transforms = {
            "overlay": transforms.Compose([transforms.ToTensor()]),
            "scene": transforms.Compose([transforms.ToTensor()]),
            "mask": transforms.Compose([transforms.ToTensor(), transforms.Normalize(0, 255)])
        }
        self.version = "train" if train else "test"

        overlay_dir = path.join(data_location, self.version, "overlay")
        self.scenes = sorted(os.listdir(overlay_dir))
        self.objects = sorted(os.listdir(path.join(overlay_dir, self.scenes[0])))
        self.images_per_obj_folder = len(os.listdir(path.join(overlay_dir, self.scenes[0], self.objects[0])))
        self.images_per_scene_folder = len(self.objects) * self.images_per_obj_folder

    def __getitem__(self, item):
        scene_index = item // self.images_per_scene_folder
        item -= scene_index * self.images_per_scene_folder
        object_index = item // self.images_per_obj_folder
        item -= object_index * self.images_per_obj_folder
        filename = str(item).zfill(4) + ".png"
        subpath = path.join(self.scenes[scene_index], self.objects[object_index], filename)
        result = []
        with open(path.join(self.data_location, self.version, "overlay", subpath), "rb") as fp:
            result.append(self.transforms["overlay"](Image.open(fp)))
        if self.include_orig_scene:
            with open(path.join(self.data_location, self.version, "scene", subpath), "rb") as fp:
                result.append(self.transforms["scene"](Image.open(fp)))
        if self.include_mask:
            with open(path.join(self.data_location, self.version, "mask", subpath), "rb") as fp:
                result.append(self.transforms["mask"](Image.open(fp)))
        result.append(scene_index)
        result.append(object_index)
        return tuple(result)

    def __len__(self):
        return len(self.scenes) * self.images_per_scene_folder


if __name__ == "__main__":
    ds = BAMDataset("../../../data/bam", True)
    item = ds[905]
