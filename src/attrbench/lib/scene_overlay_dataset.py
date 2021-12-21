from torchvision import transforms
from torch.utils.data import Dataset
from os import path
import os
from PIL import Image


class SceneOverlayDataset(Dataset):
    def __init__(self, data_location, train, include_orig_scene=False, include_mask=False):
        self.data_location = data_location
        self.include_orig_scene = include_orig_scene
        self.include_mask = include_mask
        self.sample_shape = (3, 128, 128)
        self.train_transforms = {
            "overlay": transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((.4481, .4329, .4004), (.2315, .2264, .2330))
            ]),
            "scene": transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((.4508, .4418, .4108), (.2230, .2195, .2255))
            ]),
            "mask": transforms.Compose([transforms.ToTensor()])
        }
        self.test_transforms = {
            "overlay": transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((.4481, .4329, .4004), (.2315, .2264, .2330))
            ]),
            "scene": transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((.4508, .4418, .4108), (.2230, .2195, .2255))
            ]),
            "mask": transforms.Compose([transforms.ToTensor()])
        }
        self.version = "train" if train else "test"
        self.num_classes = 10

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
        transf = self.train_transforms if self.version == "train" else self.test_transforms
        with open(path.join(self.data_location, self.version, "overlay", subpath), "rb") as fp:
            result.append(transf["overlay"](Image.open(fp)))
        if self.include_orig_scene:
            with open(path.join(self.data_location, self.version, "scene", subpath), "rb") as fp:
                result.append(transf["scene"](Image.open(fp)))
        if self.include_mask:
            with open(path.join(self.data_location, self.version, "mask", subpath), "rb") as fp:
                result.append(transf["mask"](Image.open(fp)).int())
        result.append(scene_index)
        result.append(object_index)
        return tuple(result)

    def __len__(self):
        return len(self.scenes) * self.images_per_scene_folder
