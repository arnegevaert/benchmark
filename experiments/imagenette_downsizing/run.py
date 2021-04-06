import argparse
from torchvision import transforms, datasets
from os import path
import os
import torch
import glob
from experiments.imagenette_downsizing.models import Resnet18
from experiments.lib import MethodLoader
from attrbench.suite import Suite
from torch.utils.data import DataLoader
import time
import logging


def get_data(im_size):
    transform_val = transforms.Compose([
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return datasets.ImageFolder(path.join(data_loc, "imagenette2", 'val'), transform=transform_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("suite_config", type=str)
    parser.add_argument("method_config", type=str)
    parser.add_argument("-b", "--batch-size", type=int, required=True)
    parser.add_argument("-n", "--num-samples", type=int, required=True)
    parser.add_argument("-c", "--cuda", action="store_true")
    parser.add_argument("-i", "--save-images", action="store_true")
    parser.add_argument("-a", "--save-attrs", action="store_true")
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    # Parse arguments
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"

    if not path.isdir(args.output):
        os.makedirs(args.output)

    data_loc = os.environ["BM_DATA_LOC"] if "BM_DATA_LOC" in os.environ else path.join(path.dirname(__file__), "../../data")
    models_dir = path.join(data_loc, "models", "ImageNette", "04")

    logging.basicConfig(
        format='[%(asctime)s %(levelname)s] %(message)s',
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S")

    logging.info("Saving images" if args.save_images else "Not saving images")
    logging.info("Saving attributions" if args.save_attrs else "Not saving attributions")

    for im_size in [64, 96, 128, 160, 192, 224, 256]:
        match = glob.glob(path.join(models_dir, f"resnet18_imagenette_size_{im_size}_*"))
        assert(len(match) == 1)
        filename = match[0]
        dataset = get_data(im_size)
        model = Resnet18(num_classes=10, params_loc=filename)
        ml = MethodLoader(model=model, last_conv_layer=model.get_last_conv_layer(),
                          reference_dataset=dataset)
        methods = ml.load_config(args.method_config)
        bm_suite = Suite(model, methods,
                         DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4),
                         device,
                         save_images=args.save_images,
                         save_attrs=args.save_attrs,
                         seed=args.seed)
        bm_suite.load_config(args.suite_config)
        bm_suite.run(args.num_samples, verbose=True)
        bm_suite.save_result(path.join(args.output, f"size_{im_size}.h5"))
