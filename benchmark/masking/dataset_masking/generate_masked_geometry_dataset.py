from os import path
import os
from PIL import Image, ImageDraw
import random


def generate_masked_geometry_dataset(location: str, height: int, width: int, train_size: int, test_size: int):
    data_locations = [(path.join(location, "train"), train_size), (path.join(location, "test"), test_size)]
    objs = ["circle", "square"]
    for loc, size in data_locations:
        os.makedirs(loc)
        os.makedirs(path.join(loc, "imgs"))
        os.makedirs(path.join(loc, "masks"))
        for label, obj in enumerate(objs):
            for i in range(size // len(objs)):
                img, img_mask = _draw_image(height, width, obj)
                img.save(path.join(loc, "imgs", f"{label}_{i}.png"))
                img_mask.save(path.join(loc, "masks", f"{label}_{i}.png"))


def _draw_image(img_height, img_width, obj="circle"):
    # Open image and mask image
    img, draw = _open_image(img_height, img_width)
    img_mask, draw_mask = _open_image(img_height, img_width, mode="L")  # 8-bit pixels

    # Add random grey polygons for background noise
    for _ in range(3):
        color = random.randint(0, 255)
        color_rgb = (color, color, color)
        n_points = random.randint(3, 6)
        xy = [(random.randint(0, img_width), random.randint(0, img_height)) for _ in range(n_points)]
        draw.polygon(xy, fill=color_rgb)

    # Decide object size and coordinates
    size = random.randint(int(img_width * .05), int(img_width * .3))
    x, y = (random.randint(int(d * .1), int(d * .9)) for d in [img_width, img_height])

    # Draw object
    if obj is "circle":
        radius = size // 2
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="red")
        draw_mask.ellipse((x - radius, y - radius, x + radius, y + radius), fill="white")
    elif obj is "square":
        draw.rectangle(((x, y), (x + size, y + size)), fill="blue")
        draw_mask.rectangle(((x, y), (x + size, y + size)), fill="white")

    return img, img_mask


def _open_image(height, width, mode="RGB"):
    img = Image.new(mode, (height, width))
    draw = ImageDraw.Draw(img)
    return img, draw


if __name__ == "__main__":
    generate_masked_geometry_dataset("../../../data/geometry/basic",
                                     height=100, width=100, train_size=5000, test_size=1000)
