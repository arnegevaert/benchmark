from pycocotools.coco import COCO
from os import path
import os
from skimage import io
import numpy as np
from PIL import Image


def extract_coco_objects(coco_dir, output_dir):
    annotation_filename = path.join(coco_dir, "annotations", "instances_train2017.json")
    segment_dir = path.join(output_dir, "segments")
    coco = COCO(annotation_filename)

    for obj_name in OBJ_NAMES:
        print(f"Creating {obj_name} segments...")
        masks = []
        # Create directory to save segments
        dst_dir = path.join(segment_dir, obj_name)
        if not path.isdir(dst_dir):
            os.makedirs(dst_dir)

        # Load images
        cat_ids = coco.getCatIds(catNms=[obj_name])  # Gets ID for given label name
        img_ids = coco.getImgIds(catIds=cat_ids)  # Gets IDs for images of given label
        imgs = coco.loadImgs(img_ids)
        num_imgs = 0
        for imgdata in imgs:
            try:
                # Read image file and annotations
                image = io.imread(path.join(coco_dir, "images", "train2017", imgdata["file_name"]))
                annotation_ids = coco.getAnnIds(imgIds=imgdata["id"], catIds=cat_ids, iscrowd=None)
                annotations = coco.loadAnns(annotation_ids)
                # Get largest annotation
                max_area = 0
                max_ann = None
                for ann in annotations:
                    if ann["area"] > max_area:
                        max_ann = ann
                        max_area = ann["area"]
                annotation = max_ann
                # Mask and crop largest annotation
                base_x, base_y, bbox_w, bbox_h = annotation["bbox"]
                mask2d = coco.annToMask(annotation)
                mask = np.stack((mask2d,) * 3, axis=-1)
                cropped_image = Image.fromarray((image * mask)[int(base_y):int(base_y + bbox_h),
                                                               int(base_x):int(base_x + bbox_w), :]).convert("RGBA")
                # Replace (0,0,0) pixels with transparency (0,0,0,0)
                newData = []
                for item in cropped_image.getdata():
                    if item[0] == 0 and item[1] == 0 and item[2] == 0:
                        newData.append((0, 0, 0, 0))
                    else:
                        newData.append(item)
                cropped_image.putdata(newData)
                # Save image and mask
                with open(path.join(dst_dir, imgdata["file_name"][:-4] + ".png"), "wb") as fp:
                    cropped_image.save(fp, format="png")
                masks.append((imgdata["file_name"], mask2d[int(base_y):int(base_y + bbox_h),
                                                           int(base_x):int(base_x + bbox_w)]))
                # Break loop if enough images are created
                num_imgs += 1
                if num_imgs == NUM_IMAGES_PER_CLASS:
                    break
            except:
                continue  # Some images throw exceptions, just ignore these
        # Save the last 100 object masks used for validation
        masks.sort()
        save_start = int(NUM_IMAGES_PER_CLASS * TRAIN_VAL_RATIO)
        print(f"Saving {len(masks[save_start:])} masks for class {obj_name}")
        np.save(path.join(dst_dir, "val_mask.npy"), np.array([m[1] for m in masks[save_start:]]))


def rename_miniplace_scenes():
    pass


def overlay_objects_on_scenes():
    pass


def divide_train_val():
    pass


if __name__ == "__main__":
    NUM_IMAGES_PER_CLASS = 1000
    TRAIN_VAL_RATIO = 0.9
    OBJ_NAMES = [
        'backpack', 'bird', 'dog', 'elephant', 'kite', 'pizza', 'stop_sign',
        'toilet', 'truck', 'zebra'
    ]
    SCENE_NAMES = [
        'bamboo_forest', 'bedroom', 'bowling_alley', 'bus_interior', 'cockpit',
        'corn_field', 'laundromat', 'runway', 'ski_slope', 'track/outdoor'
    ]
    coco_dir = "../../data/coco"
    out_dir = "../../data/bam"

    extract_coco_objects(coco_dir, out_dir)
