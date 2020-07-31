from pycocotools.coco import COCO
from os import path
import os
from skimage import io
import numpy as np
from PIL import Image
import shutil
from tqdm import tqdm


def extract_coco_objects(coco_dir, output_dir):
    annotation_filename = path.join(coco_dir, "annotations", "instances_train2017.json")
    workspace_dir = path.join(output_dir, "workspace")
    segment_dir = path.join(workspace_dir, "segments")
    coco = COCO(annotation_filename)

    print("Extracting COCO objects...")
    for obj_name in tqdm(OBJ_NAMES):
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
                # Add alpha channel to image array
                image = np.concatenate([image, np.ones((*image.shape[:2], 1)) * 255], axis=-1)
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
                mask = np.stack((mask2d,) * 4, axis=-1)
                cropped_image_array = (image * mask)[int(base_y):int(base_y + bbox_h),
                                                     int(base_x):int(base_x + bbox_w), :].astype(np.uint8)
                cropped_image = Image.fromarray(cropped_image_array, mode="RGBA")
                # Save image segment
                with open(path.join(dst_dir, str(num_imgs).zfill(4) + ".png"), "wb") as fp:
                    cropped_image.save(fp, format="png")
                # Break loop if enough images are created
                num_imgs += 1
                if num_imgs == NUM_IMAGES_PER_CLASS:
                    break
            except Exception as e:
                continue  # Some images throw exceptions, just ignore these


def extract_scenes(miniplaces_dir, out_dir):
    print("Extracting miniplaces scenes...")
    for scene_name in tqdm(SCENE_NAMES):
        src_dir = path.join(miniplaces_dir, "images", "train", scene_name[0], scene_name)
        dst_dir = path.join(out_dir, "workspace", "scenes", scene_name.split('/')[0])
        shutil.copytree(src_dir, dst_dir)


def overlay_objects_on_scenes(out_dir):
    # Workspace directories
    workspace_dir = path.join(out_dir, "workspace")
    segment_workspace = path.join(workspace_dir, "segments")
    scene_workspace = path.join(workspace_dir, "scenes")

    # Dataset directories
    scene_dir = path.join(out_dir, "scene")
    overlay_dir = path.join(out_dir, "overlay")
    mask_dir = path.join(out_dir, "mask")

    objects = os.listdir(segment_workspace)
    scenes = os.listdir(scene_workspace)

    for dir in [overlay_dir, scene_dir, mask_dir]:
        for scene in scenes:
            for obj in objects:
                os.makedirs(path.join(dir, scene, obj), exist_ok=True)

    print("Overlaying objects on scenes...")
    for obj in objects:
        obj_filenames = sorted(os.listdir(path.join(segment_workspace, obj)))[:NUM_IMAGES_PER_CLASS]
        for scene in tqdm(scenes, desc=obj):
            scene_filenames = sorted(os.listdir(path.join(scene_workspace, scene)))
            for i in range(NUM_IMAGES_PER_CLASS):
                # Read object and scene images, get width and height
                obj_filename, scene_filename = obj_filenames[i], scene_filenames[i]
                with open(path.join(scene_workspace, scene, scene_filename), "rb") as fp:
                    scene_image = Image.open(fp).convert("RGBA")
                with open(path.join(segment_workspace, obj, obj_filename), "rb") as fp:
                    obj_image = Image.open(fp).convert("RGBA")
                scene_w, scene_h = scene_image.size
                obj_w, obj_h = obj_image.size

                # Resize the object to fit in the scene. Resized object is between 1/3-1/2 of the scene
                resize_low, resize_high = np.sqrt(3), np.sqrt(2)
                if float(obj_w) / scene_w > float(obj_h) / scene_h:
                    new_obj_w = np.random.randint(int(scene_w / resize_low), int(scene_w / resize_high))
                    new_obj_h = int(float(new_obj_w) / obj_w * obj_h)
                else:
                    new_obj_h = np.random.randint(int(scene_h / resize_low), int(scene_h / resize_high))
                    new_obj_w = int(float(new_obj_h) / obj_h * obj_w)
                obj_image = obj_image.resize((new_obj_w, new_obj_h), Image.BILINEAR)

                # Randomly generate a location to place the object
                row = np.random.randint(0, scene_h - new_obj_h)
                col = np.random.randint(0, scene_w - new_obj_w)

                filename = str(i).zfill(4) + ".jpg"
                # Save scene to scene_only folder
                with open(path.join(scene_dir, scene, obj, filename), "wb") as fp:
                    scene_image.convert("RGB").save(fp, format="jpeg")

                # Paste object on scene and save in overlay folder
                scene_image.paste(obj_image, (col, row), obj_image)
                scene_image = scene_image.convert("RGB")
                with open(path.join(overlay_dir, scene, obj, filename), "wb") as fp:
                    scene_image.save(fp, format="jpeg")

                # Create black-and-white mask and save in mask folder
                # Start from black RGBA image
                mask_image = Image.fromarray(np.zeros((*np.array(scene_image).shape[:2], 4), dtype=np.uint8), mode="RGBA")
                mask_image.paste(obj_image, (col, row), obj_image)  # Paste object on same place
                mask_array = np.array(mask_image)  # [height, width, 4]
                mask_array = mask_array[:, :, -1]  # Take only alpha channel: [width, height, 1]
                mask_array = (mask_array > 0).astype(np.uint8) * 255
                with open(path.join(mask_dir, scene, obj, filename), "wb") as fp:
                    Image.fromarray(mask_array, mode="L").save(fp, format="png")



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
    miniplaces_dir = "../../data/miniplaces"
    out_dir = "../../data/bam"

    print("Cleaning directory...")
    shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    extract_coco_objects(coco_dir, out_dir)
    extract_scenes(miniplaces_dir, out_dir)
    overlay_objects_on_scenes(out_dir)
