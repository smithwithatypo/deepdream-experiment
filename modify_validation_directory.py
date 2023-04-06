# make folders for each class name
# make dictionary of images and their class from val_annotations.txt
# move each file into the corresponding folder

import os
import shutil

val_dataset_dir = 'data/tiny-imagenet-200/val/images'
val_annotations_path = 'data/tiny-imagenet-200/val/val_annotations.txt'


def read_val_annotations(val_annotations_path):
    with open(val_annotations_path, 'r') as f:
        lines = f.readlines()
    image_class_dict = {}
    for line in lines:
        image, class_name, x, y, h, w = line.split('\t')
        image_class_dict[image] = class_name
    return image_class_dict


def get_class_names(val_annotations_path):
    with open(val_annotations_path, 'r') as f:
        lines = f.readlines()
    class_names = set()
    for line in lines:
        image, class_name, x, y, h, w = line.split('\t')
        class_names.add(class_name)
    return class_names


def make_class_folders(class_names, val_dataset_dir):
    for class_name in class_names:
        os.mkdir(os.path.join(val_dataset_dir, class_name))


def put_images_in_class_folders(val_dataset_dir, image_class_dict):
    for image, class_name in image_class_dict.items():
        src = os.path.join(val_dataset_dir, image)
        dst = os.path.join(val_dataset_dir, class_name, image)
        shutil.move(src, dst)


d = read_val_annotations(val_annotations_path)
class_names = get_class_names(val_annotations_path)
make_class_folders(class_names, val_dataset_dir)
put_images_in_class_folders(val_dataset_dir, d)
print("done")
