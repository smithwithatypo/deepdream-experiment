import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import stephens_helper_functions as hf


import IPython.display as display
import PIL.Image

import os
import pathlib
import itertools
import random


# TODO: run augmentation on a whole class and export to augmented_small_data/

# class_name = "n01443537"
# id = 0

# input_file = hf.make_filename(class_name, id, extension="JPEG")
# original_img = hf.read_image_from_local_storage(input_file, folder=class_name)
# print(f"Original Image:")
# hf.show(original_img)

# new_image = original_img.copy()
# print(f"New Image:")
# hf.show(new_image)


# hf.export_image_to_local_storage(new_image, id, folder=class_name)


'''
Plan
- make list of all classes and numbers (crawl the filepath)
- set variables for id and number  (this is where the for-loop will go)
- read image from local storage
- test augment one image
- export image to local storage

'''

class_list = ["n01443537"]  # TODO: list comprehension through the filepath
id_list = [0]  # TODO: list comprehension through the filepath

for class_name in class_list:
    for id in id_list:
        input_file = hf.make_filename(class_name, id, extension="JPEG")
        original_img = hf.read_image_from_local_storage(
            input_file, folder=class_name)
        print(f"Original Image:")
        hf.show(original_img)

        new_image = original_img.copy()
        print(f"New Image:")
        hf.show(new_image)

        hf.export_image_to_local_storage(new_image, id, folder=class_name)
