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


# TEST: print an image
id = "n01443537"
number = 0

input_file = hf.make_filename(id, number, extension="JPEG")
original_img = hf.read_image_from_local_storage(input_file, folder=id)
print(f"Original Image:")
hf.show(original_img)

new_image = original_img.copy()
print(f"New Image:")
hf.show(new_image)


hf.export_image_to_local_storage(new_image, number, folder=id)
