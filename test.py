import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import stephens_helper_functions as hf
import DeepDream


import IPython.display as display
import PIL.Image

import os
import pathlib
import itertools
import random
import time

time_start = time.time()

# TEST: create a dataset
train_ds = hf.create_dataset()
class_names = train_ds.class_names

test_dataset = train_ds.take(1)

for images, labels in test_dataset:
    test_image = images[0]
    test_label = class_names[labels[0]]
    break
print(test_image)
print(test_label)


print(f"------ {round((time.time() - time_start), 2)} seconds ------")
