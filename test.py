import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import helper_functions as hf


import IPython.display as display
import PIL.Image

import os
import pathlib
import itertools
import random
import time

time_start = time.time()

all_combos = hf.find_all_combinations(start=0, end=10)
# print(len(all_combos))

selected_combos = hf.pick_random_choices(all_combos, 9, seed=1)
selected_combos.append(selected_combos[0])
selected_combos[0] = (3, 5)
selected_combos = hf.add_prefix(selected_combos)


for index, combo in enumerate(selected_combos):
    print(f"augmented{index+1} activated layers: {combo}")
    print(f"augmented{index+1} steps: 40")
    print("")
