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
import time


time_start = time.time()


# Create the base model: InceptionV3

base_model = tf.keras.applications.InceptionV3(
    include_top=False, weights='imagenet')


"""# Testing """  # Executing code"""


# Decides which random layers will be activated

quantity_of_choices = 5
seed = 1    # select None for pure randomness

random_layer_combinations = hf.pick_random_choices(
    quantity=quantity_of_choices, start=0, end=10, seed=seed)
random_layer_combinations_with_prefix = hf.add_prefix(
    random_layer_combinations)


activation_layers = random_layer_combinations_with_prefix[0]

print(f"Activation layers: {activation_layers}")
layers = [base_model.get_layer(name).output for name in activation_layers]
dream_model = hf.create_layer_activated_model(
    base_model=base_model, layers=layers)
deepdream = hf.DeepDream(dream_model)


''' TESTING'''
train_ds = hf.create_dataset()

# this works. Now make a custom augmentation
normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# (tutorial comment): Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

augmented_image = hf.run_deep_dream_simple(img=first_image, steps=10,
                                           step_size=0.01)  # prints augmented image

print(augmented_image)
# hf.show(augmented_image)
# tf.keras.utils.save_img("./data/modified1/test.jpg", augmented_image)


print(f"------ {round((time.time() - time_start), 2)} seconds ------")


# DEPRECATED CODE
#
#
#
# id = "n01443537"
# numbers = [0, 10, 100, 102, 103]

# for number in numbers:
#     input_file = hf.make_filename(id, number, extension="JPEG")
#     original_img = hf.read_image_from_colab_storage(input_file, folder=id)

#     print(f"Original Image:")
#     hf.show(original_img)

#     hf.run_deep_dream_simple(img=original_img, steps=100,
#                              step_size=0.01)  # prints augmented image


# class DeepDream(tf.Module):
#     def __init__(self, model):
#         self.model = model

#     @tf.function(
#         input_signature=(
#             tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
#             tf.TensorSpec(shape=[], dtype=tf.int32),
#             tf.TensorSpec(shape=[], dtype=tf.float32),)
#     )
#     def __call__(self, img, steps, step_size):
#         print("Tracing")
#         loss = tf.constant(0.0)
#         for n in tf.range(steps):
#             with tf.GradientTape() as tape:
#                 # This needs gradients relative to `img`
#                 # `GradientTape` only watches `tf.Variable`s by default
#                 tape.watch(img)
#                 loss = hf.calc_loss(img, self.model)

#             # Calculate the gradient of the loss with respect to the pixels of the input image.
#             gradients = tape.gradient(loss, img)

#             # Normalize the gradients.
#             gradients /= tf.math.reduce_std(gradients) + 1e-8

#             # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
#             # You can update the image by directly adding the gradients (because they're the same shape!)
#             img = img + gradients*step_size
#             img = tf.clip_by_value(img, -1, 1)

#         return loss, img


# please use create_dataset() instead
# id = "n01443537"
# number = 103              # try 0, 10, 100, 102, 103

# input_file = hf.make_filename(id, number, extension="JPEG")
# test_img = hf.read_image_from_colab_storage(input_file, folder=id)
# hf.show(test_img)

# original_img = test_img
