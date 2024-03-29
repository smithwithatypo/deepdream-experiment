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


augmented_batch = "augmented10"   # get from augmented/metadata.txt
names = ['mixed2', 'mixed3']   # change activated layers here
activated_gpu = int(6)  # change activated GPU here


'''
Don't need to change anything below this line
To run:  $ nohup python augment_dataset.py > augmented##.log &
'''

start_time = time.time()
# tf.debugging.set_log_device_placement(True)  # shows if GPU or CPU is used

# GPU config
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use one GPU
        tf.config.set_visible_devices(gpus[activated_gpu], 'GPU')
        # Allow TensorFlow to allocate only as much GPU memory as needed
        tf.config.experimental.set_memory_growth(gpus[activated_gpu], True)
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)


filepath = f"./data/tiny-imagenet-200/train/"
directories = hf.make_list_of_directories_from_filepath(filepath)

base_model = tf.keras.applications.InceptionV3(
    include_top=False, weights='imagenet')

layers = [base_model.get_layer(name).output for name in names]

dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)


def deprocess(img):
    img = 255*(img + 1.0)/2.0
    return tf.cast(img, tf.uint8)


def show(img):
    display.display(PIL.Image.fromarray(np.array(img)))


def calc_loss(img, model):
    # Pass forward the image through the model to retrieve the activations.
    # Converts the image into a batch of size 1.
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)
    if len(layer_activations) == 1:
        layer_activations = [layer_activations]

    losses = []
    for act in layer_activations:
        loss = tf.math.reduce_mean(act)
        losses.append(loss)

    return tf.reduce_sum(losses)

# original tutorial code for single GPU


class DeepDream(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.int32),
            tf.TensorSpec(shape=[], dtype=tf.float32),)
    )
    def __call__(self, img, steps, step_size):
        print("Tracing")
        loss = tf.constant(0.0)
        for n in tf.range(steps):
            with tf.GradientTape() as tape:
                # This needs gradients relative to `img`
                # `GradientTape` only watches `tf.Variable`s by default
                tape.watch(img)
                loss = calc_loss(img, self.model)

            # Calculate the gradient of the loss with respect to the pixels of the input image.
            gradients = tape.gradient(loss, img)

            # Normalize the gradients.
            gradients /= tf.math.reduce_std(gradients) + 1e-8

            # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
            # You can update the image by directly adding the gradients (because they're the same shape!)
            img = img + gradients*step_size
            img = tf.clip_by_value(img, -1, 1)

        return loss, img


deepdream = DeepDream(dream_model)


def run_deep_dream_simple(img, steps=100, step_size=0.01):
    # Convert from uint8 to the range expected by the model.
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    img = tf.convert_to_tensor(img)
    step_size = tf.convert_to_tensor(step_size)
    steps_remaining = steps
    step = 0
    while steps_remaining:
        if steps_remaining > 100:
            run_steps = tf.constant(100)
        else:
            run_steps = tf.constant(steps_remaining)
        steps_remaining -= run_steps
        step += run_steps

        loss, img = deepdream(img, run_steps, tf.constant(step_size))

        display.clear_output(wait=True)
        show(deprocess(img))
        print("Step {}, loss {}".format(step, loss))

    result = deprocess(img)
    display.clear_output(wait=True)
    show(result)

    return result


black_white_images = set()
count = 0

for folder in directories:
    filepath = f"./data/tiny-imagenet-200/train/{folder}/images/"
    files = hf.make_list_of_files_from_filepath(filepath)
    for file in files:
        if os.path.exists(f"./data/tiny-imagenet-200/augmented/{augmented_batch}/{folder}/images/{file}"):
            # print("Already augmented")
            continue

        original_img = hf.read_image_from_local_storage(
            file, folder=folder, route=None)
        # print(f"Original Image:", type(original_img), original_img.shape)

        if original_img.shape == (64, 64):
            print("Image is not RGB")
            black_white_images.add(file)
            continue

        dream_img = run_deep_dream_simple(img=original_img,
                                          steps=20, step_size=0.01)

        # print(f"Dream Image:", type(dream_img), dream_img.shape)
        hf.show(dream_img)
        # break  # debugging before export

        hf.export_image_to_local_storage(
            dream_img, folder=folder, file=file, batch=augmented_batch)
        # break  # comment out to run on all files
    # count += 1
    # if count >= 2:
    #     break  # runs on only 2 folders
    # break  # comment out to run on all folders

print(
    f"Number of black and white images: {len(black_white_images)}")
end_time = time.time()
total_time = end_time - start_time

print("Time elapsed:", round(total_time, 2), "seconds")
