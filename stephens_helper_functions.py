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


def create_dataset():
    filepath = "/home/wpx1/deepdream/data/tiny-imagenet-200"
    data_dir = pathlib.Path(filepath).with_suffix('')

    batch_size = 32
    img_height = 64
    img_width = 64

    train_data_dir = str(data_dir) + "/train"

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_data_dir,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    return train_ds


def pick_random_choices(quantity, start=0, end=0, seed=None):
    ''' pick a quantity of random items in an array.
        (optional random seed) '''

    num_list = range(start, end + 1)
    arr = []

    for i in range(start, end + 1):
        for subset in itertools.combinations(num_list, i):
            arr.append(subset)

    if seed:
        random.seed(seed)

    return random.choices(arr, k=quantity)


def add_prefix(input_tuple):
    ''' converts tuples to lists, then prefixes each index with "mixed",
        then converts each item to a string '''
    array = list()

    for item in input_tuple:
        array.append(list(item))

    for sub_array in array:
        for index, item in enumerate(sub_array):
            sub_array[index] = "mixed" + str(item)
    return array


def deprocess(img):
    # Normalize an image
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

        loss, img = DeepDream(img, run_steps, tf.constant(step_size))

        # display.clear_output(wait=True)
        # show(deprocess(img))
        # print("Step {}, loss {}".format(step, loss))

    result = deprocess(img)
    # display.clear_output(wait=True)
    show(result)

    return result


def create_layer_activated_model(base_model, layers):
    ''' returns a model with specified layers activated '''

    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)
    return dream_model


''' Archive / Trash

# def unzip():
#     dataset_url = "file:///content/drive/MyDrive/IMICS Research Lab/data/tiny-imagenet-200.zip"
#     archive = tf.keras.utils.get_file(origin=dataset_url, extract=True, archive_format='zip')


# def read_image_from_colab_storage(image, folder, route="train"):
#     "read an image from /root/.keras/datasets/tiny-imagenet-200"

#     if (route == "train" or
#         route == "test" or
#             route == "val"):
#         pass
#     else:
#         print("Please input route=\"train\" or \"test\" or \"val\" ")

#     test_image_path = tf.keras.utils.get_file(
#         image, f"file:///home/wpx1/deepdream/data/tiny-imagenet-200/{route}/{folder}/images/{image}")
#     print(f"Look here for the file: {test_image_path}")

#     img = PIL.Image.open(test_image_path)
#     final_img = np.array(img)

#     return final_img


# def make_filename(filename, number=0, extension="JPEG"):
#     "append number and file extension to a filename"

#     result = f"{filename}_{number}.{extension}"
#     return result

'''
