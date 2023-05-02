import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import time

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

'''testing for resnet'''
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, Add, Dense, Flatten
from tensorflow.keras.models import Model
''' end testing'''

start_time = time.time()

batch_size = 32
img_height = 64
img_width = 64

train_dataset_dir = "/home/wpx1/deepdream/data/tiny-imagenet-200/train/"
val_dataset_dir = "/home/wpx1/deepdream/data/tiny-imagenet-200/val/images/"

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dataset_dir,
    seed=1,
    image_size=(img_height, img_width),
    batch_size=batch_size)
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dataset_dir,
    seed=1,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


num_classes = len(class_names)


# data_augmentation = keras.Sequential([
#     layers.RandomFlip("horizontal",
#                       input_shape=(img_height,
#                                    img_width,
#                                    3)),
#     layers.RandomRotation(0.1),
#     layers.RandomZoom(0.1)
# ]
# )

callbacks = [
    keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor="val_loss",
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-2,
        # "no longer improving" being further defined as "for at least 3 epochs"
        patience=3,
        verbose=1,
    )
]

# model = Sequential([
#     # data_augmentation,    # uncomment to use data augmentation
#     layers.Rescaling(1./255),
#     layers.Conv2D(16, 3, padding='same', activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Conv2D(32, 3, padding='same', activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Conv2D(64, 3, padding='same', activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Dropout(0.2),
#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(num_classes, name="outputs")
# ])


def resnet_block(inputs, filters, kernel_size, strides):
    x = Conv2D(filters, kernel_size=kernel_size,
               strides=strides, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    shortcut = inputs
    if strides != 1 or inputs.shape[-1] != filters:
        shortcut = Conv2D(filters, kernel_size=1,
                          strides=strides, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def create_resnet(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    num_blocks_list = [3, 4, 6, 3]
    filters_list = [64, 128, 256, 512]
    kernel_size_list = [3, 3, 3, 3]
    strides_list = [1, 2, 2, 2]

    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        filters = filters_list[i]
        kernel_size = kernel_size_list[i]
        strides = strides_list[i]

    for j in range(num_blocks):
        if j == 0:
            x = resnet_block(x, filters, kernel_size, strides)
        else:
            x = resnet_block(x, filters, kernel_size, 1)

    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    return model


model = create_resnet(input_shape=(64, 64, 3), num_classes=200)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=False),
              metrics=['accuracy'])


epochs = 15
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=callbacks
)


end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: ", round(elapsed_time, 2))
