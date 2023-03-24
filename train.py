import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import time

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

start_time = time.time()
# tf.debugging.set_log_device_placement(True)  # shows if GPU or CPU is used

batch_size = 32
img_height = 64
img_width = 64

train_dataset_dir = "/home/wpx1/deepdream/data/tiny-imagenet-200/small_dataset/"
# val_dataset_dir = "/home/wpx1/deepdream/data/tiny-imagenet-200/val/"

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=1,
    image_size=(img_height, img_width),
    batch_size=batch_size)
val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=1,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


num_classes = len(class_names)

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                   img_width,
                                   3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
]
)

model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, name="outputs")
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

epochs = 10
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)


end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: ", round(elapsed_time, 2))
