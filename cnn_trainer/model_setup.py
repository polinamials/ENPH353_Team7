import numpy as np
import tensorflow as tf
import cv2
import h5py

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Conv2D,
    Concatenate,
    MaxPool2D,
    Dropout,
    Flatten,
)


input_image = Input(shape=(90, 160, 3))
output_image = Conv2D(64, 4, padding="same")(input_image)
output_image = MaxPool2D(8)(output_image)
output_image = Conv2D(32, 4, padding="same")(output_image)
output_image = MaxPool2D(2)(output_image)
output_image = Dropout(0.5)(output_image)

output_image = Conv2D(16, 4, padding="same")(output_image)
output_image = MaxPool2D(2)(output_image)
output_image = Dropout(0.5)(output_image)

output_image = Flatten()(output_image)
output_image = Dense(128, activation="relu")(output_image)
output_image = Dropout(0.5)(output_image)
output_image = Dense(64, activation="relu")(output_image)
output_image = Dropout(0.5)(output_image)
output_image = Dense(32, activation="relu")(output_image)
output_image = Dropout(0.5)(output_image)
output_image = Dense(3, activation="softmax")(output_image)


image_net = Model(input_image, output_image)
image_net.summary()
image_net.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
image_net.save("/home/fizzer/cnn_trainer/image_net")
