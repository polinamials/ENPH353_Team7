import numpy as np
import tensorflow as tf
from tensorflow import keras

# from model_setup import velocities_to_actions
import h5py

print("imported all the stuff")

model = tf.keras.models.load_model("/home/fizzer/ros_ws/cnn_trainer/image_net")
print("loaded model")


h5f_imgs = h5py.File("/home/fizzer/ros_ws/cnn_trainer/data/img_data.h5", "r")
img_data = h5f_imgs["img_data"][:]
h5f_imgs.close()
print("loaded images data")

h5f_vel = h5py.File("/home/fizzer/ros_ws/cnn_trainer/data/vel_data.h5", "r")
vel_data = h5f_vel["vel_data"][:]
h5f_vel.close()
print("loaded velocties data")


shift = 8

img_data = img_data[:-shift]
vel_data = vel_data[shift:]


velocities_to_actions = {
    (0.25, 0.0): np.array([1, 0, 0]),
    (0.25, 1.0): np.array([0, 1, 0]),
    (0.25, -1.0): np.array([0, 0, 1]),
}

actions = np.array([velocities_to_actions[tuple(i)] for i in vel_data])

print(img_data.shape)
print(actions.shape)


model.fit(img_data, actions, batch_size=64, epochs=2, validation_split=0.2)

model.save("/home/fizzer/ros_ws/cnn_trainer/trained_model")
