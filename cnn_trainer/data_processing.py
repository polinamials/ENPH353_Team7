import numpy as np
import cv2
import h5py

# Loading all the data
imgs = np.load("/home/fizzer/ros_ws/cnn_trainer/data/imgs_0.npy")
vels = np.load("/home/fizzer/ros_ws/cnn_trainer/data/vels_0.npy")

# remove all data where the robot is not moving forward
not_moving_fwd_idx = np.where(vels[:, 0] == 0.0)[0]

img_data = np.delete(imgs, not_moving_fwd_idx, axis=0)
vel_data = np.delete(vels, not_moving_fwd_idx, axis=0)

print(img_data.shape)
print(vel_data.shape)

print("starting data normalization")
norm_img_data = img_data / 255
print(norm_img_data.dtype)
print("finished normalizing data")


img_data_h5py = h5py.File("/home/fizzer/ros_ws/cnn_trainer/data/img_data.h5", "w")
img_data_h5py.create_dataset("img_data", data=norm_img_data)

vel_data_h5py = h5py.File("/home/fizzer/ros_ws/cnn_trainer/data/vel_data.h5", "w")
vel_data_h5py.create_dataset("vel_data", data=vel_data)

img_data_h5py.close()
vel_data_h5py.close()

"""cv2.imshow("win", img_data[50])
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
