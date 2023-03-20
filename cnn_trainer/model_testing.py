import numpy as np
import tensorflow as tf
from tensorflow import keras
from model_setup import velocities_to_actions
import cv2

model = tf.keras.models.load_model("/home/fizzer/cnn_trainer/trained_model")
imgs_val = np.load("/home/fizzer/cnn_trainer/data/imgs_val_0.npy")
vels_val = np.load("/home/fizzer/cnn_trainer/data/vels_val_0.npy")

not_moving_fwd_idx = np.where(vels_val[:, 0] == 0.0)[0]

img_val_data = np.delete(imgs_val, not_moving_fwd_idx, axis=0)
vel_val_data = np.delete(vels_val, not_moving_fwd_idx, axis=0)

val_x = img_val_data / 255
val_y = np.array([velocities_to_actions[tuple(i)] for i in vel_val_data])

print(img_val_data.shape)
print(vel_val_data.shape)


model.evaluate(val_x, val_y)
pred = model.predict(val_x)

max_indices = np.array([np.argmax(i) for i in pred])
# max_indices
pred_actions = []
for idx in max_indices:
    arr = np.zeros(3)
    arr[idx] = 1
    pred_actions.append(arr)


i = 11

print("predicted value: ", pred_actions[i])
print("actual action: ", val_y[i])
"""
cv2.imshow("win", val_x[i])
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

# print(val_y)
# print(pred_actions)


pred_actions = np.array(pred_actions)
print(pred_actions.shape)

straight_idx = np.where(val_y[:, 1] == 0.0)[0]
left_turns_idx = np.where(val_y[:, 1] > 0.0)[0]
right_turns_idx = np.where(val_y[:, 1] < 0.0)[0]


going_straight_true = val_y[straight_idx, :]
turning_left_true = val_y[left_turns_idx, :]
turning_right_true = val_y[right_turns_idx, :]

going_straight_img = val_x[straight_idx, :]
turning_left_img = val_x[left_turns_idx, :]
turning_right_img = val_x[right_turns_idx, :]


going_straight_pred = pred_actions[straight_idx]
turning_left_pred = pred_actions[left_turns_idx]
turning_right_pred = pred_actions[right_turns_idx]

# straight
print("Going straight:")
"""for i in range(len(going_straight_true)):
    print(
        going_straight_true[i],
        going_straight_pred[i],
        np.allclose(going_straight_true[i], going_straight_pred[i]),
    )"""
model.evaluate(going_straight_img, going_straight_true)

# left
print("Turning left:")
"""for i in range(len(turning_left_true)):
    print(
        turning_left_true[i],
        turning_left_pred[i],
        np.allclose(turning_left_true[i], turning_left_pred[i]),
    )"""

model.evaluate(turning_left_img, turning_left_true)

# right
# print("Turning right:")
"""for i in range(len(turning_right_true)):
    print(
        turning_right_true[i],
        turning_right_pred[i],
        np.allclose(turning_right_true[i], turning_right_pred[i]),
    )
"""
