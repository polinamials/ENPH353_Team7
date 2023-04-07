import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import string


# bruh
class PlateDetector:
    def __init__(self):
        self.FRAME_CROP_IDX = 340
        self.AREA_LIMITS = np.array(
            [
                [[98, 98, 98], [110, 110, 110]],
                [[88, 86, 86], [90, 90, 90]],
                [[120, 120, 120], [123, 123, 123]],
                [[199, 199, 199], [203, 203, 203]],
                [[169, 169, 169], [176, 177, 177]],
            ]
        )

        self.SYMBOL_LIMITS = np.array(
            [[[90, 0, 0], [255, 50, 50]], [[180, 50, 50], [255, 120, 200]]]
        )

        self.KERNEL_E = np.ones((2, 2), np.uint8)
        self.KERNEL_D = np.ones((10, 10), np.uint8)

        # self.KERNEL_E_SYM = np.ones((2, 2), np.uint8)
        self.KERNEL_D_SYM = np.ones((2, 2), np.uint8)

        self.SYM_HEIGHT = 150
        self.SYM_WIDTH = 100
        self.SYM_PAD = 10

        self.HEIGHT = 1800
        self.WIDTH = 600

        self.EPSILON = 0.03

        self.PLATE_X_START = 1200
        self.PLATE_X_END = 1500
        self.PLATE_Y_START = 25
        self.PLATE_Y_END = 575

        self.model = tf.keras.models.load_model(
            "/home/fizzer/ros_ws/src/controller_pkg/cnn_trainer/plate_model"
        )
        keys = list(string.ascii_uppercase + "0123456789")
        values = list(np.arange(0, 36))
        self.labels_to_int = dict(zip(keys, values))
        self.int_to_labels = dict(zip(values, keys))

        # data structure:
        # for three frames
        # stack = [{'B':0.56,'C':0.73,'J':0.27,'O':0.34},{'B':0.89,'C':0.90,'1':0.61,'O':0.19},{'B':0.99,'C':0.98,'1':0.89,'0':0.78}]
        self.symbol_prob_stack = []

    def _detect_area(self, img):
        img = img[self.FRAME_CROP_IDX :, :]

        h, w, d = img.shape

        mask = np.zeros((h, w), dtype="uint8")

        for lower, upper in self.AREA_LIMITS:
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
            mask = mask + cv2.inRange(img, lower, upper)

        output = cv2.bitwise_and(img, img, mask=mask)

        erode = cv2.erode(output, self.KERNEL_E, 2)
        dial = cv2.dilate(erode, self.KERNEL_D, 2)

        grey = cv2.cvtColor(dial, cv2.COLOR_BGR2GRAY)

        contours, hierarchy = cv2.findContours(
            grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        perimeters = np.array([cv2.arcLength(c, True) for c in contours])
        areas = np.array([cv2.contourArea(c) for c in contours])

        contours = np.array(contours, dtype=object)

        # remove magic numbers
        contours = np.delete(
            contours,
            np.where(np.logical_or((perimeters <= 600.0), (areas <= 20000)))[0],
            axis=0,
        )
        if len(contours) != 0:
            approx_contours = []
            for c in contours:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, self.EPSILON * peri, True)
                approx_contours.append(approx)

            # make sure we have 4 corners
            approx_contours = np.array(approx_contours, dtype=object).squeeze()
            approx_contours = np.array(approx_contours, dtype="float32").squeeze()

            if approx_contours.shape[0] == 4:
                return img, approx_contours
            else:
                empty = np.array([])
                return empty, empty

        else:
            empty = np.array([])
            return empty, empty

    def _crop_area(self, frame):
        img, approx_contours = self._detect_area(frame)
        img_copy = np.copy(img)

        if approx_contours.shape[0] != 0:
            src = approx_contours[np.argsort(approx_contours[:, 0])]
            src[0:2] = src[0:2][np.argsort(src[0:2][:, 1])]
            src[2:4] = src[2:4][np.argsort(src[2:4][:, 1])]

            dst = np.array(
                [[0, 0], [0, self.HEIGHT], [self.WIDTH, 0], [self.WIDTH, self.HEIGHT]],
                dtype="float32",
            )
            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(img_copy, M, (self.WIDTH, self.HEIGHT))
            return warped

        else:
            empty = np.array([])
            return empty

    def _crop_plate(self, frame):
        area = self._crop_area(frame)

        if area.shape[0] != 0:
            return area[
                self.PLATE_X_START : self.PLATE_X_END,
                self.PLATE_Y_START : self.PLATE_Y_END,
            ]
        else:
            empty = np.array([])
            return empty

    def _get_symbols(self, frame):
        plate = self._crop_plate(frame)

        if plate.shape[0] != 0:
            h, w, d = plate.shape
            mask = np.zeros((h, w), dtype="uint8")

            for lower, upper in self.SYMBOL_LIMITS:
                lower = np.array(lower, dtype="uint8")
                upper = np.array(upper, dtype="uint8")
                mask = mask + cv2.inRange(plate, lower, upper)

            output = cv2.bitwise_and(plate, plate, mask=mask)

            dial = cv2.dilate(output, self.KERNEL_D_SYM, 2)
            grey = cv2.cvtColor(dial, cv2.COLOR_BGR2GRAY)

            contours, hierarchy = cv2.findContours(
                grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )

            # img_copy = np.copy(plate)
            # cont_img = cv2.drawContours(img_copy, contours, -1, (0, 255, 0), 3)

            perimeters = np.array([cv2.arcLength(c, True) for c in contours])
            areas = np.array([cv2.contourArea(c) for c in contours])

            # magic numbers
            sym_contours = np.delete(
                contours,
                np.where(np.logical_or((perimeters <= 200.0), (areas <= 500)))[0],
                axis=0,
            )

            areas = np.delete(
                areas,
                np.where(np.logical_or((perimeters <= 200.0), (areas <= 500)))[0],
                axis=0,
            )

            # we keep only the four largest contours.
            # This is for the case when quite large contours of the inner loops
            # of letters like o and p remain.
            sort = areas.argsort()[::-1]
            sym_contours = np.array(sym_contours, dtype=object)
            sym_contours = sym_contours[sort]
            sym_contours = sym_contours[:4]

            if sym_contours.shape[0] == 4:
                # print("sym contours: ", sym_contours.shape)

                bounding_boxes = np.array(
                    [cv2.boundingRect(c) for c in sym_contours], dtype=object
                )

                # print("symbols: ", len(bounding_boxes))

                (sym_contours, bounding_boxes) = zip(
                    *sorted(
                        zip(sym_contours, bounding_boxes),
                        key=lambda b: b[1][0],
                        reverse=False,
                    )
                )

                grey_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

                symbols = np.array(
                    [
                        grey_plate[
                            b[1] - self.SYM_PAD : b[1] + self.SYM_PAD + b[3],
                            b[0] - self.SYM_PAD : b[0] + self.SYM_PAD + b[2],
                        ]
                        for b in bounding_boxes
                    ],
                    dtype=object,
                )

                return symbols
            else:
                return np.array([])

        else:
            return np.array([])

    def _get_model_symbols(self, frame):
        symbols = self._get_symbols(frame)

        if symbols.shape[0] != 0:
            model_symbols = []
            for s in symbols:
                h, w = s.shape
                crop_top = 0
                crop_bottom = h
                crop_left = 0
                crop_right = w

                if self.SYM_HEIGHT - h >= 0:
                    top = (self.SYM_HEIGHT - h) // 2
                    bottom = self.SYM_HEIGHT - h - top
                else:
                    top = 0
                    bottom = 0
                    crop_top = (h - self.SYM_HEIGHT) // 2
                    crop_bottom = self.SYM_HEIGHT + crop_top

                if self.SYM_WIDTH - w >= 0:
                    left = (self.SYM_WIDTH - w) // 2
                    right = self.SYM_WIDTH - w - left
                else:
                    left = 0
                    right = 0
                    crop_left = (w - self.SYM_WIDTH) // 2
                    crop_right = self.SYM_WIDTH + crop_left

                borderType = cv2.BORDER_REPLICATE
                s = s[crop_top:crop_bottom, crop_left:crop_right]
                model_symbols.append(
                    cv2.copyMakeBorder(s, top, bottom, left, right, borderType)
                )
            model_symbols = np.array(model_symbols)
            model_symbols = model_symbols[..., np.newaxis]
            return model_symbols
        else:
            return np.array([])

    def read_plate(self, frame):
        # already have the correct axis
        symbols = self._get_model_symbols(frame)

        if symbols.shape[0] != 0:
            pred = self.model.predict(symbols)
            highest_prob_pred = np.array([np.argmax(p) for p in pred])
            labels = np.array([self.int_to_labels[l] for l in highest_prob_pred])

            plate_number = "".join(labels)
            return plate_number

        else:
            return ""

    def _add_to_prob_stack(self, frame):
        # already have the correct axis
        symbols = self._get_model_symbols(frame)

        if symbols.shape[0] != 0:
            pred = self.model.predict(symbols)
            highest_prob = np.array([np.max(p) for p in pred])
            highest_prob_pred = np.array([np.argmax(p) for p in pred])
            labels = np.array([self.int_to_labels[l] for l in highest_prob_pred])

            self.symbol_prob_stack.append(highest_prob)

        else:
            return np.array([])


# imgs = np.load("/home/fizzer/ros_ws/src/controller_pkg/cnn_trainer/data/plate_imgs.npy")
# plate_model = tf.keras.models.load_model(
#     "/home/fizzer/ros_ws/src/controller_pkg/cnn_trainer/plate_model"
# )
# pd = PlateDetector()
# img = imgs[143]

# num = pd.read_plate(img)
# print(num)
