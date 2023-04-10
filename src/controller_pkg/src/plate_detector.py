import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import string


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
        self.SYM_PAD = 0

        self.PNUM_HEIGHT = 375
        self.PNUM_WIDTH = 225
        self.PNUM_PAD = 0

        self.HEIGHT = 1800
        self.WIDTH = 600

        self.EPSILON = 0.03

        self.PLATE_X_START = 1200
        self.PLATE_X_END = 1500
        self.PLATE_Y_START = 25
        self.PLATE_Y_END = 575

        # original vals: 600 and 20000
        self.AREA_PER_LIM = 500.0
        self.AREA_AREA_LIM = 8000.0
        self.SYM_PER_LIM = 200.0
        self.SYM_AREA_LIM = 500.0

        self.plate_model = tf.keras.models.load_model(
            "/home/fizzer/ros_ws/src/controller_pkg/cnn_trainer/plate_model_2"
        )
        self.pnum_model = tf.keras.models.load_model(
            "/home/fizzer/ros_ws/src/controller_pkg/cnn_trainer/pnums_model"
        )
        keys = list(string.ascii_uppercase + "0123456789")
        values = list(np.arange(0, 36))
        self.labels_to_int = dict(zip(keys, values))
        self.int_to_labels = dict(zip(values, keys))

        self.symbol_prob_stack = []
        self.pnum_prob_stack = []
        # testing only
        self.symbol_stack = []
        self.pnum_stack = []

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

        # remove magic numbers
        # for areas used to be 20
        contours = np.delete(
            contours,
            np.where(
                np.logical_or(
                    (perimeters <= self.AREA_PER_LIM), (areas <= self.AREA_AREA_LIM)
                )
            )[0],
            axis=0,
        )

        conv_conts = []
        for c in contours:
            if cv2.isContourConvex(cv2.convexHull(c)):
                # print("convex")
                conv_conts.append(c)

        contours = np.array(conv_conts)
        if len(contours) != 0:
            approx_contours = []
            for c in contours:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, self.EPSILON * peri, True)
                approx_contours.append(approx)

            # make sure we have 4 corners
            approx_contours = np.array(approx_contours).squeeze()
            # approx_contours = np.array(approx_contours, dtype="float32").squeeze()

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
            src = approx_contours[np.argsort(approx_contours[:, 0])].astype("float32")
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

    # PARKING NUMBER DETECTION

    def _get_model_pnum(self, frame):
        area = self._crop_area(frame)

        if area.shape[0] != 0:
            m = 650
            n = 1100
            o = 15
            p = 585
            pnum = area[m:n, o:p]
            grey = cv2.cvtColor(pnum, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(grey, 25, 255, cv2.THRESH_BINARY_INV)

            contours, hierarchy = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            conv_contours = []
            for c in contours:
                if cv2.isContourConvex(cv2.convexHull(c)):
                    conv_contours.append(c)
            contours = np.array(conv_contours).squeeze()

            if contours.shape[0] == 2:
                bounding_boxes = np.array(
                    [cv2.boundingRect(c) for c in contours], dtype=object
                )
                (contours, bounding_boxes) = zip(
                    *sorted(
                        zip(contours, bounding_boxes),
                        key=lambda b: b[1][0],
                        reverse=False,
                    )
                )
                box = bounding_boxes[1]
                box_peri = (box[2] + box[3]) * 2

                if box_peri < 725:
                    pad = 100

                else:
                    pad = 30
                num = np.array(
                    grey[
                        box[1] - pad : box[1] + pad + box[3],
                        box[0] - pad : box[0] + pad + box[2],
                    ]
                )
                # print(num.shape)

                h, w = num.shape
                crop_top = 0
                crop_bottom = h
                crop_left = 0
                crop_right = w

                if self.PNUM_HEIGHT - h >= 0:
                    top = (self.PNUM_HEIGHT - h) // 2
                    bottom = self.PNUM_HEIGHT - h - top
                else:
                    top = 0
                    bottom = 0
                    crop_top = (h - self.PNUM_HEIGHT) // 2
                    crop_bottom = self.PNUM_HEIGHT + crop_top

                if self.PNUM_WIDTH - w >= 0:
                    left = (self.PNUM_WIDTH - w) // 2
                    right = self.PNUM_WIDTH - w - left
                else:
                    left = 0
                    right = 0
                    crop_left = (w - self.PNUM_WIDTH) // 2
                    crop_right = self.PNUM_WIDTH + crop_left

                borderType = cv2.BORDER_REPLICATE
                num = num[crop_top:crop_bottom, crop_left:crop_right]

                model_num = cv2.copyMakeBorder(
                    num, top, bottom, left, right, borderType
                )

                model_num = cv2.resize(model_num, [100, 150])[..., np.newaxis][
                    np.newaxis, ...
                ]

                return model_num

            else:
                return np.array([])

        else:
            return np.array([])

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
                grey, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            img_copy = np.copy(plate)

            perimeters = np.array([cv2.arcLength(c, True) for c in contours])
            areas = np.array([cv2.contourArea(c) for c in contours])

            # magic numbers
            contours = np.delete(
                contours,
                np.where(
                    np.logical_or(
                        (perimeters <= self.SYM_PER_LIM), (areas <= self.SYM_AREA_LIM)
                    )
                )[0],
                axis=0,
            )

            sym_contours = []
            for c in contours:
                if cv2.isContourConvex(cv2.convexHull(c)):
                    # print("convex")
                    sym_contours.append(c)

            sym_contours = np.array(sym_contours)

            # get areas again
            areas = np.array([cv2.contourArea(c) for c in sym_contours])

            # we keep only the four largest contours.
            sort = areas.argsort()[::-1]
            sym_contours = sym_contours[sort]
            sym_contours = sym_contours[:4]

            if sym_contours.shape[0] == 4:
                # print("sym contours: ", sym_contours.shape)

                bounding_boxes = np.array(
                    [cv2.boundingRect(c) for c in sym_contours], dtype=object
                )

                # print("symbols: ", len(bounding_boxes))
                for box in bounding_boxes:
                    cont_img = cv2.rectangle(img_copy, box, (0, 0, 0))
                cont_img = cv2.cvtColor(
                    cv2.drawContours(img_copy, sym_contours, -1, (0, 0, 0), 3),
                    cv2.COLOR_BGR2GRAY,
                )

                # sorting from left to right
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

                return symbols, cont_img
            else:
                return np.array([]), np.array([])

        else:
            return np.array([]), np.array([])

    def _get_model_symbols(self, frame):
        symbols, cont_img = self._get_symbols(frame)

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
            return model_symbols, cont_img
        else:
            return np.array([]), np.array([])

    def read_plate(self, frame):
        # already have the correct axis
        symbols, cont_img = self._get_model_symbols(frame)

        if symbols.shape[0] != 0:
            model_symbols = symbols / 255
            pred = self.plate_model.predict(model_symbols)
            highest_prob_pred = np.array([np.argmax(p) for p in pred])
            labels = np.array([self.int_to_labels[l] for l in highest_prob_pred])

            plate_number = "".join(labels)
            return plate_number, symbols, cont_img

        else:
            return "", np.array([]), np.array([])

    def add_to_prob_stack(self, frame):
        symbols, cont_img = self._get_model_symbols(frame)
        pnum = self._get_model_pnum(frame)

        if (symbols.shape[0] != 0) & (pnum.shape[0] != 0):
            model_symbols = symbols / 255
            model_pnum = pnum / 255

            plate_pred = self.plate_model.predict(model_symbols)
            pnum_pred = self.pnum_model(model_pnum)

            self.symbol_prob_stack.append(plate_pred)
            self.pnum_prob_stack.append(pnum_pred)

            # print(self.pnum_prob_stack)

            # testing only
            self.pnum_stack.append(pnum)
            self.symbol_stack.append(symbols)

    def get_stack_size(self):
        return len(self.symbol_prob_stack)

    def clear_sym_prob_stack(self):
        self.symbol_prob_stack = []

    def clear_sym_stack(self):
        self.symbol_stack = []

    def clear_pnum_prob_stack(self):
        self.pnum_prob_stack = []

    def clear_pnum_stack(self):
        self.pnum_stack = []

    def read_best_plate(self):
        self.symbol_prob_stack = np.array([self.symbol_prob_stack]).squeeze()
        self.pnum_prob_stack = np.array(self.pnum_prob_stack).squeeze()

        if self.symbol_prob_stack.shape[0] != 0:
            max_prob_in_each_row = np.max(self.symbol_prob_stack, axis=-1)

            if max_prob_in_each_row.ndim > 1:
                best_row_for_each = np.argmax(max_prob_in_each_row, axis=0)
                best_probs_for_each = np.array(
                    [max_prob_in_each_row[best_row_for_each[i], i] for i in range(4)]
                )
                int_labels = np.array(
                    [
                        np.where(
                            self.symbol_prob_stack[row][s] == best_probs_for_each[s]
                        )
                        for s, row in enumerate(best_row_for_each)
                    ]
                ).squeeze()
                best_syms = np.array(
                    [
                        self.symbol_stack[row][s]
                        for s, row in enumerate(best_row_for_each)
                    ]
                ).squeeze()

                pnum = (
                    np.unravel_index(
                        self.pnum_prob_stack.argmax(), self.pnum_prob_stack.shape
                    )[1]
                    + 1
                )

            else:
                best_row_for_each = np.zeros(4).astype(int)
                best_probs_for_each = max_prob_in_each_row
                int_labels = np.array(
                    [
                        np.where(self.symbol_prob_stack[s] == best_probs_for_each[s])
                        for s, row in enumerate(best_row_for_each)
                    ]
                ).squeeze()
                best_syms = np.copy(self.symbol_stack).squeeze()

                pnum = np.argmax(self.pnum_prob_stack) + 1

            labels = np.array([self.int_to_labels[l] for l in int_labels])
            plate_number = "".join(labels)
            return plate_number, best_syms, pnum
        else:
            return "", np.array([]), 0


# imgs = np.load("/home/fizzer/ros_ws/src/controller_pkg/cnn_trainer/data/plate_imgs.npy")

# pd = PlateDetector()
# img1 = imgs[1]
# img2 = imgs[33]
# img3 = imgs[34]
# img4 = imgs[35]


# pd.add_to_prob_stack(img1)
# pd.add_to_prob_stack(img2)
# pd.add_to_prob_stack(img3)
# pd.add_to_prob_stack(img4)


# plate, syms, cont_img = pd.read_plate(img4)

# syms = syms.squeeze()
# cont_img = cv2.resize(cont_img, (275, 150))

# print(syms.shape)
# print(cont_img.shape)

# final_img = np.concatenate([cont_img, syms[0], syms[1], syms[2], syms[3]], axis=1)
# print(plate)

# plate, syms, pnum = pd.read_best_plate()
# pred = np.argmax(pd.pnum_model(pnum))

# print(pnum)

# np.save("/home/fizzer/ros_ws/src/controller_pkg/cnn_trainer/plates/pnums.npy", pnums)

# if len(pd.pnum_stack) != 0:
#     im = pd.pnum_stack[0].astype("uint8").squeeze()

#     cv2.imshow("win", im)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print("no pnum detected")
