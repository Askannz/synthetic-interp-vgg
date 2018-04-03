import cv2
import numpy as np
from vgg16 import vgg16
from monitor import Sender


class MaskingProcessor:

    def __init__(self, monitor=False):

        if monitor:
            self.monitor = True
            self.sender = Sender()
        else:
            self.monitor = False

        self.vgg16 = vgg16("../vgg16_weights.npz")

    def process(self, img, classes_ids, parameters):

        MASK_SIZE = parameters["size"]
        MASK_STRIDE = parameters["stride"]
        MASK_COLOR = parameters["color"]
        MASK_SHAPE = parameters["shape"]

        if MASK_SHAPE == "square":
            alpha = np.ones((MASK_SIZE, MASK_SIZE), np.float32)
        elif MASK_SHAPE == "disc":
            alpha = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (MASK_SIZE, MASK_SIZE)) > 0).astype(np.float32)
        elif MASK_SHAPE == "disc_grad":
            half = int(MASK_SIZE / 2)
            X = np.arange(MASK_SIZE)
            Y = np.arange(MASK_SIZE)
            x, y = np.meshgrid(X, Y)
            x0, y0 = half, half
            d2 = (x - x0)**2 + (y - y0)**2
            alpha = np.clip(((half - 1)**2 - d2) / ((half - 1) ** 2),
                            0.0, 1.0).astype(np.float32)

        alpha = np.stack([alpha, alpha, alpha], axis=2)
        color_flat = np.zeros((MASK_SIZE, MASK_SIZE, 3), np.uint8)
        color_flat[:, :] = MASK_COLOR

        heatmap = np.zeros((224, 224), np.float32)

        probas = self.vgg16.classify(img)
        original_class_proba = sum([probas[k] for k in classes_ids])

        x0 = 0
        while x0 + MASK_SIZE <= 224:
            y0 = 0
            while y0 + MASK_SIZE <= 224:

                img_masked = img.copy()
                img_part = img_masked[y0:y0 + MASK_SIZE, x0:x0 + MASK_SIZE]
                img_masked[y0:y0 + MASK_SIZE, x0:x0 + MASK_SIZE] = (color_flat.astype(
                    np.float32) * alpha + img_part.astype(np.float32) * (1 - alpha)).astype(np.uint8)

                if self.monitor:
                    self.sender.send(img_masked)

                probas = self.vgg16.classify(img_masked)

                class_proba = sum([probas[k] for k in classes_ids])

                diff = original_class_proba - class_proba

                x, y = x0 + int(MASK_SIZE / 2), y0 + int(MASK_SIZE / 2)
                half_stride = int(MASK_STRIDE / 2)
                heatmap[y - half_stride:y + half_stride,
                        x - half_stride:x + half_stride] = diff

                y0 += MASK_STRIDE
            x0 += MASK_STRIDE

        return heatmap
