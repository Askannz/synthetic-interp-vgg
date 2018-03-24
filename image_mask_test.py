import sys
import cv2
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from vgg16 import vgg16

IMAGE_PATH = sys.argv[1]
CLASS_ID = int(sys.argv[2])
MASK_TYPE = sys.argv[3]
RESULTS_PATH = sys.argv[4]
MASK_SIZE = 64
MASK_STRIDE = 4
MASK_COLOR = [128, 128, 128]

timestamp_start = time.strftime("%Y-%m-%d %H:%M")

img = cv2.cvtColor(cv2.imread(IMAGE_PATH), cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))

if MASK_TYPE == "square":
    alpha = np.ones((MASK_SIZE, MASK_SIZE), np.float32)
elif MASK_TYPE == "disc":
    alpha = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (MASK_SIZE, MASK_SIZE)) > 0).astype(np.float32)
elif MASK_TYPE == "disc_grad":
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

vgg16 = vgg16("vgg16_weights.npz")
probas = vgg16.classify(img)
original_class_proba = probas[CLASS_ID]

x0 = 0
while x0 + MASK_SIZE <= 224:
    y0 = 0
    while y0 + MASK_SIZE <= 224:

        print(x0, y0)

        img_masked = img.copy()
        img_part = img_masked[y0:y0 + MASK_SIZE, x0:x0 + MASK_SIZE]
        img_masked[y0:y0 + MASK_SIZE, x0:x0 + MASK_SIZE] = (color_flat.astype(
            np.float32) * alpha + img_part.astype(np.float32) * (1 - alpha)).astype(np.uint8)

        probas = vgg16.classify(img_masked)

        class_proba = probas[CLASS_ID]

        diff = original_class_proba - class_proba

        x, y = x0 + int(MASK_SIZE / 2), y0 + int(MASK_SIZE / 2)
        half_stride = int(MASK_STRIDE / 2)
        heatmap[y - half_stride:y + half_stride,
                x - half_stride:x + half_stride] = diff

        y0 += MASK_STRIDE
    x0 += MASK_STRIDE

timestamp_end = time.strftime("%Y-%m-%d %H:%M")
mask_parameters = {"size": MASK_SIZE, "stride": MASK_STRIDE, "type": MASK_TYPE}
results = {"img": img, "heatmap": heatmap, "timestamp_start": timestamp_start,
           "timestamp_end": timestamp_end, "mask": mask_parameters, "class_id": CLASS_ID, "img_path": IMAGE_PATH}
pickle.dump(results, open(RESULTS_PATH, "wb"))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.imshow(heatmap)
plt.show()