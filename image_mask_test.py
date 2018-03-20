import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from vgg16 import vgg16

IMAGE_PATH = sys.argv[1]
CLASS_ID = int(sys.argv[2])
MASK_SIZE = 32
MASK_STRIDE = 16
MASK_COLOR = [128, 128, 128]

vgg16 = vgg16("vgg16_weights.npz")

img = cv2.cvtColor(cv2.imread(IMAGE_PATH), cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))

heatmap = np.zeros((224, 224), np.float32)

probas = vgg16.classify(img)
original_class_proba = probas[CLASS_ID]

x0 = 0
while x0 + MASK_SIZE <= 224:
    y0 = 0
    while y0 + MASK_SIZE <= 224:

        print(x0, y0)

        img_masked = img.copy()
        img_masked[y0:y0+MASK_SIZE, x0:x0+MASK_SIZE] = MASK_COLOR

        probas = vgg16.classify(img_masked)

        class_proba = probas[CLASS_ID]

        diff = original_class_proba - class_proba

        x, y = x0 + int(MASK_SIZE/2), y0 + int(MASK_SIZE/2)
        half_stride = int(MASK_STRIDE/2)
        heatmap[y-half_stride:y+half_stride, x-half_stride:x+half_stride] = diff

        y0 += MASK_STRIDE
    x0 += MASK_STRIDE

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.imshow(heatmap)
plt.show()
