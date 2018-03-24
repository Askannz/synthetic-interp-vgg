import sys
import cv2
import numpy as np
from vgg16 import vgg16
import matplotlib.pyplot as plt

IMAGE_PATH_1 = sys.argv[1]
IMAGE_PATH_2 = sys.argv[2]

vgg16 = vgg16("vgg16_weights.npz")

img_1 = cv2.cvtColor(cv2.imread(IMAGE_PATH_1), cv2.COLOR_BGR2RGB)
_, pool5_1 = vgg16.classify_with_dump(img_1)
img_2 = cv2.cvtColor(cv2.imread(IMAGE_PATH_2), cv2.COLOR_BGR2RGB)
_, pool5_2 = vgg16.classify_with_dump(img_2)

print(pool5_1.shape)
pool5_1 = pool5_1.transpose(2, 1, 0).reshape(512, 49).transpose(1, 0).reshape(49, 32, 16)
pool5_2 = pool5_2.transpose(2, 1, 0).reshape(512, 49).transpose(1, 0).reshape(49, 32, 16)
pool5_1 = np.max(pool5_1, axis=0)
pool5_2 = np.max(pool5_2, axis=0)
diff = np.abs(pool5_1 - pool5_2)

maxval = max(np.max(pool5_1), np.max(pool5_2))
pool5_1 = maxval * pool5_1 / np.max(pool5_1)
pool5_2 = maxval * pool5_2 / np.max(pool5_2)
diff = maxval * diff / np.max(diff)

plt.subplot(1, 3, 1)
plt.imshow(pool5_1)
plt.subplot(1, 3, 2)
plt.imshow(pool5_2)
plt.subplot(1, 3, 3)
plt.imshow(diff)
plt.show()
