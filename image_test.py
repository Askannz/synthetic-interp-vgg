import sys
import cv2
import numpy as np
from vgg16 import vgg16
from imagenet_classes import class_names

IMAGE_PATH = sys.argv[1]
TOP_SIZE = 5

vgg16 = vgg16("vgg16_weights.npz")

img = cv2.cvtColor(cv2.imread(IMAGE_PATH), cv2.COLOR_BGR2RGB)

probas = vgg16.classify(img)

top_indices = np.argsort(probas)[::-1][:TOP_SIZE]

for i in top_indices:

    name = class_names[i]
    print("%s %f" % (name, probas[i]))
