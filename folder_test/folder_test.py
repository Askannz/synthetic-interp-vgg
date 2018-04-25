import sys
from os.path import join
from os import listdir
import json
import cv2
import numpy as np
from vgg16 import vgg16

DIR_PATH = sys.argv[1]
RESULTS_PATH = sys.argv[2]
CLASSES_IDS = eval(sys.argv[3])
TOP_SIZE = 5

vgg16 = vgg16("../vgg16_weights.npz")

files_list = listdir(DIR_PATH)

results = {"files": [], "top_indices": [], "classes_probas": [], "top_probas": []}

for f in files_list:

    results["files"].append(f)

    path = join(DIR_PATH, f)
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    probas = vgg16.classify(img)

    top_indices = np.argsort(probas)[::-1][:TOP_SIZE]
    top_probas = probas[top_indices]

    classes_probas = probas[CLASSES_IDS]

    results["top_indices"].append(top_indices.tolist())
    results["top_probas"].append(top_probas.tolist())
    results["classes_probas"].append(classes_probas.tolist())

json.dump(results, open(RESULTS_PATH, "w"))
