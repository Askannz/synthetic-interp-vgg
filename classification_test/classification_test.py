import sys
from os.path import join
from os import listdir
import json
import time
import cv2
import numpy as np
from vgg16 import vgg16

MODELS_DIR = sys.argv[1]
RESULTS_PATH = sys.argv[2]
TOP_SIZE = 5

vgg16 = vgg16("../vgg16_weights.npz")

objects_list = json.load(open(join(MODELS_DIR, "list.json"), "r"))

timestamp = time.strftime("%Y-%m-%d %H:%M")
results = {"time_start": timestamp, "results": {}}

for k, obj in enumerate(objects_list):

    relative_path = obj["rel_path"]
    obj["absolute_path"] = join(MODELS_DIR, relative_path)

    config = json.load(open(join(obj["absolute_path"], "config.json"), "r"))
    obj["basename"] = config["basename"]

    print("Testing object %s (%d/%d)" % (obj["basename"], k+1, len(objects_list)))

    renders_path = join(obj["absolute_path"], "renders", "original")

    obj["renders_filenames"] = listdir(renders_path)

    results["results"][obj["basename"]] = {"classes_ids": obj["classes_ids"], "renders_filenames": obj["renders_filenames"], "top_indices": [], "top_probas": [], "classes_probas": []}

    for i, render_f in enumerate(obj["renders_filenames"]):

        print("%d/%d" % (i+1, len(obj["renders_filenames"])))

        path = join(renders_path, render_f)
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

        probas = vgg16.classify(img)

        top_indices = np.argsort(probas)[::-1][:TOP_SIZE]
        top_probas = probas[top_indices]

        classes_probas = probas[obj["classes_ids"]]

        results["results"][obj["basename"]]["top_indices"].append(top_indices.tolist())
        results["results"][obj["basename"]]["top_probas"].append(top_probas.tolist())
        results["results"][obj["basename"]]["classes_probas"].append(classes_probas.tolist())

    timestamp = time.strftime("%Y-%m-%d %H:%M")
    results["time_end"] = timestamp
    json.dump(results, open(RESULTS_PATH, "w"))
