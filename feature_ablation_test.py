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

vgg16 = vgg16("vgg16_weights.npz")

objects_list = json.load(open(join(MODELS_DIR, "list.json"), "r"))

timestamp = time.strftime("%Y-%m-%d %H:%M")
results = {"time_start": timestamp, "results": {}}

for k, obj in enumerate(objects_list):
    relative_path = obj["rel_path"]
    obj["absolute_path"] = join(MODELS_DIR, relative_path)

    config = json.load(open(join(obj["absolute_path"], "config.json"), "r"))
    obj["variants"] = config["variants"]
    obj["basename"] = config["basename"]

    renders_path = join(obj["absolute_path"], "renders", "ablation")

    obj["renders_filenames"] = listdir(join(renders_path, "original"))

    results["results"][obj["basename"]] = {"classes_ids": obj["classes_ids"], "renders_filenames": obj["renders_filenames"], "by_variant":{}}

    for j, variant in enumerate(obj["variants"]):

        print("Testing variant %s (%d/%d) of object %s (%d/%d)" % (variant, j+1, len(obj["variants"]), obj["basename"], k+1, len(objects_list)))

        results["results"][obj["basename"]]["by_variant"][variant] = {}
        top_indices_variant = []
        top_probas_variant = []
        classes_probas_variant = []

        for i, render_f in enumerate(obj["renders_filenames"]):

            print("%d/%d" % (i+1, len(obj["renders_filenames"])))

            path = join(renders_path, variant, render_f)
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

            probas = vgg16.classify(img)

            top_indices = np.argsort(probas)[::-1][:TOP_SIZE]
            top_probas = probas[top_indices]

            classes_probas = probas[obj["classes_ids"]]

            top_indices_variant.append(top_indices.tolist())
            top_probas_variant.append(top_probas.tolist())
            classes_probas_variant.append(classes_probas.tolist())

        results["results"][obj["basename"]]["by_variant"][variant]["top_indices"] = top_indices_variant
        results["results"][obj["basename"]]["by_variant"][variant]["top_probas"] = top_probas_variant
        results["results"][obj["basename"]]["by_variant"][variant]["classes_probas"] = classes_probas_variant

    timestamp = time.strftime("%Y-%m-%d %H:%M")
    results["time_end"] = timestamp
    json.dump(results, open(RESULTS_PATH, "w"))
