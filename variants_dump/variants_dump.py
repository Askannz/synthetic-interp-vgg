import sys
import json
import time
import cv2
from os.path import join
from os import listdir
import pickle
from vgg16 import vgg16

CONFIG_PATH = sys.argv[1]
RESULTS_PATH = sys.argv[2]
config = json.load(open(CONFIG_PATH, "r"))
BASENAME = config["basename"]
RENDERS_PATH = config["renders_path"]
SUBFOLDER = "ablation"
VARIANTS = config["variants"]
LAYERS = config["layers"]

vgg16 = vgg16("../vgg16_weights.npz")


for i, variant in enumerate(VARIANTS):

    variant_path = join(RENDERS_PATH, SUBFOLDER, variant)
    files_list = listdir(variant_path)

    timestamp = time.strftime("%Y-%m-%d %H:%M")
    variant_results = {"time_start": timestamp, "variant": variant,
    "basename": BASENAME, "layers": LAYERS, "renders_path": RENDERS_PATH, "subfolder": SUBFOLDER, "results":{}}

    for j, filename in enumerate(files_list):

        print("Variant %s (%d/%d) file %s (%d/%d)" %
              (variant, i + 1, len(VARIANTS), filename, j + 1, len(files_list)))

        filepath = join(variant_path, filename)
        img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
        probas, layers_dump = vgg16.classify_with_dump(img, LAYERS)

        variant_results["results"][filename] = (probas, layers_dump)

    timestamp = time.strftime("%Y-%m-%d %H:%M")
    variant_results["time_end"] = timestamp
    pickle.dump(variant_results, open(join(RESULTS_PATH, variant), "wb"))
