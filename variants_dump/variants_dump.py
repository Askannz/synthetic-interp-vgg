import sys
import json
import time
import cv2
import numpy as np
from os.path import join
from os import listdir
import h5py
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
    variant_header = {"time_start": timestamp, "variant": variant,
    "basename": BASENAME, "layers": LAYERS, "renders_path": RENDERS_PATH, "subfolder": SUBFOLDER, "files": files_list}

    layers_data = {}
    for layer in LAYERS:
        layers_data[layer] = []

    for j, filename in enumerate(files_list):

        print("Variant %s (%d/%d) file %s (%d/%d)" %
              (variant, i + 1, len(VARIANTS), filename, j + 1, len(files_list)))

        filepath = join(variant_path, filename)
        img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
        _, layers_dump = vgg16.classify_with_dump(img, LAYERS)

        for k, layer in enumerate(LAYERS):
            layers_data[layer].append(layers_dump[layer])

    timestamp = time.strftime("%Y-%m-%d %H:%M")
    variant_header["time_end"] = timestamp

    json.dump(variant_header, open(join(RESULTS_PATH, variant) + ".json", "w"))
    h5f = h5py.File(join(RESULTS_PATH, variant) + ".h5", 'w')
    for layer in LAYERS:
        layers_data[layer] = np.array(layers_data[layer])
        h5f.create_dataset(layer, data=layers_data[layer])
    h5f.close()
