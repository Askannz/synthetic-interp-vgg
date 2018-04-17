import sys
from os.path import join, split
from os import listdir
import json
import time
import cv2
import numpy as np

DEBUG = False
RANDOM_TH = 0.05

MODELS_DIR = sys.argv[1]
RESULTS_PATH = sys.argv[2]

H, W = 448, 448

objects_list = json.load(open(join(MODELS_DIR, "list.json"), "r"))

timestamp = time.strftime("%Y-%m-%d %H:%M")
results = {"time_start": timestamp, "results": {}}

for k, obj in enumerate(objects_list):
    relative_path = obj["rel_path"]
    obj["absolute_path"] = join(MODELS_DIR, relative_path)

    renders_path = join(obj["absolute_path"], "renders", "ablation")

    obj["variants"] = [v for v in listdir(join(renders_path)) if v != 'original']
    obj["basename"] = split(relative_path)[1]

    obj["renders_filenames"] = listdir(join(renders_path, "original"))

    results["results"][obj["basename"]] = {}

    for j, variant in enumerate(obj["variants"]):

        print("Testing variant %s (%d/%d) of object %s (%d/%d)" % (variant, j+1, len(obj["variants"]), obj["basename"], k+1, len(objects_list)))

        results["results"][obj["basename"]][variant] = {}

        for i, render_f in enumerate(obj["renders_filenames"]):

            print("%d/%d" % (i+1, len(obj["renders_filenames"])))

            original_path = join(renders_path, "original", render_f)
            variant_path = join(renders_path, variant, render_f)

            original_img = cv2.imread(original_path)
            variant_img = cv2.imread(variant_path)

            difference = np.sum(np.abs(original_img.astype(np.float32) - variant_img.astype(np.float32)), axis=2) / (3 * 255)
            importance = np.sum(difference) / (H * W)

            # DEBUG
            if DEBUG and np.random.rand() < RANDOM_TH:
                print("Variant :", variant)
                print("Importance : %f %%" % (importance * 100))
                difference_stacked = np.stack([difference, difference, difference], axis=2)
                debug = variant_img.astype(np.float32) * (1.0 - difference_stacked) + np.array([255, 0, 255], np.float32) * difference_stacked
                debug = debug.astype(np.uint8)
                cv2.imshow('Original', original_img)
                cv2.imshow('Variant', variant_img)
                cv2.imshow('Debug', debug)
                cv2.waitKey(0)

            results["results"][obj["basename"]][variant][render_f] = importance

    timestamp = time.strftime("%Y-%m-%d %H:%M")
    results["time_end"] = timestamp
    json.dump(results, open(RESULTS_PATH, "w"))
