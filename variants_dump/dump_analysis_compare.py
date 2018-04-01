import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt

PATH_ORIGINAL = sys.argv[1]
PATH_VARIANT = sys.argv[2]

original_results = pickle.load(open(PATH_ORIGINAL, "rb"))
variant_results = pickle.load(open(PATH_VARIANT, "rb"))
LAYERS = original_results["layers"]
BASENAME = variant_results["basename"]
VARIANT = variant_results["variant"]

original = {}
variant = {}

for layer in LAYERS:
    original[layer] = []
    variant[layer] = []
    for filename in original_results["results"].keys():
        layer_original = original_results["results"][filename][1][layer]
        layer_variant = variant_results["results"][filename][1][layer]

        if layer_original.ndim == 3:
            layer_original = np.max(np.max(layer_original, axis=0), axis=0)
            layer_variant = np.max(np.max(layer_variant, axis=0), axis=0)

        original[layer].append(layer_original)
        variant[layer].append(layer_variant)

    original[layer] = np.array(original[layer])
    variant[layer] = np.array(variant[layer])


nblayers = len(LAYERS)
BAR_WIDTH = 0.1

for layer in LAYERS:
    original_max = np.max(original[layer], axis=0)
    variant_max = np.max(variant[layer], axis=0)
    indices_sorted = np.argsort(original_max)
    plt.figure()
    plt.bar(np.arange(original_max.size), original_max[indices_sorted], 1)
    plt.bar(np.arange(original_max.size), variant_max[indices_sorted], 1)
    plt.grid()
    plt.title(layer)

plt.show()
