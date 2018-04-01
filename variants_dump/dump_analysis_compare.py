import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt


def make_disp(data):

    disp = {}
    for layer in LAYERS:

        if data[layer].ndim == 4:
            max_layer = np.max(np.max(data[layer], axis=1), axis=1)
        else:
            max_layer = data[layer]

        cumul = np.sum(max_layer, axis=0)

        disp[layer] = np.abs(cumul)

    return disp


PATH_ORIGINAL = sys.argv[1]
PATH_VARIANT = sys.argv[2]

original_results = pickle.load(open(PATH_ORIGINAL, "rb"))
variant_results = pickle.load(open(PATH_VARIANT, "rb"))
LAYERS = original_results["layers"]
BASENAME = variant_results["basename"]
VARIANT = variant_results["variant"]

diffs = {}

for layer in LAYERS:
    diffs[layer] = []
    for filename in original_results["results"].keys():
        layer_original = original_results["results"][filename][1][layer]
        layer_variant = variant_results["results"][filename][1][layer]
        diffs[layer].append(layer_original - layer_variant)

    diffs[layer] = np.array(diffs[layer])

diffs_disp = make_disp(diffs)

nblayers = len(LAYERS)

for layer in LAYERS:
    plt.figure()
    plt.plot(diffs_disp[layer])
    plt.title(layer)

plt.show()
