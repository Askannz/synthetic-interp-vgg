import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt

PATH_ORIGINAL = sys.argv[1]
PATH_VARIANT = sys.argv[2]

original_results = h5py.File(PATH_ORIGINAL, 'r')
variant_results = h5py.File(PATH_VARIANT, 'r')
LAYERS = list(original_results.keys())

original = {}
variant = {}

for layer in LAYERS:
    original[layer] = []
    variant[layer] = []
    for i in range(original_results[layer].shape[0]):
        layer_original = original_results[layer][i]
        layer_variant = variant_results[layer][i]

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
    original_mean = np.mean(original[layer], axis=0)
    variant_mean = np.mean(variant[layer], axis=0)
    indices_sorted = np.argsort(original_mean)
    plt.figure()
    plt.bar(np.arange(original_mean.size), original_mean[indices_sorted], 0.5)
    plt.bar(np.arange(original_mean.size) + 0.5, variant_mean[indices_sorted], 0.5)
    plt.xlabel('Filter id')
    plt.ylabel('Activation')
    plt.grid()
    plt.title(layer)

plt.show()
