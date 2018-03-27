import sys
import json
import numpy as np
import matplotlib.pyplot as plt

RESULTS_PATH = sys.argv[1]

results = json.load(open(RESULTS_PATH, "r"))

for basename in results["results"].keys():

    classes_ids = results["results"][basename]["classes_ids"]
    original_classes_probas = results["results"][basename]["by_variant"]["original"]["classes_probas"]
    original_classes_probas = np.sum(original_classes_probas, axis=1)

    for variant in results["results"][basename]["by_variant"]:

        if variant == "original":
            continue

        variant_classes_probas = results["results"][basename]["by_variant"][variant]["classes_probas"]
        variant_classes_probas = np.sum(variant_classes_probas, axis=1)

        classes_probas = np.stack([original_classes_probas, variant_classes_probas], axis=1)
        classes_probas_sorted = classes_probas[classes_probas[:, 0].argsort()]

        plt.plot(classes_probas_sorted[:, 0], marker='+', label='original')
        plt.plot(classes_probas_sorted[:, 1], marker='*', label=variant)
        plt.legend(loc='upper left')
        plt.grid()
        plt.title("%s %s" % (basename, variant))
        plt.show()
