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

        plt.bar(np.arange(classes_probas_sorted.shape[0]), classes_probas_sorted[:, 0], 0.5, label='original')
        plt.bar(np.arange(classes_probas_sorted.shape[0]) + 0.5, classes_probas_sorted[:, 1], 0.5, label=variant)
        plt.legend(loc='upper left')
        plt.grid()
        plt.xlabel("View number")
        plt.ylabel("Probability for true class")
        plt.title("Results for %s %s, sorted by original probabilities" % (basename, variant))
        plt.show()
