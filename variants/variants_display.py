import sys
import json
import numpy as np
import matplotlib.pyplot as plt

RESULTS_PATH = sys.argv[1]

importance_data = None
if len(sys.argv) > 2:
    IMPORTANCES_PATH = sys.argv[2]
    importance_data = json.load(open(IMPORTANCES_PATH, "r"))

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

        locs = np.arange(classes_probas.shape[0])

        if importance_data is None:
            classes_probas_sorted = classes_probas[classes_probas[:, 0].argsort()]
            labels = locs + 1
        else:
            render_filenames = results["results"][basename]["renders_filenames"]
            importances_dict = importance_data["results"][basename][variant]
            importances = np.array([importances_dict[f] for f in render_filenames])

            indices_sorted = importances.argsort()[::-1]
            classes_probas_sorted = classes_probas[indices_sorted]
            importances_norm = importances / max(1e-7, np.max(importances))
            labels = ["%2.0f" % (100 * v) for v in importances_norm[indices_sorted]]

        plt.bar(locs, classes_probas_sorted[:, 0], 0.5, label='original', align='center')
        plt.bar(locs + 0.5, classes_probas_sorted[:, 1], 0.5, label=variant, align='center')
        plt.legend(loc='upper left')
        plt.grid()
        plt.xlabel("View number")
        plt.ylabel("Probability for true class")
        plt.xticks(locs + 0.25, labels)
        plt.title("Results for %s %s, sorted by original probabilities" % (basename, variant))
        plt.show()
