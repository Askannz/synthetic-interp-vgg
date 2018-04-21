import sys
import json
import numpy as np
import matplotlib.pyplot as plt

RESULTS_PATH = sys.argv[1]
IMPORTANCES_PATH = sys.argv[2]

results = json.load(open(RESULTS_PATH, "r"))
importance_data = json.load(open(IMPORTANCES_PATH, "r"))

for basename in results["results"].keys():

    classes_ids = results["results"][basename]["classes_ids"]
    original_classes_probas = results["results"][basename]["by_variant"]["original"]["classes_probas"]
    original_classes_probas = np.sum(original_classes_probas, axis=1)

    for variant in results["results"][basename]["by_variant"]:

        if variant == "original":
            continue

        variant_classes_probas = results["results"][basename]["by_variant"][variant]["classes_probas"]
        variant_classes_probas = np.sum(variant_classes_probas, axis=1)

        diffs = original_classes_probas - variant_classes_probas

        render_filenames = results["results"][basename]["renders_filenames"]
        importances_dict = importance_data["results"][basename][variant]
        importances = np.array([importances_dict[f] for f in render_filenames])
        importances_norm = importances / max(1e-7, np.max(importances))

        plt.scatter(importances_norm, diffs)
        plt.ylim([-0.25, 1])
        plt.axhline(0, 0, 1, color="red")
        plt.title(basename + " " + variant)
        plt.show()
