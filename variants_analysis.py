import sys
import json
import numpy as np

RESULTS_PATH = sys.argv[1]
ANALYSIS_RESULTS_PATH = sys.argv[2]

results = json.load(open(RESULTS_PATH, "r"))

analysis_results = {}

for basename in results["results"].keys():

    classes_ids = results["results"][basename]["classes_ids"]
    original_classes_probas = results["results"][basename]["by_variant"]["original"]["classes_probas"]
    original_classes_probas = np.sum(original_classes_probas, axis=1)

    print(original_classes_probas)

    analysis_results[basename] = {"max_drops":{}}

    for variant in results["results"][basename]["by_variant"]:

        if variant == "original":
            continue

        classes_probas = results["results"][basename]["by_variant"][variant]["classes_probas"]
        classes_probas = np.sum(classes_probas, axis=1)

        print(classes_probas)

        analysis_results[basename]["max_drops"][variant] = np.max(original_classes_probas - classes_probas)

json.dump(analysis_results, open(ANALYSIS_RESULTS_PATH, "w"))
