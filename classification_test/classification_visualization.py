import sys
from os.path import join
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

RESULTS_PATH = sys.argv[1]
MODELS_DIR = sys.argv[2]
TOP_SIZE = 5

results = json.load(open(RESULTS_PATH, "r"))

links = json.load(open(join(MODELS_DIR, "links.json"), "r"))

for basename in results["results"].keys():

    classes_ids = results["results"][basename]["classes_ids"]

    classes_probas = np.sum(results["results"][basename]["classes_probas"], axis=1)
    top_indices = results["results"][basename]["top_indices"]

    correct_prediction_mask = []
    for inds in top_indices:
        correct_prediction_mask.append(inds[0] in classes_ids)
    correct_prediction_mask = np.array(correct_prediction_mask)

    indices_sorted = np.argsort(classes_probas)

    classes_probas_sorted = classes_probas[indices_sorted]
    correct_prediction_mask_sorted = correct_prediction_mask[indices_sorted]

    x_indices = np.arange(classes_probas.size)

    nb_correct_predictions = np.sum(correct_prediction_mask.astype(np.int32))
    nb_predictions = correct_prediction_mask.size

    plt.figure()
    plt.bar(x_indices, classes_probas_sorted, 1)
    plt.bar(x_indices[correct_prediction_mask_sorted], classes_probas_sorted[correct_prediction_mask_sorted], 1)
    plt.ylim([0.0, 1.0])
    plt.title(basename + " %d / %d (%.1f%%)" % (nb_correct_predictions, nb_predictions, 100 * nb_correct_predictions / nb_predictions))

    filenames_sorted = []
    for index in indices_sorted:
        filenames_sorted.append(results["results"][basename]["renders_filenames"][index])
    rel_path = links[basename]

    plt.figure()
    for i, f in enumerate(filenames_sorted[:TOP_SIZE]):
        plt.subplot(1, TOP_SIZE, i+1)
        absolute_path = join(MODELS_DIR, rel_path, "renders", "original", f)
        img = cv2.cvtColor(cv2.imread(absolute_path), cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.title("%.3f" % classes_probas_sorted[i])

    plt.figure()
    for i, f in enumerate(filenames_sorted[-TOP_SIZE:]):
        plt.subplot(1, TOP_SIZE, i+1)
        absolute_path = join(MODELS_DIR, rel_path, "renders", "original", f)
        img = cv2.cvtColor(cv2.imread(absolute_path), cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.title("%.3f" % classes_probas_sorted[-TOP_SIZE + i])

    plt.show()
