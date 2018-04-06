import os
import sys
import random
import pickle
import matplotlib.pyplot as plt
from utils import make_blended_heatmap

RESULTS_PATH = sys.argv[1]

walk = list(os.walk(RESULTS_PATH))

while True:
    p = random.choice(walk)
    files_list = p[2]
    if len(files_list) == 0:
        continue
    else:
        filename = random.choice(files_list)
        filepath = os.path.join(p[0], filename)
        data = pickle.load(open(filepath, "rb"))

        results = data["results"]
        image = data["image"]

        ordered_results = {"disc": {}, "disc_grad": {}, "square": {}}

        for parameters, heatmap in results:

            shape = parameters["shape"]
            size = parameters["size"]

            ordered_results[shape][size] = heatmap

        sizes = ordered_results["disc"].keys()
        shapes = ordered_results.keys()

        for j, size in enumerate(sorted(sizes)):
            for i, shape in enumerate(shapes):
                heatmap = ordered_results[shape][size]
                blended_heatmap = make_blended_heatmap(image, heatmap)
                plt.subplot(len(sizes), len(shapes), j * len(shapes) + i + 1)
                plt.imshow(blended_heatmap)
                plt.title(str((size, shape)))
        plt.tight_layout()
        plt.show()
