import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils import make_blended_heatmap

RESULTS_PATH = sys.argv[1]

walk = list(os.walk(RESULTS_PATH))

for p in walk:
    files_list = p[2]
    if len(files_list) == 0:
        continue
    else:
        for filename in files_list:
            filepath = os.path.join(p[0], filename)
            data = pickle.load(open(filepath, "rb"))

            results = data["results"]
            image = data["image"]

            ordered_results = {}

            for parameters, heatmap in results:

                shape = parameters["shape"]
                size = parameters["size"]
                color = parameters["color"][0]
                if shape != "square" or size != 32:
                    continue
                ordered_results[color] = heatmap

            colors = ordered_results.keys()

            plt.figure()
            for j, color in enumerate(colors):
                heatmap = ordered_results[color]
                blended_heatmap = make_blended_heatmap(image, heatmap)
                plt.subplot(1, len(colors), j + 1)
                plt.imshow(blended_heatmap)
                plt.title(str(color))
            plt.tight_layout()
            plt.figure()
            plt.imshow(image)
            plt.show()
